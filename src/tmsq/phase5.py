"""Phase 5: geometry-aware bit allocation and rounding at a fixed average bit budget (4.0 bits per weight).

Specification: the Phase 5 pre-registration in notes/LOG.md. Everything is per-column (one scale per feature),
tied weights, ``b`` in FP32. ``W`` is a single instance ``[m, n]`` throughout this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .analysis import EvalParams, EvalSet, forward_single, fp_geometry, stratum_stats
from .metrics import mse_from_strata, pearson_columns, r2_from_strata
from .quantize import SCALE_EPS

ALLOC_ARMS = {  # arm -> (geometry key, descending)
    "margin_low": ("margin", False),
    "margin_high": ("margin", True),
    "near_thr": ("near_thr_cal", True),
    "norm": ("norm", True),
}
N_RANDOM = 20
RANDOM_SALT = 5_150_005
CAL_N = 8192
GREEDY_PASSES = 2
GREEDY_ROUNDS = 5
SAMPLE_CHUNK = 2048


# ---------------------------------------------------------------- quantizer with per-column bit widths


def qmax_cols(bits: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(2 ** (np.asarray(bits, dtype=np.int64) - 1) - 1, dtype=torch.float32)


def col_scales(W: torch.Tensor, bits: np.ndarray) -> torch.Tensor:
    """``s_i = max(max|W_i| / qmax(b_i), SCALE_EPS)``, shape ``[1, n]`` (same arithmetic as ``rtn_scales``)."""
    q = qmax_cols(bits)[None, :]
    return (W.abs().amax(dim=0, keepdim=True) / q).clamp_min(SCALE_EPS)


def rtn_mixed(W: torch.Tensor, bits: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-column symmetric RTN with column-specific bit widths -> ``(Q, codes, s)``."""
    q = qmax_cols(bits)[None, :]
    s = col_scales(W, bits)
    codes = torch.clamp(torch.round(W / s), -q, q)
    return s * codes, codes, s


# ---------------------------------------------------------------- allocation rules


def allocation_bits(key: np.ndarray, represented: np.ndarray, descending: bool) -> np.ndarray:
    """Represented columns with a finite key, ranked by key (stable on feature index): first floor(R/2) -> 5 bits,
    last floor(R/2) -> 3 bits, the middle one (R odd) -> 4 bits. Every other column -> 4 bits."""
    n = len(key)
    bits = np.full(n, 4, dtype=np.int64)
    pool = np.flatnonzero(represented & np.isfinite(key))
    k = key[pool]
    order = pool[np.argsort(-k if descending else k, kind="stable")]
    return _split(bits, order)


def random_bits(represented: np.ndarray, train_seed: int, instance: int, draw: int) -> np.ndarray:
    bits = np.full(len(represented), 4, dtype=np.int64)
    rng = np.random.default_rng([RANDOM_SALT, int(train_seed), int(instance), int(draw)])
    return _split(bits, rng.permutation(np.flatnonzero(represented)))


def _split(bits: np.ndarray, order: np.ndarray) -> np.ndarray:
    h = len(order) // 2
    bits[order[:h]] = 5
    bits[order[len(order) - h :]] = 3
    return bits


# ---------------------------------------------------------------- calibration data


def cal_seed(train_seed: int, instance: int) -> int:
    """Generator seed of the calibration set (disjoint from every other stream: tests/test_seed_independence.py)."""
    return 7_000_000_000_003 + 1_000_033 * int(train_seed) + 7919 * int(instance)


def calibration_set(n: int, p: float | np.ndarray, seed: int, N: int = CAL_N) -> torch.Tensor:
    """``N`` natural-distribution samples: ``x_i = 0`` w.p. ``1 - p_i`` else ``Uniform[0, 1]``."""
    g = torch.Generator().manual_seed(seed)
    p_t = torch.as_tensor(np.broadcast_to(np.asarray(p, dtype=np.float64), (n,)).copy(), dtype=torch.float32)
    on = torch.rand(N, n, generator=g) < p_t
    return torch.rand(N, n, generator=g) * on


def near_thr_cal(W: torch.Tensor, b: torch.Tensor, X: torch.Tensor, tau: float, eps: float) -> np.ndarray:
    """``P(|pre_i - tau| <= eps | x_i = 0)`` on the calibration set; NaN where feature i is never inactive."""
    pre = (X @ W.T) @ W + b
    inact = X == 0
    near = ((pre - tau).abs() <= eps) & inact
    cnt = inact.sum(0).double()
    out = near.sum(0).double() / cnt
    return torch.where(cnt > 0, out, torch.full_like(out, float("nan"))).numpy()


# ---------------------------------------------------------------- greedy rounding (arms e and e2)


def objective_terms(pre: torch.Tensor, X: torch.Tensor, imp: torch.Tensor, objective: str) -> torch.Tensor:
    """Per-(sample, feature) objective: leak = I relu(pre)^2 1(x = 0); loss = I (x - relu(pre))^2."""
    r = torch.relu(pre)
    if objective == "leak":
        return imp * r**2 * (X == 0)
    if objective == "loss":
        return imp * (X - r) ** 2
    raise ValueError(objective)


def objective_value(Q: torch.Tensor, b: torch.Tensor, X: torch.Tensor, imp: torch.Tensor, objective: str) -> float:
    pre = (X @ Q.T) @ Q + b
    return float(objective_terms(pre, X, imp, objective).sum() / X.shape[0])


def greedy_round(
    W: torch.Tensor,
    b: torch.Tensor,
    X: torch.Tensor,
    importance: np.ndarray,
    objective: str,
    columns: np.ndarray,
    bits: int = 4,
    passes: int = GREEDY_PASSES,
    rounds: int = GREEDY_ROUNDS,
) -> tuple[torch.Tensor, dict]:
    """Start from per-column RTN codes; for each column in ``columns`` (in order), evaluate every single-entry flip
    to the other neighbouring grid point of ``W/s`` in parallel and accept the best one if it strictly lowers the
    calibration objective; up to ``rounds`` accepted flips per column per pass. Entries that are exactly on the
    grid (e.g. the column maximum) are never flipped, so every code stays within one step of ``W/s`` and in
    ``[-qmax, qmax]``. Pre-activations are updated exactly (tied weights; see the pre-registration).
    Returns ``(Q float32 [m, n] on the grid s * code, info)``."""
    m, n = W.shape
    bits_v = np.full(n, bits, dtype=np.int64)
    _, codes, s = rtn_mixed(W, bits_v)
    qmax = float(2 ** (bits - 1) - 1)
    u = (W / s).double()
    lo, hi = torch.floor(u).clamp(-qmax, qmax), torch.ceil(u).clamp(-qmax, qmax)
    codes = codes.double().clone()
    s64 = s.double()[0]
    Xd, bd = X.double(), b.double()
    imp = torch.as_tensor(np.asarray(importance, dtype=np.float64))
    Q = s64[None, :] * codes
    H = Xd @ Q.T  # [N, m]
    PRE = H @ Q + bd  # [N, n]
    N = Xd.shape[0]
    J0 = float(objective_terms(PRE, Xd, imp, objective).sum() / N)
    J = J0
    n_flips = 0
    for _ in range(passes):
        for i in columns:
            i = int(i)
            for _r in range(rounds):
                alt = torch.where(codes[:, i] == lo[:, i], hi[:, i], lo[:, i])
                valid = alt != codes[:, i]
                if not bool(valid.any()):
                    break
                delta = s64[i] * (alt - codes[:, i])  # [m]
                d = _flip_deltas(PRE, H, Q, Xd, imp, i, delta, objective) / N
                d = torch.where(valid, d, torch.full_like(d, float("inf")))
                k = int(torch.argmin(d))
                if not float(d[k]) < -(1e-12 + 1e-10 * abs(J)):
                    break
                _apply_flip(PRE, H, Q, Xd, i, k, float(delta[k]))
                codes[k, i] = alt[k]
                J += float(d[k])
                n_flips += 1
    Qf = (s * codes.float()).float()
    J_final = objective_value(Qf.double(), bd, Xd, imp, objective)
    return Qf, {"J0": J0, "J_tracked": J, "J_final": J_final, "n_flips": n_flips, "codes": codes.float()}


def _flip_deltas(PRE, H, Q, X, imp, i, delta, objective) -> torch.Tensor:
    """Objective change (summed over samples) of flipping ``Q[k, i] += delta[k]``, for every k at once."""
    m, n = Q.shape
    xi = X[:, i]
    # feature i itself, all samples: pre_i += delta h_k + (Q_ki + delta) delta x_i
    new_i = PRE[None, :, i] + delta[:, None] * H.T + ((Q[:, i] + delta) * delta)[:, None] * xi[None, :]
    old_i = objective_terms(PRE[:, i], X[:, i], imp[i], objective)
    out = (objective_terms(new_i, X[None, :, i], imp[i], objective) - old_i[None, :]).sum(1)
    # other features j != i, only samples with x_i > 0: pre_j += Q_kj delta x_i
    S = torch.nonzero(xi > 0).flatten()
    mask = torch.ones(n, dtype=torch.bool)
    mask[i] = False
    for c0 in range(0, len(S), SAMPLE_CHUNK):
        sel = S[c0 : c0 + SAMPLE_CHUNK]
        pre_s, x_s = PRE[sel][:, mask], X[sel][:, mask]
        new = pre_s[None] + (delta[:, None, None] * xi[sel][None, :, None]) * Q[:, mask][:, None, :]
        f_new = objective_terms(new, x_s[None], imp[mask], objective)
        f_old = objective_terms(pre_s, x_s, imp[mask], objective)
        out = out + (f_new - f_old[None]).sum((1, 2))
    return out


def _apply_flip(PRE, H, Q, X, i, k, delta) -> None:
    dh = delta * X[:, i]
    PRE[:, i] += delta * H[:, k] + (Q[k, i] + delta) * dh
    others = torch.ones(Q.shape[1], dtype=torch.bool)
    others[i] = False
    PRE[:, others] += dh[:, None] * Q[k, others][None, :]
    H[:, k] += dh
    Q[k, i] += delta


# ---------------------------------------------------------------- evaluation on THE stratified eval set


def fp_reference(W: torch.Tensor, b: torch.Tensor, es: EvalSet, params: EvalParams, p, importance) -> dict:
    pre, out = forward_single(W, W, b, es.X, True, params.forward_chunk)
    fp = stratum_stats(es, out, pre, params)
    geom = fp_geometry(W, b, p, params)
    r2 = r2_from_strata(p, fp.cond_mse, fp.leak_energy)
    eligible = geom["represented"] & (np.nan_to_num(r2, nan=-np.inf) >= params.eligible_r2)
    mse = mse_from_strata(p, fp.cond_mse, fp.leak_energy)
    return {"W": W, "stats": fp, "geom": geom, "r2": r2, "eligible": eligible, "mse": mse,
            "loss": float((np.asarray(importance) * mse).sum())}  # fmt: skip


def evaluate_q(Wq: torch.Tensor, b: torch.Tensor, es: EvalSet, params: EvalParams, p, importance, ref: dict):
    """-> (model-level dict, per-feature DataFrame) for one quantized weight matrix (tied, b in FP32)."""
    _, out = forward_single(Wq, Wq, b, es.X, False, params.forward_chunk)
    q = stratum_stats(es, out, None, params)
    fp, el = ref["stats"], ref["eligible"]
    mse_q = mse_from_strata(p, q.cond_mse, q.leak_energy)
    r2_q = r2_from_strata(p, q.cond_mse, q.leak_energy)
    dr2 = ref["r2"] - r2_q
    dfalse = q.false_act - fp.false_act
    corr = pearson_columns(fp.act_out, q.act_out)
    comp = (corr >= params.survival_threshold) & (dr2 <= params.composite_dr2 + 1e-9)
    comp = comp & (dfalse <= params.composite_dfalse + 1e-9)
    comp_f = np.where(np.isnan(dfalse), np.nan, comp.astype(float))
    loss_q = float((np.asarray(importance) * mse_q).sum())
    n_el = int(el.sum())
    model = {
        "loss_fp": ref["loss"], "loss_q": loss_q, "r": loss_q / ref["loss"] - 1.0, "n_eligible": n_el,
        "composite_frac": float(np.nanmean(comp_f[el])) if n_el and not np.all(np.isnan(comp_f[el])) else np.nan,
        "worst_dR2": float(np.nanmax(dr2[el])) if n_el else np.nan,
        "mean_dR2": float(np.nanmean(dr2[el])) if n_el else np.nan,
        "frob_err": float(torch.linalg.norm(Wq - ref["W"]).item()),
    }  # fmt: skip
    feats = pd.DataFrame({"feature": np.arange(len(dr2)), "eligible": el, "dR2": dr2,
                          "dcond_mse": q.cond_mse - fp.cond_mse, "dleak_energy": q.leak_energy - fp.leak_energy,
                          "dfalse_act": dfalse})  # fmt: skip
    return model, feats

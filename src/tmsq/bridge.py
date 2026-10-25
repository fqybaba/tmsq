"""Phase 6b (real-model bridge): quantizers, streamed statistics, predictors and the registered tests.

Pure torch / numpy / scikit-learn, so the main venv can test it. The model- and SAE-loading code lives in
``tmsq.experiments.bridge`` (``.venv-bridge`` only). The plan is ``notes/LOG.md`` Phase 6b (D1–D7); the
pre-registration fixes the constants below before any quantized run.

Notation for the SAE (``pythia-70m-deduped-res-sm``, ``blocks.4.hook_resid_post``):
``z = (x - b_dec) @ W_enc + b_enc`` (pre-activation), ``f = ReLU(z)``, ``e_j = W_enc[:, j]``, ``d_k = W_dec[k]``.
With ``x - b_dec ≈ sum_k f_k d_k`` the pre-activation of j is ``sum_k f_k (e_j · d_k) + b_enc_j``, the analog of
the toy model's ``sum_k (W_j · W_k) x_k + b_j``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ACTIVE_RATE = 0.001  # Duan: a feature is active if its FP firing rate over all tokens exceeds 0.001
SURVIVE = 0.9  # Duan: survived iff c_j > 0.9 (strict)
DAMAGED = 0.5  # Duan: damaged iff c_j < 0.5
EPS_FLOOR = 1e-8  # Duan's code floors the four predictors' inputs at 1e-8
NEAR_FRAC = 0.05  # near-threshold band and leak threshold, as a fraction of mu_j+ (the toy model's 0.05)
VAR_FLOOR = 1e-12  # Duan's code clamps var_x * var_y at 1e-12


# ----------------------------------------------------------------------------------------------- quantizers


def qrange(bits: int) -> tuple[int, int]:
    """Duan's nominal signed range ``[-2^(b-1), 2^(b-1) - 1]``."""
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def rtn_lastdim(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Code arm (Duan's released script): one scale per index of the LAST dim, reduced over all other dims.

    ``s = max|w| / qmax`` over each group (a zero scale becomes 1), ``Q = s * clamp(round(w / s), qmin, qmax)``.
    For TransformerLens ``W_Q/W_K/W_V`` ([n_heads, d_model, d_head]) this shares d_head scales across heads.
    """
    qmin, qmax = qrange(bits)
    s = w.abs().amax(dim=tuple(range(w.ndim - 1)), keepdim=True) / qmax
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.clamp(torch.round(w / s), qmin, qmax) * s


def rtn_rows(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Stated arm: true per-output-channel RTN on an ``nn.Linear`` weight ``[out, in]`` (one scale per row)."""
    if w.ndim != 2:
        raise ValueError("rtn_rows expects an nn.Linear weight [out, in]")
    qmin, qmax = qrange(bits)
    s = w.abs().amax(dim=1, keepdim=True) / qmax
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.clamp(torch.round(w / s), qmin, qmax) * s


def group_norms(e: torch.Tensor, arm: str) -> torch.Tensor:
    """Norm of ``e`` over each scale group: per last-dim index (code arm) or per row (stated arm)."""
    if arm == "code":
        return e.pow(2).sum(dim=tuple(range(e.ndim - 1)), keepdim=True).sqrt()
    if arm == "stated":
        return e.pow(2).sum(dim=1, keepdim=True).sqrt()
    raise ValueError(arm)


def noise_matched(w: torch.Tensor, q: torch.Tensor, arm: str, gen: torch.Generator) -> torch.Tensor:
    """Matched-noise control: each scale group's rounding error replaced by a Gaussian direction of the same norm."""
    e = q - w
    g = torch.randn(w.shape, generator=gen, dtype=w.dtype)
    gn = group_norms(g, arm)
    return w + g * (group_norms(e, arm) / torch.where(gn == 0, torch.ones_like(gn), gn))


# ----------------------------------------------------------------------------------------- streamed stats


class FPStats:
    """Per-feature FP statistics over all tokens, streamed: firing count, sum, max (Duan's predictors)."""

    def __init__(self, d: int) -> None:
        self.n = 0
        self.fire = np.zeros(d, np.float64)
        self.sum = np.zeros(d, np.float64)
        self.max = np.zeros(d, np.float64)

    def update(self, f: torch.Tensor) -> None:
        """``f``: [T, d] activations (ReLU output)."""
        self.n += f.shape[0]
        self.fire += (f > 0).sum(0).double().numpy()
        self.sum += f.sum(0, dtype=torch.float64).numpy()
        self.max = np.maximum(self.max, f.amax(0).double().numpy())

    def table(self) -> dict[str, np.ndarray]:
        rate = self.fire / self.n
        mean = self.sum / self.n
        mu_plus = np.divide(self.sum, self.fire, out=np.full_like(self.sum, np.nan), where=self.fire > 0)
        return {"rate": rate, "mean": mean, "peak": self.max.copy(), "mu_plus": mu_plus}


class NearThrStats:
    """Fraction of tokens with ``-NEAR_FRAC * mu_j+ < z_j <= 0`` (off, but within 5% of mu_j+ of switching on)."""

    def __init__(self, delta: np.ndarray) -> None:
        self.delta = torch.as_tensor(delta, dtype=torch.float32)
        self.n = 0
        self.count = np.zeros(len(delta), np.float64)

    def update(self, z: torch.Tensor) -> None:
        self.n += z.shape[0]
        self.count += ((z > -self.delta) & (z <= 0)).sum(0).double().numpy()

    def value(self) -> np.ndarray:
        return self.count / self.n


SUM_KEYS = ("n", "sx", "sy", "sxx", "syy", "sxy", "n_pos_y", "n_act", "sy_act", "syy_act", "n_leak")


class PairStats:
    """Streamed sums for FP activations ``x`` and variant activations ``y`` per token group.

    Unshifted sums, computed and accumulated in float64 as in Duan's code. Groups are additive, so any union of
    groups gives the statistics of that token subset.
    """

    def __init__(self, n_groups: int, leak_thr: np.ndarray) -> None:
        d = len(leak_thr)
        self.leak_thr = torch.as_tensor(leak_thr, dtype=torch.float64)
        self.s = {k: np.zeros((n_groups, d), np.float64) for k in SUM_KEYS}

    def update(self, g: int, x: torch.Tensor, y: torch.Tensor) -> None:
        """``x``, ``y``: [T, d] FP and variant activations (ReLU outputs) of the same tokens, all in group ``g``."""
        s = self.s
        x, y = x.double(), y.double()
        act = x > 0
        s["n"][g] += x.shape[0]
        s["sx"][g] += x.sum(0).numpy()
        s["sy"][g] += y.sum(0).numpy()
        s["sxx"][g] += (x * x).sum(0).numpy()
        s["syy"][g] += (y * y).sum(0).numpy()
        s["sxy"][g] += (x * y).sum(0).numpy()
        s["n_pos_y"][g] += (y > 0).sum(0).double().numpy()
        ya = torch.where(act, y, torch.zeros_like(y))
        s["n_act"][g] += act.sum(0).double().numpy()
        s["sy_act"][g] += ya.sum(0).numpy()
        s["syy_act"][g] += (ya * ya).sum(0).numpy()
        s["n_leak"][g] += ((~act) & (y > self.leak_thr)).sum(0).double().numpy()

    def merged(self, groups: list[int] | None = None) -> dict[str, np.ndarray]:
        idx = slice(None) if groups is None else groups
        return {k: v[idx].sum(0) for k, v in self.s.items()}


def pearson(n: np.ndarray, sx, sy, sxx, syy, sxy) -> np.ndarray:
    """Pearson r from sums; ``var_x * var_y`` clamped at ``VAR_FLOOR`` as in Duan's code."""
    cxy = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    r = cxy / np.sqrt(np.maximum(vx * vy, VAR_FLOOR))
    return np.clip(r, -1.0, 1.0)


def outcomes(m: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-feature outcomes from merged sums: Duan's all-token ``corr``, active-stratum ``corr_act``, ``dfalse``.

    A variant feature that never fires (``y == 0`` everywhere) gets ``corr = 0`` exactly (Duan: a constant
    compressed activation has zero covariance and is damaged); ``corr_act = 0`` exactly when ``y == 0`` on every
    FP-active token. Because ``x = 0`` outside the active stratum, its x-side sums equal the all-token ones.
    """
    n, na = m["n"], m["n_act"]
    corr = np.where(m["n_pos_y"] > 0, pearson(n, m["sx"], m["sy"], m["sxx"], m["syy"], m["sxy"]), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ca = pearson(na, m["sx"], m["sy_act"], m["sxx"], m["syy_act"], m["sxy"])
        corr_act = np.where(na >= 2, np.where(m["sy_act"] > 0, ca, 0.0), np.nan)
        dfalse = m["n_leak"] / (n - na)
    return {"corr": corr, "corr_act": corr_act, "dfalse": dfalse}


def label(corr: np.ndarray) -> np.ndarray:
    """0 = damaged (< 0.5), 1 = degraded, 2 = survived (> 0.9)."""
    return np.where(corr > SURVIVE, 2, np.where(corr < DAMAGED, 0, 1))


# ------------------------------------------------------------------------------------------- predictors


def duan_predictors(rate: np.ndarray, mean: np.ndarray, peak: np.ndarray) -> dict[str, np.ndarray]:
    """Duan's four FP statistics as implemented in his code (floors at 1e-8; log concentration floored at 0)."""
    r = np.maximum(rate, EPS_FLOOR)
    mu = np.maximum(mean, EPS_FLOOR)
    pk = np.maximum(peak, EPS_FLOOR)
    return {
        "rarity": -np.log(r),
        "log_mean": np.log(mu),
        "log_peak": np.log(pk),
        "log_conc": np.maximum(np.log(pk / mu), 0.0),
    }


def margin_analog(
    W_enc: torch.Tensor, W_dec: torch.Tensor, b_enc: torch.Tensor, mu_plus: np.ndarray, active: np.ndarray,
    bias: torch.Tensor | None = None, chunk: int = 1024,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Margin analog for the active features j (NaN elsewhere).

    ``interf_j = max(0, max_{k != j, k active} (e_j · d_k) * mu_k+)`` (feature k at its typical active size),
    ``margin_j = (-bias_j - interf_j) / mu_j+`` with ``bias = b_enc`` by default (the zero-feature point is
    ``x = b_dec`` because the SAE subtracts ``b_dec`` before encoding).
    """
    bias = b_enc if bias is None else bias
    act = np.flatnonzero(active)
    mu = torch.as_tensor(mu_plus[act], dtype=torch.float32)
    D = W_dec[act] * mu[:, None]  # [n_act, d_in]: d_k scaled by mu_k+
    interf = np.full(W_enc.shape[1], np.nan)
    for i in range(0, len(act), chunk):
        cols = act[i : i + chunk]
        G = D @ W_enc[:, cols]  # [n_act, c]: (e_j · d_k) mu_k+
        G[torch.arange(i, i + len(cols)), torch.arange(len(cols))] = -torch.inf  # exclude k = j
        interf[cols] = torch.clamp(G.amax(0), min=0).double().numpy()
    margin = np.full(W_enc.shape[1], np.nan)
    margin[act] = (-bias.double().numpy()[act] - interf[act]) / mu_plus[act]
    return {"interf": interf, "margin": margin}


def interference_sum(
    W_enc: torch.Tensor, W_dec: torch.Tensor, mu_plus: np.ndarray, rate: np.ndarray, active: np.ndarray,
    chunk: int = 1024,
) -> np.ndarray:  # fmt: skip
    """Density-weighted positive interference ``sum_{k != j active} rate_k (e_j · d_k)+ mu_k+`` (plan D3)."""
    act = np.flatnonzero(active)
    w = torch.as_tensor(rate[act] * mu_plus[act], dtype=torch.float32)
    out = np.full(W_enc.shape[1], np.nan)
    for i in range(0, len(act), chunk):
        cols = act[i : i + chunk]
        G = torch.clamp(W_dec[act] @ W_enc[:, cols], min=0)
        G[torch.arange(i, i + len(cols)), torch.arange(len(cols))] = 0
        out[cols] = (w @ G).double().numpy()
    return out


# ---------------------------------------------------------------------------------------------- tests


@dataclass(frozen=True)
class Registered:
    """Constants of the registered tests (fixed in the pre-registration)."""

    bits: int = 6
    dauc_min: float = 0.02
    rho_min: float = 0.10
    cv_folds: int = 5
    cv_repeats: int = 20
    perm_repeats: int = 5
    n_perm_a: int = 1000
    n_perm_b: int = 10000
    n_strata: int = 10
    alpha: float = 0.05
    seed: int = 6060


def _zscore(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / (s if s > 0 else 1.0)


def cv_folds(y: np.ndarray, repeats: int, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold splits, repeated with seeds ``seed + r``."""
    from sklearn.model_selection import StratifiedKFold

    out = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        out.extend(skf.split(np.zeros(len(y)), y))
    return out


def cv_auc(y: np.ndarray, X: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Mean held-out AUC of an L2 logistic regression (C = 1), standardized within each training fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = X.reshape(len(y), -1)
    aucs = []
    for tr, te in splits:
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(C=1.0, max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function((X[te] - mu) / sd)))
    return float(np.mean(aucs))


def strata(control: np.ndarray, n_strata: int) -> np.ndarray:
    """Quantile bins of ``control`` (ties broken by rank) used for stratified permutation.

    1-D: ``n_strata`` bins. 2-D ``[n, c]``: ``max(2, n_strata // c)`` bins per column, crossed (e.g. 5 x 5).
    """
    control = np.asarray(control)
    if control.ndim == 1:
        ranks = np.argsort(np.argsort(control, kind="stable"), kind="stable")
        return (ranks * n_strata) // len(control)
    per = max(2, n_strata // control.shape[1])
    cells = np.zeros(len(control), dtype=np.int64)
    for j in range(control.shape[1]):
        cells = cells * per + strata(control[:, j], per)
    return cells


def permute_within(a: np.ndarray, bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = a.copy()
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        out[idx] = a[rng.permutation(idx)]
    return out


def _rank(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(a)


class CellProjector:
    """Residuals after a fit with cell fixed effects plus within-cell linear slopes on the control rank(s).

    The cells are the permutation strata, so the statistic conditions on exactly what the permutation null
    holds fixed (a single cell gives one global linear fit).
    """

    def __init__(self, control: np.ndarray, cells: np.ndarray) -> None:
        C = np.asarray(control, dtype=np.float64).reshape(len(cells), -1)
        R = np.column_stack([_rank(C[:, j]) for j in range(C.shape[1])])
        self.groups = []
        for k in np.unique(cells):
            idx = np.flatnonzero(cells == k)
            X = np.column_stack([np.ones(len(idx)), R[idx]])
            u, s, _ = np.linalg.svd(X, full_matrices=False)  # orthonormal basis of the column space
            self.groups.append((idx, u[:, s > 1e-10 * s.max()]))

    def resid(self, v: np.ndarray) -> np.ndarray:
        out = np.asarray(v, dtype=np.float64).copy()
        for idx, q in self.groups:
            vk = out[idx]
            out[idx] = vk - q @ (q.T @ vk)
        return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    den = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def partial_spearman(a: np.ndarray, b: np.ndarray, control: np.ndarray, cells: np.ndarray | None = None) -> float:
    """Spearman of ``a`` and ``b`` given ``control`` ([n] or [n, c]): Pearson of rank residuals after a fit with
    cell fixed effects plus within-cell slopes on the control rank(s) (``cells=None``: one global linear fit)."""
    cells = np.zeros(len(a), dtype=np.int64) if cells is None else cells
    proj = CellProjector(control, cells)
    return _corr(proj.resid(_rank(a)), proj.resid(_rank(b)))


SPLINE_KNOTS = 6


class FoldDesign:
    """CV design with a flexible control base: an additive cubic spline basis (``SPLINE_KNOTS`` knots per control
    column) fit within each training fold. The base model is logistic(spline(controls)); the full model appends
    the predictor, standardized within the training fold. L2 logistic regression with C = 1."""

    def __init__(self, y: np.ndarray, control: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
        from sklearn.preprocessing import SplineTransformer

        self.y = y
        C = np.asarray(control, dtype=np.float64).reshape(len(y), -1)
        self.folds = []
        for tr, te in splits:
            st = SplineTransformer(n_knots=SPLINE_KNOTS, degree=3).fit(C[tr])
            self.folds.append((tr, te, st.transform(C[tr]), st.transform(C[te])))

    def auc(self, pred: np.ndarray | None = None) -> float:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        aucs = []
        for tr, te, btr, bte in self.folds:
            if pred is None:
                xtr, xte = btr, bte
            else:
                mu, sd = pred[tr].mean(), pred[tr].std()
                sd = sd if sd > 0 else 1.0
                xtr = np.column_stack([btr, (pred[tr] - mu) / sd])
                xte = np.column_stack([bte, (pred[te] - mu) / sd])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(xtr, self.y[tr])
            aucs.append(roc_auc_score(self.y[te], clf.decision_function(xte)))
        return float(np.mean(aucs))


def _cells(C: np.ndarray, reg: Registered) -> np.ndarray:
    return strata(C[:, 0] if C.shape[1] == 1 else C, reg.n_strata)


def family_a(y: np.ndarray, control: np.ndarray, pred: np.ndarray, reg: Registered) -> dict[str, float]:
    """Does ``pred`` add to ``control`` (log peak, or [n, c] controls) in predicting survival?

    dAUC of the spline-base design (``FoldDesign``) over ``cv_repeats`` x ``cv_folds`` CV, and a stratified
    permutation p: the statistic is the dAUC on the first ``perm_repeats`` repeats, recomputed for the observed
    predictor and for each within-cell permutation of it (the base model is unchanged by permutation).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import SplineTransformer

    C = np.asarray(control, dtype=np.float64).reshape(len(y), -1)
    splits = cv_folds(y, reg.cv_repeats, reg.cv_folds, reg.seed)
    full = FoldDesign(y, C, splits)
    a0, a1 = full.auc(), full.auc(pred)
    sub = FoldDesign(y, C, splits[: reg.perm_repeats * reg.cv_folds])
    b0 = sub.auc()
    obs = sub.auc(pred) - b0
    bins = _cells(C, reg)
    rng = np.random.default_rng(reg.seed)
    null = np.array([sub.auc(permute_within(pred, bins, rng)) - b0 for _ in range(reg.n_perm_a)])
    basis = SplineTransformer(n_knots=SPLINE_KNOTS, degree=3).fit_transform(C)
    coef = LogisticRegression(C=1.0, max_iter=2000).fit(np.column_stack([basis, _zscore(pred)]), y).coef_[0]
    return {
        "auc_base": a0, "auc_full": a1, "dauc": a1 - a0, "dauc_perm_stat": obs,
        "p": float((1 + (null >= obs).sum()) / (1 + reg.n_perm_a)), "null_mean": float(null.mean()),
        "null_sd": float(null.std()), "coef_pred": float(coef[-1]),
    }  # fmt: skip


def family_b(dfalse: np.ndarray, control: np.ndarray, pred: np.ndarray, reg: Registered) -> dict[str, float]:
    """Cell-consistent partial Spearman of ``pred`` with new false activations given ``control``; two-sided p from
    permutations of ``pred`` within the same cells."""
    C = np.asarray(control, dtype=np.float64).reshape(len(pred), -1)
    cells = _cells(C, reg)
    proj = CellProjector(C, cells)
    rb = proj.resid(_rank(dfalse))
    ra = _rank(pred)
    obs = _corr(proj.resid(ra), rb)
    rng = np.random.default_rng(reg.seed + 1)
    null = np.array([_corr(proj.resid(permute_within(ra, cells, rng)), rb) for _ in range(reg.n_perm_b)])
    return {
        "rho": obs, "p": float((1 + (np.abs(null) >= abs(obs)).sum()) / (1 + reg.n_perm_b)),
        "null_mean": float(null.mean()), "null_sd": float(null.std()),
    }  # fmt: skip


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def decide_a(r: dict[str, float], p_holm: float, reg: Registered, predicted_sign: int) -> str:
    """Family A verdict. ``predicted_sign``: -1 (T: more near-threshold mass -> lower survival) or 0 (two-sided)."""
    if not (r["dauc"] >= reg.dauc_min and p_holm < reg.alpha):
        return "no added information beyond peak"
    if predicted_sign == 0:
        return f"adds information beyond peak (coefficient {'+' if r['coef_pred'] > 0 else '-'})"
    return "supported" if np.sign(r["coef_pred"]) == predicted_sign else "adds information, opposite to prediction"


def decide_b(r: dict[str, float], p_holm: float, reg: Registered, predicted_sign: int) -> str:
    """Family B verdict. ``predicted_sign``: +1 (T: more near-threshold mass -> more leakage) or 0 (two-sided)."""
    if not (abs(r["rho"]) >= reg.rho_min and p_holm < reg.alpha):
        return "no association beyond peak"
    if predicted_sign == 0:
        return f"associated beyond peak ({'+' if r['rho'] > 0 else '-'})"
    return "supported" if np.sign(r["rho"]) == predicted_sign else "associated, opposite to prediction"

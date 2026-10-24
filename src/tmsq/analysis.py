"""Phase 2 evaluation: stratified eval sets, per-feature damage metrics and the quantization sweep loop.

Everything here works on ONE model instance at a time (``W [m, n]``, ``b [n]``) so the eval set of an
instance is built once, the FP model is run once, and every sweep configuration reuses both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .data import FeatureSampler
from .metrics import (
    DIM_MIN_NORM,
    REPRESENTED_NORM,
    d_plus,
    dimensionality,
    feature_norms,
    leak_mean,
    margin,
    margin_tau,
    max_overlap,
    mse_from_strata,
    peak_ratio,
    pearson_columns,
    perturbation_covariates,
    r2_from_strata,
    superposition_score,
    var_x,
)
from .quantize import QuantConfig, apply_quant


@dataclass
class EvalParams:
    n_active: int = 2000
    n_inactive: int = 20000
    n_nat_min: int = 40000
    n_nat_max: int = 120000
    tau: float = 0.05
    near_eps: float = 0.05
    seed_base: int = 10_000
    survival_threshold: float = 0.9
    composite_dr2: float = 0.1
    composite_dfalse: float = 0.05
    eligible_r2: float = 0.5
    forward_chunk: int = 65536


@dataclass
class EvalSet:
    """Fixed eval set of one instance: rows ``X`` and per-feature stratum row indices."""

    X: torch.Tensor  # [R, n]
    idx_act: torch.Tensor  # [n_active, n]
    idx_inact: torch.Tensor | None  # [n_inactive_eval, n] or None when p = 1
    p: float | np.ndarray  # scalar (settings A-C) or per-feature vector (setting D)
    n_active_eval: int
    n_inactive_eval: int
    n_forced_active: int
    n_natural: int
    has_inactive: np.ndarray | None = None  # [n] bool; False for p_i = 1 (setting D); None = all True
    n_forced_inactive: int = 0


def eval_seed(train_seed: int, instance: int, params: EvalParams) -> int:
    """Distinct from the training stream (``FeatureSampler(seed=train_seed + 1)``)."""
    return params.seed_base + 7919 * int(train_seed) + int(instance)


def build_eval_set(n: int, p: float, seed: int, params: EvalParams) -> EvalSet:
    """Natural batch plus forced-active batches where needed (notes/LOG.md, Phase 2 plan)."""
    S = 1.0 - p
    if S > 0:
        n_nat = int(min(params.n_nat_max, max(params.n_nat_min, math.ceil(1.1 * params.n_inactive / S))))
    else:
        n_nat = params.n_nat_min
    sampler = FeatureSampler(n, float(p), n_instances=1, seed=seed, n_chunks=1)
    X = sampler.sample(n_nat)[0]
    active = X > 0
    blocks = [X]
    idx_act = torch.empty(params.n_active, n, dtype=torch.long)
    offset, n_forced = n_nat, 0
    for i in range(n):
        rows = torch.nonzero(active[:, i]).squeeze(1)
        if rows.numel() >= params.n_active:
            idx_act[:, i] = rows[: params.n_active]
        else:
            Xf = sampler.sample_conditional(params.n_active, feature=i, active=True)[0]
            blocks.append(Xf)
            idx_act[:, i] = torch.arange(offset, offset + params.n_active)
            offset += params.n_active
            n_forced += 1
    if p >= 1.0:
        idx_inact, n_inact = None, 0
    else:
        counts = [int((~active[:, i]).sum()) for i in range(n)]
        n_inact = min(params.n_inactive, min(counts))
        idx_inact = torch.empty(n_inact, n, dtype=torch.long)
        for i in range(n):
            idx_inact[:, i] = torch.nonzero(~active[:, i]).squeeze(1)[:n_inact]
    return EvalSet(torch.cat(blocks, 0), idx_act, idx_inact, float(p), params.n_active, n_inact, n_forced, n_nat)


def build_eval_set_vec(n: int, p: np.ndarray, seed: int, params: EvalParams) -> EvalSet:
    """Setting D: per-feature frequencies. A natural batch of ``n_nat_min`` rows; per feature, the active stratum
    is the first ``n_active`` natural active rows or a forced-active batch, and the inactive stratum is the first
    ``n_inactive`` natural inactive rows or a forced-inactive batch (no common minimum, so frequent head features
    do not shrink everyone's stratum). Features with ``p_i = 1`` have no inactive stratum (``has_inactive`` False).
    """
    p = np.asarray(p, dtype=np.float64)
    n_nat = params.n_nat_min
    sampler = FeatureSampler(n, torch.as_tensor(p, dtype=torch.float32), n_instances=1, seed=seed, n_chunks=1)
    X = sampler.sample(n_nat)[0]
    active = X > 0
    blocks = [X]
    offset = n_nat
    idx_act = torch.empty(params.n_active, n, dtype=torch.long)
    idx_inact = torch.zeros(params.n_inactive, n, dtype=torch.long)
    has_inactive = p < 1.0
    n_forced_act = n_forced_inact = 0
    for i in range(n):
        rows = torch.nonzero(active[:, i]).squeeze(1)
        if rows.numel() >= params.n_active:
            idx_act[:, i] = rows[: params.n_active]
        else:
            blocks.append(sampler.sample_conditional(params.n_active, feature=i, active=True)[0])
            idx_act[:, i] = torch.arange(offset, offset + params.n_active)
            offset += params.n_active
            n_forced_act += 1
        if not has_inactive[i]:
            continue
        rows_in = torch.nonzero(~active[:, i]).squeeze(1)
        if rows_in.numel() >= params.n_inactive:
            idx_inact[:, i] = rows_in[: params.n_inactive]
        else:
            blocks.append(sampler.sample_conditional(params.n_inactive, feature=i, active=False)[0])
            idx_inact[:, i] = torch.arange(offset, offset + params.n_inactive)
            offset += params.n_inactive
            n_forced_inact += 1
    return EvalSet(torch.cat(blocks, 0), idx_act, idx_inact, p, params.n_active, params.n_inactive, n_forced_act,
                   n_nat, has_inactive=has_inactive, n_forced_inactive=n_forced_inact)  # fmt: skip


def forward_single(
    W_enc: torch.Tensor, W_dec: torch.Tensor, b: torch.Tensor, X: torch.Tensor, want_pre: bool, chunk: int
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """``pre = (X W_enc^T) W_dec + b``, ``out = ReLU(pre)`` for ``W [m, n]``, ``X [R, n]``, in row chunks."""
    pres, outs = [], []
    for start in range(0, X.shape[0], chunk):
        x = X[start : start + chunk]
        pre = (x @ W_enc.T) @ W_dec + b
        outs.append(torch.relu(pre))
        if want_pre:
            pres.append(pre)
    return (torch.cat(pres, 0) if want_pre else None), torch.cat(outs, 0)


@dataclass
class StratumStats:
    cond_mse: np.ndarray
    peak_out: np.ndarray
    p99_out: np.ndarray
    leak_energy: np.ndarray
    false_act: np.ndarray
    near_thr: np.ndarray | None
    act_out: np.ndarray = field(repr=False)  # [n_active, n], kept for the survival correlation


def stratum_stats(es: EvalSet, out: torch.Tensor, pre: torch.Tensor | None, params: EvalParams) -> StratumStats:
    n = es.X.shape[1]
    ar = torch.arange(n)
    act_x = es.X[es.idx_act, ar]
    act_out = out[es.idx_act, ar]
    cond_mse = ((act_x - act_out) ** 2).mean(0)
    peak = act_out.amax(0)
    p99 = torch.quantile(act_out, 0.99, dim=0)
    nan = torch.full((n,), float("nan"))
    if es.idx_inact is None:
        leak, false_act, near = nan, nan, None
    else:
        inact = out[es.idx_inact, ar]
        leak = (inact**2).mean(0)
        false_act = (inact > params.tau).double().mean(0)
        near = None
        if pre is not None:
            near = ((pre[es.idx_inact, ar] - params.tau).abs() <= params.near_eps).double().mean(0).numpy()
        if es.has_inactive is not None:  # setting D: p_i = 1 features have no inactive stratum
            mask = torch.as_tensor(~es.has_inactive)
            leak = leak.masked_fill(mask, float("nan"))
            false_act = false_act.masked_fill(mask, float("nan"))
            if near is not None:
                near = np.where(es.has_inactive, near, np.nan)
    return StratumStats(
        cond_mse.numpy().astype(np.float64),
        peak.numpy().astype(np.float64),
        p99.numpy().astype(np.float64),
        leak.numpy().astype(np.float64),
        false_act.numpy().astype(np.float64),
        near,
        act_out.numpy(),
    )


def fp_geometry(W: torch.Tensor, b: torch.Tensor, p: float | np.ndarray, params: EvalParams) -> dict[str, np.ndarray]:
    """FP geometry predictors for one instance (``W [m, n]``)."""
    W3, b2 = W[None], b[None]
    return {
        "norm": feature_norms(W3)[0],
        "represented": feature_norms(W3)[0] > REPRESENTED_NORM,
        "D": dimensionality(W3)[0],
        "D_plus": d_plus(W3)[0],
        "max_overlap": max_overlap(W3)[0],
        "margin": margin(W3, b2)[0],
        "margin_tau": margin_tau(W3, b2, params.tau)[0],
        "superposition": superposition_score(W3)[0],
        "bias": b.numpy().astype(np.float64),
        "peak_ratio": peak_ratio(W3)[0],
        "leak_mean": leak_mean(W3, p)[0],
        "clean_resp": np.maximum(0.0, feature_norms(W3)[0] ** 2 + b.numpy().astype(np.float64)),
    }


def evaluate_instance(
    W: torch.Tensor,
    b: torch.Tensor,
    es: EvalSet,
    params: EvalParams,
    configs: list[QuantConfig],
    importance: np.ndarray,
    p: float | np.ndarray,
    instance: int = 0,
    train_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All sweep configurations for one instance -> (per-feature rows, per-model rows).

    ``instance`` salts the noise-control directions so they are independent across model instances.
    """
    W = W.detach().float()
    b = b.detach().float()
    n = W.shape[1]
    importance = np.asarray(importance, dtype=np.float64)
    pre_fp, out_fp = forward_single(W, W, b, es.X, True, params.forward_chunk)
    fp = stratum_stats(es, out_fp, pre_fp, params)
    del pre_fp, out_fp
    geom = fp_geometry(W, b, p, params)
    mse_fp = mse_from_strata(p, fp.cond_mse, fp.leak_energy)
    r2_fp = r2_from_strata(p, fp.cond_mse, fp.leak_energy)
    var = var_x(p)
    eligible = geom["represented"] & (np.nan_to_num(r2_fp, nan=-np.inf) >= params.eligible_r2)
    loss_fp = float((importance * mse_fp).sum())

    feat_frames, model_rows = [], []
    for cfg in configs:
        qr = apply_quant(cfg, W[None], b[None], instance=instance, train_seed=train_seed)
        _, out_q = forward_single(qr.W_enc[0], qr.W_dec[0], qr.b[0], es.X, False, params.forward_chunk)
        q = stratum_stats(es, out_q, None, params)
        del out_q
        corr = pearson_columns(fp.act_out, q.act_out)
        mse_q = mse_from_strata(p, q.cond_mse, q.leak_energy)
        r2_q = r2_from_strata(p, q.cond_mse, q.leak_energy)
        dr2 = r2_fp - r2_q
        dfalse = q.false_act - fp.false_act
        survived = corr >= params.survival_threshold
        # tolerance: a rate of exactly k/N can sit one ulp above the literal threshold
        composite = survived & (dr2 <= params.composite_dr2 + 1e-9) & (dfalse <= params.composite_dfalse + 1e-9)
        composite_f = np.where(np.isnan(dfalse), np.nan, composite.astype(float))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(eligible, dr2 / r2_fp, np.nan)
        cov = perturbation_covariates(W[None], qr.Wq, b[None], qr.b)
        rows = {
            "feature": np.arange(n),
            "importance": importance,
            "p": np.broadcast_to(np.asarray(p, dtype=np.float64), (n,)).copy(),
            "bits": cfg.bits,
            "granularity": cfg.granularity,
            "method": cfg.method,
            "noise_seed": cfg.noise_seed,
            "ablation": cfg.ablation,
            "config": cfg.label,
            **{k: v for k, v in geom.items()},
            "near_thr": fp.near_thr if fp.near_thr is not None else np.full(n, np.nan),
            "cond_mse_fp": fp.cond_mse,
            "false_act_fp": fp.false_act,
            "leak_energy_fp": fp.leak_energy,
            "peak_out_fp": fp.peak_out,
            "p99_out_fp": fp.p99_out,
            "mse_fp": mse_fp,
            "R2_fp": r2_fp,
            "var_x": np.broadcast_to(np.asarray(var, dtype=np.float64), (n,)).copy(),
            "eligible": eligible,
            "cond_mse_q": q.cond_mse,
            "false_act_q": q.false_act,
            "leak_energy_q": q.leak_energy,
            "peak_out_q": q.peak_out,
            "p99_out_q": q.p99_out,
            "mse_q": mse_q,
            "R2_q": r2_q,
            "dR2": dr2,
            "dcond_mse": q.cond_mse - fp.cond_mse,
            "dfalse_act": dfalse,
            "dleak_energy": q.leak_energy - fp.leak_energy,
            "dLoss": importance * (mse_q - mse_fp),
            "rel_damage": rel,
            "surv_corr": corr,
            "survived": survived,
            "composite_survived": composite_f,
            "norm_q": cov["norm_q"][0],
            "killed": cov["killed"][0],
            "d_norm2": cov["d_norm2"][0],
            "dW": cov["dW"][0],
            "d_overlap": cov["d_overlap"][0],
            "margin_q": cov["margin_q"][0],
            "margin_flip": cov["margin_flip"][0],
            "n_active_eval": es.n_active_eval,
            "n_inactive_eval": es.n_inactive_eval,
            "forced_active": (es.idx_act >= es.n_natural).all(0).numpy(),
            "forced_inactive": (
                (
                    (es.idx_inact >= es.n_natural).all(0).numpy()
                    & (es.has_inactive if es.has_inactive is not None else True)
                )
                if es.idx_inact is not None
                else np.zeros(n, dtype=bool)
            ),
        }
        feat_frames.append(pd.DataFrame(rows))
        loss_q = float((importance * mse_q).sum())
        n_el = int(eligible.sum())
        model_rows.append(
            {
                "bits": cfg.bits,
                "granularity": cfg.granularity,
                "method": cfg.method,
                "noise_seed": cfg.noise_seed,
                "ablation": cfg.ablation,
                "config": cfg.label,
                "loss_fp": loss_fp,
                "loss_q": loss_q,
                "loss_increase": loss_q - loss_fp,
                "loss_increase_rel": (loss_q - loss_fp) / loss_fp if loss_fp > 0 else np.nan,
                "n_eligible": n_el,
                "survived_frac": float(survived[eligible].mean()) if n_el else np.nan,
                "composite_frac": float(np.nanmean(composite_f[eligible]))
                if n_el and not np.all(np.isnan(composite_f[eligible]))
                else np.nan,
                "worst_dR2": float(np.nanmax(dr2[eligible])) if n_el else np.nan,
                "mean_dR2": float(np.nanmean(dr2[eligible])) if n_el else np.nan,
                "n_killed_represented": int(cov["killed"][0][geom["represented"]].sum()),
                "n_switched_on": int(((~eligible) & (np.nan_to_num(q.false_act, nan=0.0) > params.tau)).sum()),
                "frob_err": float(np.linalg.norm((qr.Wq[0] - W).numpy())),
            }
        )
    feats = pd.concat(feat_frames, ignore_index=True)
    for c in feats.columns:
        if feats[c].dtype == np.float64:
            feats[c] = feats[c].astype(np.float32)
    return feats, pd.DataFrame(model_rows)


def dead_feature_note() -> str:
    return f"D, D_plus, superposition, peak_ratio are NaN for columns with norm < {DIM_MIN_NORM}"

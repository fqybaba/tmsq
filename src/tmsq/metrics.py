"""Per-feature geometry metrics (CLAUDE.md, "Per-feature geometry metrics").

All functions accept ``W`` of shape ``[K, m, n]`` (``W_i`` = column ``i``) as a torch tensor or numpy array
and return numpy arrays. Per-feature outputs have shape ``[K, n]``.
"""

from __future__ import annotations

import numpy as np
import torch

REPRESENTED_NORM = 0.5
DIM_MIN_NORM = 0.1


def _np(W) -> np.ndarray:
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    return np.asarray(W, dtype=np.float64)


def gram(W) -> np.ndarray:
    """``G = W^T W``, shape ``[K, n, n]``; ``G[k, i, j] = W_i . W_j``."""
    W = _np(W)
    return np.einsum("kmi,kmj->kij", W, W)


def feature_norms(W) -> np.ndarray:
    """``norm_i = ||W_i||``, shape ``[K, n]``."""
    return np.linalg.norm(_np(W), axis=1)


def represented(W, threshold: float = REPRESENTED_NORM) -> np.ndarray:
    """Boolean ``[K, n]``: feature is represented iff ``norm_i > threshold``."""
    return feature_norms(W) > threshold


def dimensionality(W, min_norm: float = DIM_MIN_NORM) -> np.ndarray:
    """``D_i = ||W_i||^4 / sum_j (W_i . W_j)^2`` (sum includes ``j = i``); NaN if ``norm_i < min_norm``.

    Deliberate deviation from the paper, which assigns unlearned features ``D_i = 0``; use
    ``np.nan_to_num`` when summing.
    """
    G = gram(W)
    norms = np.sqrt(np.einsum("kii->ki", G))
    denom = (G**2).sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = norms**4 / denom
    D[norms < min_norm] = np.nan
    return D


def feats_per_dim(W) -> np.ndarray:
    """``||W||_F^2 / m``, shape ``[K]``."""
    W = _np(W)
    return (W**2).sum(axis=(1, 2)) / W.shape[1]


def dims_per_feat(W) -> np.ndarray:
    """The paper's ``D* = m / ||W||_F^2``, shape ``[K]``."""
    with np.errstate(divide="ignore"):
        return 1.0 / feats_per_dim(W)  # inf for an all-zero instance


def max_overlap(W) -> np.ndarray:
    """``max_overlap_i = max_{j != i} (W_i . W_j)``, shape ``[K, n]``."""
    G = gram(W).copy()
    n = G.shape[1]
    idx = np.arange(n)
    G[:, idx, idx] = -np.inf
    return G.max(axis=2)


def margin(W, b) -> np.ndarray:
    """``margin_i = -b_i - max(0, max_overlap_i)``: worst-case single-feature leak headroom (``x_j <= 1``).

    Positive margin means no single other active feature can switch feature ``i`` on. Trained models in
    superposition typically have NEGATIVE margins (CLAUDE.md running example: ``-0.059``).
    """
    b = _np(b)
    return -b - np.maximum(0.0, max_overlap(W))


def superposition_score(W) -> np.ndarray:
    """``sum_{j != i} (W_i_hat . W_j)^2``: the paper's per-feature "superposition" colouring, shape ``[K, n]``.

    0 for a feature orthogonal to all others; values near or above 1 mean other features can activate it as
    strongly as itself. NaN where ``norm_i < DIM_MIN_NORM``.
    """
    G = gram(W)
    norms = np.sqrt(np.einsum("kii->ki", G))
    with np.errstate(divide="ignore", invalid="ignore"):
        proj = G / norms[:, :, None]
    n = G.shape[1]
    idx = np.arange(n)
    proj[:, idx, idx] = 0.0
    score = (proj**2).sum(axis=2)
    score[norms < DIM_MIN_NORM] = np.nan
    return score


def angles_2d(W) -> np.ndarray:
    """Angle (radians, in ``(-pi, pi]``) of each column for ``m = 2``, shape ``[K, n]``."""
    W = _np(W)
    if W.shape[1] != 2:
        raise ValueError("angles_2d needs m = 2")
    return np.arctan2(W[:, 1, :], W[:, 0, :])


def is_pentagon(W, norm_threshold: float = REPRESENTED_NORM, d_tol: float = 0.05) -> np.ndarray:
    """Boolean ``[K]``: all five ``norm_i > norm_threshold`` and all five ``D_i`` within ``0.4 +- d_tol``.

    Operational definition from CLAUDE.md design note 9. It is a LOOSER proxy for "adjacent angles within
    72 deg +- 10" (every such configuration passes, but some with gaps off by up to ~30 deg pass too) and it is
    norm-sensitive; the experiment scripts record the sorted angular gaps alongside it.
    """
    W = _np(W)
    if W.shape[2] != 5:
        raise ValueError("is_pentagon expects n = 5")
    norms_ok = (feature_norms(W) > norm_threshold).all(axis=1)
    D = dimensionality(W)
    d_ok = np.all(np.abs(np.nan_to_num(D, nan=-1.0) - 0.4) <= d_tol, axis=1)
    return norms_ok & d_ok


def geometry_table(W, b) -> dict[str, np.ndarray]:
    """All per-feature geometry metrics as flat ``[K, n]`` arrays (for building tidy tables)."""
    return {
        "norm": feature_norms(W),
        "represented": represented(W),
        "D": dimensionality(W),
        "max_overlap": max_overlap(W),
        "margin": margin(W, b),
        "superposition": superposition_score(W),
        "bias": _np(b),
    }


# ----------------------------------------------------------------------------------------------------
# Phase 2: extra geometry predictors, per-feature damage metrics (CLAUDE.md "Per-feature damage
# metrics" and design notes 3-6) and realized-perturbation covariates (design note 10).
# ----------------------------------------------------------------------------------------------------

TAU_FALSE_ACT = 0.05


def d_plus(W, min_norm: float = DIM_MIN_NORM) -> np.ndarray:
    """Sign-aware dimensionality ``||W_i||^4 / (||W_i||^4 + sum_{j!=i} max(0, W_i . W_j)^2)`` (design note H2)."""
    G = gram(W)
    norms = np.sqrt(np.einsum("kii->ki", G))
    off = np.maximum(0.0, G).copy()
    idx = np.arange(G.shape[1])
    off[:, idx, idx] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        D = norms**4 / (norms**4 + (off**2).sum(axis=2))
    D[norms < min_norm] = np.nan
    return D


def peak_ratio(W, min_norm: float = DIM_MIN_NORM) -> np.ndarray:
    """Column peakiness ``max|W_i| / ||W_i||`` (in ``[1/sqrt(m), 1]``); NaN for dead columns."""
    W = _np(W)
    norms = np.linalg.norm(W, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(W).max(axis=1) / norms
    r[norms < min_norm] = np.nan
    return r


def margin_tau(W, b, tau: float = TAU_FALSE_ACT) -> np.ndarray:
    """Threshold-consistent margin ``tau - b_i - max(0, max_overlap_i)`` (design note 5)."""
    return tau + margin(W, b)


def leak_mean(W, p) -> np.ndarray:
    """Expected single-leaker drive ``sum_{j!=i} p_j max(0, W_i . W_j) / 2``; ``p`` scalar, ``[K, 1]`` or ``[K, n]``."""
    G = np.maximum(0.0, gram(W))
    idx = np.arange(G.shape[1])
    G[:, idx, idx] = 0.0
    p = np.broadcast_to(np.asarray(p, dtype=np.float64), (G.shape[0], G.shape[1]))
    return (G * p[:, None, :]).sum(axis=2) / 2.0


def var_x(p) -> np.ndarray:
    """``Var(x_i) = p/3 - p^2/4`` for ``x_i = 0`` w.p. ``1-p`` else Uniform[0,1)."""
    p = np.asarray(p, dtype=np.float64)
    return p / 3.0 - p**2 / 4.0


def mse_from_strata(p, cond_mse, leak_energy) -> np.ndarray:
    """``MSE_i = p cond_mse_i + (1-p) E[x_hat_i^2 | x_i = 0]``; the inactive term is 0 when ``p = 1``."""
    p = np.asarray(p, dtype=np.float64)
    leak = np.where(p >= 1.0, 0.0, np.asarray(leak_energy, dtype=np.float64))
    return p * np.asarray(cond_mse, dtype=np.float64) + (1.0 - p) * leak


def r2_from_strata(p, cond_mse, leak_energy) -> np.ndarray:
    """``R2_i = 1 - MSE_i / Var(x_i)``; NaN when ``Var = 0`` (``p = 0``)."""
    v = var_x(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - mse_from_strata(p, cond_mse, leak_energy) / v
    return np.where(v > 0, r2, np.nan)


def pearson_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation of two ``[N, n]`` arrays; NaN where either column is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    const = (np.ptp(a, axis=0) == 0) | (np.ptp(b, axis=0) == 0)  # dtype-independent constant guard
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denom = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a * b).sum(axis=0) / denom
    return np.where((denom > 0) & ~const, r, np.nan)


def perturbation_covariates(W, Wq, b, b_used) -> dict[str, np.ndarray]:
    """Design-note-10 covariates from the perturbed matrix ``Wq`` (and the bias actually used)."""
    W, Wq = _np(W), _np(Wq)
    G, Gq = gram(W), gram(Wq)
    idx = np.arange(G.shape[1])
    dG = np.abs(Gq - G)
    dG[:, idx, idx] = 0.0
    norm = np.linalg.norm(W, axis=1)
    norm_q = np.linalg.norm(Wq, axis=1)
    m_fp = margin(W, b)
    m_q = margin(Wq, b_used)
    return {
        "norm_q": norm_q,
        "killed": norm_q < REPRESENTED_NORM,
        "d_norm2": norm_q**2 - norm**2,
        "dW": np.linalg.norm(Wq - W, axis=1),
        "d_overlap": dG.max(axis=2),
        "margin_q": m_q,
        "margin_flip": np.sign(m_fp) != np.sign(m_q),
    }

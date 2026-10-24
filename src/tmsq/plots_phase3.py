"""Phase 3 figures (same palette and conventions as plots.py)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import CATEGORICAL, GRID, INK, INK2, MUTED, SEQ_CMAP, save, style

TEST_LABELS = {"H1": "H1  margin_tau vs dfalse_act", "H2": "H2  D vs dR2", "H3": "H3  norm vs dR2",
               "H2_Dplus": "H2 companion  D_plus vs dR2"}  # fmt: skip


def plot_forest(primary_by_set: dict[str, dict], stem: Path, title: str) -> None:
    """Per-instance rho (dots), mean and 95% t-interval (bar) for each primary test, one column per set.

    ``primary_by_set`` maps a set label to a primary.json-style dict (``results`` -> test -> rhos / summary / verdict).
    """
    style()
    names = ["H1", "H2", "H3", "H2_Dplus"]
    sets = list(primary_by_set)
    fig, axes = plt.subplots(1, len(sets), figsize=(5.4 * len(sets), 3.8), squeeze=False, sharey=True)
    fig.subplots_adjust(wspace=0.75)
    for ax, sname in zip(axes[0], sets, strict=True):
        res = primary_by_set[sname]["results"]
        for i, n in enumerate(names):
            r = res[n]
            y = len(names) - 1 - i
            rh = np.array([v for v in r["rhos"].values() if v is not None], dtype=float)
            jitter = np.linspace(-0.12, 0.12, len(rh)) if len(rh) else np.array([])
            ax.scatter(rh, y + jitter, s=14, color=CATEGORICAL[0], alpha=0.45, linewidths=0, zorder=2)
            s = r["summary"]
            if s.get("mean") is not None and np.isfinite(s["mean"]):
                if s.get("ci_lo") is not None and np.isfinite(s["ci_lo"]):
                    ax.plot([s["ci_lo"], s["ci_hi"]], [y, y], color=INK, lw=2, zorder=3)
                ax.scatter([s["mean"]], [y], s=36, color=INK, zorder=4)
            verdict = "\n".join(textwrap.wrap(r["verdict"], 30))
            ax.text(1.03, y, verdict, transform=ax.get_yaxis_transform(), fontsize=6.5, va="center", color=INK2)
        for v, ls in ((0, "-"), (-0.2, ":"), (0.2, ":")):
            ax.axvline(v, color=MUTED if v == 0 else GRID, lw=0.9 if v == 0 else 0.8, ls=ls, zorder=0)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([TEST_LABELS[n] for n in names[::-1]], fontsize=8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("per-instance Spearman rho (dots); mean, 95% t-interval")
        ax.set_title(sname, fontsize=9)
    fig.suptitle(title, fontsize=10, color=INK)
    save(fig, stem)


def plot_damage_vs_margin(c: pd.DataFrame, stem: Path, title: str) -> None:
    """Primary cell: dfalse_act and dR2 against margin_tau, colored by D (darker = larger D)."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), squeeze=False)
    vmin, vmax = np.nanpercentile(c.D, 2), np.nanpercentile(c.D, 98)
    for ax, (y, ylab) in zip(axes[0], (("dfalse_act", "change in false-activation rate"), ("dR2", "dR2 (R2 lost)")),
                             strict=True):  # fmt: skip
        sc = ax.scatter(c.margin_tau, c[y], c=c.D, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, s=10, linewidths=0, alpha=0.8)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_xlabel("margin_tau = tau - b_i - max(0, max overlap)")
        ax.set_ylabel(ylab)
    cbar = fig.colorbar(sc, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("feature dimensionality D_i", color=INK2)
    fig.suptitle(title, fontsize=10, color=INK)
    save(fig, stem)


def plot_auc(auc: pd.DataFrame, logit: pd.DataFrame, cell: tuple, stem: Path, title: str) -> None:
    """Primary cell: per-instance mean AUC (t-interval) per predictor and CV AUC (cluster bootstrap) per set."""
    style()
    S, g, b = cell
    sel = lambda t: t[(t.S == S) & (t.granularity == g) & (t.bits == b)] if len(t) and "S" in t else t.iloc[0:0]  # noqa: E731
    a, lg = sel(auc), sel(logit)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), squeeze=False, sharex=True)
    fig.subplots_adjust(wspace=0.5)
    ax = axes[0, 0]
    if len(a):
        yy = np.arange(len(a))[::-1]
        lo = (a.auc_mean - a.auc_ci_lo).fillna(0)
        hi = (a.auc_ci_hi - a.auc_mean).fillna(0)
        ax.errorbar(a.auc_mean, yy, xerr=[lo, hi], fmt="o", ms=5, color=CATEGORICAL[0], ecolor=CATEGORICAL[0],
                    elinewidth=1.6, capsize=0)  # fmt: skip
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{p} ({'+' if o > 0 else '-'})" for p, o in zip(a.predictor, a.orientation, strict=True)],
                           fontsize=8)  # fmt: skip
    else:
        ax.text(0.5, 0.5, "primary cell fails the cell rule:\nno label-based analysis", ha="center", va="center",
                transform=ax.transAxes, color=INK2)  # fmt: skip
    ax.axvline(0.5, color=MUTED, lw=0.9)
    ax.set_xlabel("per-instance AUC for 'damaged' (mean, 95% t-interval)")
    ax.set_title("univariate, oriented by the registered sign", fontsize=9)
    ax = axes[0, 1]
    if len(lg):
        yy = np.arange(len(lg))[::-1]
        ax.errorbar(lg.cv_auc_pooled_oof, yy, xerr=[lg.cv_auc_pooled_oof - lg.cv_auc_boot_lo,
                    lg.cv_auc_boot_hi - lg.cv_auc_pooled_oof], fmt="o", ms=5, color=CATEGORICAL[1],
                    ecolor=CATEGORICAL[1], elinewidth=1.6, capsize=0)  # fmt: skip
        ax.set_yticks(yy)
        ax.set_yticklabels(lg["set"], fontsize=8)
    ax.axvline(0.5, color=MUTED, lw=0.9)
    ax.set_xlabel("cross-validated AUC (pooled out-of-fold)")
    ax.set_title("logistic regression, GroupKFold by instance", fontsize=9)
    ax.set_xlim(0.2, 1.0)
    fig.suptitle(f"{title}\nright: cluster-bootstrap interval over instances (undercovers, ~90% for nominal 95%)",
                 fontsize=9, color=INK)  # fmt: skip
    save(fig, stem)

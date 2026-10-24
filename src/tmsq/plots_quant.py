"""Phase 2 figures (same palette and conventions as plots.py)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .plots import CATEGORICAL, GRID, INK, INK2, MUTED, SEQ_CMAP, save, style
from .quantize import QuantConfig, apply_quant


def _s_colors(S_values: np.ndarray) -> dict:
    order = np.argsort(S_values)
    cols = [SEQ_CMAP(v) for v in np.linspace(0.3, 1.0, len(S_values))]
    return {float(S_values[i]): cols[r] for r, i in enumerate(order)}


def plot_metric_vs_bits(
    models: pd.DataFrame, value: str, ylabel: str, stem: Path, title: str, logy: bool, method: str = "rtn"
) -> None:
    """Panels = granularities; one line per S (mean over seeds, band = min..max); x = bits, 8 -> 2."""
    style()
    df = models[(models.method == method) & (models.ablation == "none")]
    grans = [g for g in ("tensor", "column", "row") if g in set(df.granularity)]
    fig, axes = plt.subplots(1, len(grans), figsize=(3.6 * len(grans), 3.4), squeeze=False, sharey=True)
    S_vals = np.sort(df.S.unique())
    colors = _s_colors(S_vals)
    for ax, g in zip(axes[0], grans, strict=True):
        d = df[df.granularity == g]
        for S in S_vals:
            sub = d[d.S == S].groupby("bits")[value].agg(["mean", "min", "max"]).sort_index(ascending=False)
            x = sub.index.to_numpy()
            ax.plot(x, sub["mean"], color=colors[float(S)], lw=1.4, marker="o", ms=3, label=f"S={S:.3g}")
            ax.fill_between(x, sub["min"], sub["max"], color=colors[float(S)], alpha=0.12, lw=0)
        ax.set_title(f"per-{g}", fontsize=9)
        ax.set_xlabel("bits")
        ax.invert_xaxis()
        if logy:  # symlog: a quantized model can be slightly BETTER than FP (negative increase)
            ax.set_yscale("symlog", linthresh=1e-5)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, -1].legend(fontsize=7, loc="best", ncol=1 if len(S_vals) <= 8 else 2)
    fig.suptitle(title, fontsize=10, color=INK)
    save(fig, stem)


def plot_rtn_vs_noise(models: pd.DataFrame, stem: Path, title: str, value: str = "loss_increase") -> None:
    """RTN (solid) vs the matched-noise control (dashed, mean over noise seeds) for up to 4 S values."""
    style()
    df = models[models.ablation == "none"]
    grans = [g for g in ("tensor", "column", "row") if g in set(df.granularity)]
    S_all = np.sort(df.S.unique())
    pick = S_all if len(S_all) <= 4 else S_all[np.linspace(0, len(S_all) - 1, 4).round().astype(int)]
    colors = _s_colors(pick)
    fig, axes = plt.subplots(1, len(grans), figsize=(3.6 * len(grans), 3.4), squeeze=False, sharey=True)
    for ax, g in zip(axes[0], grans, strict=True):
        d = df[df.granularity == g]
        for S in pick:
            for method, ls in (("rtn", "-"), ("noise", "--")):
                sub = d[(d.S == S) & (d.method == method)].groupby("bits")[value].mean().sort_index(ascending=False)
                if len(sub):
                    ax.plot(
                        sub.index,
                        sub.to_numpy(),
                        color=colors[float(S)],
                        ls=ls,
                        lw=1.4,
                        marker="o",
                        ms=3,
                        label=f"S={S:.3g} {method}",
                    )
        ax.set_title(f"per-{g}", fontsize=9)
        ax.set_xlabel("bits")
        ax.invert_xaxis()
        ax.set_yscale("symlog", linthresh=1e-5)
    axes[0, 0].set_ylabel("total weighted loss increase (symlog)")
    axes[0, -1].legend(fontsize=6.5, loc="best", ncol=2)
    fig.suptitle(f"{title}; solid = RTN, dashed = Gaussian noise with the same per-column error", fontsize=9, color=INK)
    save(fig, stem)


def plot_arrows_fp_vs_q(
    W: torch.Tensor,
    b: torch.Tensor,
    density: np.ndarray,
    seed_idx: np.ndarray,
    bits_list: list[int],
    stem: Path,
    granularity: str = "column",
) -> None:
    """Setting A: FP columns (gray) overlaid with per-column RTN columns (colored); rows = bits, cols = S; seed 0."""
    style()
    ks = np.flatnonzero(seed_idx == 0)
    n = W.shape[2]
    cols = [SEQ_CMAP(v) for v in np.linspace(0.3, 1.0, n)][::-1]
    fig, axes = plt.subplots(len(bits_list), len(ks), figsize=(1.9 * len(ks), 1.9 * len(bits_list)), squeeze=False)
    lim = max(1.3, float(W[ks].abs().max()) * 1.05)
    for r, bw in enumerate(bits_list):
        Wq = apply_quant(QuantConfig(bw, granularity), W, b).Wq
        for c, k in enumerate(ks):
            ax = axes[r, c]
            for i in range(n):
                ax.annotate(
                    "",
                    xy=(W[k, 0, i], W[k, 1, i]),
                    xytext=(0, 0),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": MUTED,
                        "lw": 1.2,
                        "alpha": 0.6,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )
                ax.annotate(
                    "",
                    xy=(Wq[k, 0, i], Wq[k, 1, i]),
                    xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": cols[i], "lw": 1.4, "shrinkA": 0, "shrinkB": 0},
                )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.axhline(0, color=GRID, lw=0.6, zorder=0)
            ax.axvline(0, color=GRID, lw=0.6, zorder=0)
            if r == 0:
                ax.set_title(f"S = {1 - density[k]:.3g}", fontsize=9)
            if r == len(bits_list) - 1:
                ax.set_xlabel("W_i[0]", color=INK2)
            if c == 0:
                ax.set_ylabel(f"{bw}-bit\nW_i[1]", color=INK2)
    fig.suptitle(
        f"Setting A, seed 0: FP columns (gray) vs per-{granularity} RTN (colored, darker = more important)",
        fontsize=9,
        color=INK,
    )
    save(fig, stem)


__all__ = ["plot_metric_vs_bits", "plot_rtn_vs_noise", "plot_arrows_fp_vs_q", "CATEGORICAL"]

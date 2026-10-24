"""Figures: matplotlib, colorblind-safe palette, labeled axes, PNG + PDF (CLAUDE.md "Engineering rules").

Palette: the validated reference palette of the dataviz skill. Ordered quantities (importance rank, sparsity
sweep) use one blue hue light->dark; identities use the fixed categorical order; signed overlaps use a
blue <-> red diverging map with a neutral gray midpoint.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#6b1313"])


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": SURFACE,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: Path) -> None:
    """Save ``<stem>.png`` and ``<stem>.pdf``."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def ordinal_colors(n: int) -> list:
    """``n`` ordered colors from the blue ramp, darkest = last."""
    return [SEQ_CMAP(v) for v in np.linspace(0.3, 1.0, n)]


def _s_label(density: float) -> str:
    return f"S = {1 - density:.3g}"


def plot_arrows_grid(
    W: np.ndarray,
    density: np.ndarray,
    sweep_idx: np.ndarray,
    seed_idx: np.ndarray,
    stem: Path,
    title: str = "Setting A: columns of W (n=5, m=2)",
) -> None:
    """Rows = seeds, columns = sparsity values; each panel draws the n feature vectors from the origin."""
    style()
    sweeps = np.unique(sweep_idx)
    seeds = np.unique(seed_idx)
    n = W.shape[2]
    colors = ordinal_colors(n)  # by importance rank (feature 0 = most important = darkest)
    colors = colors[::-1]
    fig, axes = plt.subplots(len(seeds), len(sweeps), figsize=(1.9 * len(sweeps), 1.9 * len(seeds)), squeeze=False)
    lim = max(1.3, float(np.abs(W).max()) * 1.05)
    for r, sd in enumerate(seeds):
        for c, sw in enumerate(sweeps):
            ax = axes[r, c]
            k = int(np.flatnonzero((sweep_idx == sw) & (seed_idx == sd))[0])
            for i in range(n):
                ax.annotate(
                    "",
                    xy=(W[k, 0, i], W[k, 1, i]),
                    xytext=(0, 0),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": colors[i],
                        "lw": 1.6,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
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
                ax.set_title(_s_label(float(density[k])), fontsize=9)
            if r == len(seeds) - 1:
                ax.set_xlabel("W_i[0]", color=INK2)
            if c == 0:
                ax.set_ylabel(f"seed {sd}\nW_i[1]", color=INK2)
    fig.suptitle(f"{title}; darker = more important feature", fontsize=10, color=INK)
    save(fig, stem)


def plot_gram_and_bias(
    W: np.ndarray,
    b: np.ndarray,
    density: np.ndarray,
    instances: np.ndarray,
    stem: Path,
    title: str = "Setting B: W^T W and bias (seed 0)",
) -> None:
    """One column per selected instance: ``W^T W`` heatmap (top) and bias bar chart (bottom)."""
    style()
    cols = len(instances)
    n = W.shape[2]
    fig, axes = plt.subplots(2, cols, figsize=(1.75 * cols, 4.2), squeeze=False, height_ratios=[3, 1.2])
    vmax = max(1.0, float(max(np.abs(W[int(k)].T @ W[int(k)]).max() for k in instances)))  # no clipping
    for c, k in enumerate(instances):
        k = int(k)
        G = W[k].T @ W[k]
        ax = axes[0, c]
        im = ax.imshow(G, cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(_s_label(float(density[k])), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel("feature j", color=INK2, fontsize=7)
        ax2 = axes[1, c]
        ax2.bar(np.arange(n), b[k], color=CATEGORICAL[0], width=0.8)
        ax2.axhline(0, color=AXIS, lw=0.6)
        ax2.set_xlim(-0.5, n - 0.5)
        ax2.set_xticks([])
        ax2.set_xlabel("feature i", color=INK2, fontsize=7)
        if c == 0:
            ax2.set_ylabel("bias b_i", color=INK2)
            axes[0, c].set_ylabel("feature i", color=INK2)
        else:
            ax2.tick_params(labelleft=False)
    ylims = [ax.get_ylim() for ax in axes[1]]
    lo, hi = min(y[0] for y in ylims), max(y[1] for y in ylims)
    for ax in axes[1]:
        ax.set_ylim(lo, hi)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("W_i · W_j", color=INK2)
    fig.suptitle(title, fontsize=10, color=INK)
    save(fig, stem)


def plot_norms_by_superposition(
    W: np.ndarray,
    density: np.ndarray,
    instances: np.ndarray,
    superposition: np.ndarray,
    stem: Path,
    title: str = "Setting B: feature norms, colored by superposition (seed 0)",
) -> None:
    """One column per instance: horizontal bars ``||W_i||`` colored by ``sum_{j!=i} (W_i_hat . W_j)^2``."""
    style()
    cols = len(instances)
    n = W.shape[2]
    norms = np.linalg.norm(W, axis=1)
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    fig, axes = plt.subplots(1, cols, figsize=(1.5 * cols, 3.4), squeeze=False, sharey=True)
    for c, k in enumerate(instances):
        k = int(k)
        ax = axes[0, c]
        s = np.nan_to_num(superposition[k], nan=0.0)
        ax.barh(np.arange(n), norms[k], color=SEQ_CMAP(norm(s)), height=0.8)
        ax.set_title(_s_label(float(density[k])), fontsize=8)
        ax.set_xlim(0, max(1.1, float(norms.max()) * 1.05))
        ax.invert_yaxis() if c == 0 else None
        ax.set_xlabel("||W_i||", color=INK2)
        if c == 0:
            ax.set_ylabel("feature i (0 = most important)", color=INK2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQ_CMAP)
    cbar = fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("superposition: sum_j (W_i_hat · W_j)^2 (clipped at 1)", color=INK2)
    fig.suptitle(title, fontsize=10, color=INK)
    save(fig, stem)


def plot_dims_per_feature(
    density: np.ndarray,
    dstar: np.ndarray,
    stem: Path,
    dstar_snapshot: np.ndarray | None = None,
    snapshot_label: str = "half-way snapshot",
    title: str = "Setting C: dimensions per feature D* = m / ||W||_F^2",
) -> None:
    """The paper's "sticky" D* curve: x = 1/(1-S) (log), y = D*, one point per instance."""
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = 1.0 / density
    for level, label, ls in [
        (1.0, "1 (orthogonal)", "-"),
        (0.5, "1/2 (antipodal pairs)", "--"),
        (0.4, "2/5 (pentagons)", ":"),
    ]:
        ax.axhline(level, color=MUTED, lw=0.8, ls=ls, zorder=0)
        ax.text(x.max() * 1.15, level, label, va="center", ha="left", fontsize=8, color=INK2)
    if dstar_snapshot is not None:
        ax.scatter(x, dstar_snapshot, s=18, color=CATEGORICAL[1], alpha=0.7, label=snapshot_label, zorder=2)
    ax.scatter(x, dstar, s=22, color=CATEGORICAL[0], alpha=0.85, label="final weights", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("1 / (1 - S)   (feature sparsity, log scale)")
    ax.set_ylabel("D* = m / ||W||_F^2")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(x.min() * 0.8, x.max() * 1.1)
    if dstar_snapshot is not None:
        ax.legend(loc="upper right")
    ax.set_title(title, fontsize=10)
    save(fig, stem)


def plot_feature_dims_scatter(
    density_per_feature: np.ndarray,
    D: np.ndarray,
    stem: Path,
    title: str = "Setting C: per-feature dimensionality D_i",
) -> None:
    """Scatter of ``D_i`` for every (instance, feature) against ``1/(1-S)``; NaN (dead) features drawn at 0."""
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = 1.0 / density_per_feature
    y = np.nan_to_num(D, nan=0.0)
    jitter = np.exp(np.random.default_rng(0).normal(0, 0.03, size=x.shape))
    ax.scatter(x * jitter, y, s=5, color=CATEGORICAL[0], alpha=0.25, linewidths=0, zorder=2)
    for level, ls in [(1.0, "-"), (0.5, "--"), (0.4, ":"), (1 / 3, ":")]:
        ax.axhline(level, color=MUTED, lw=0.8, ls=ls, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("1 / (1 - S)   (log scale; small horizontal jitter)")
    ax.set_ylabel("D_i  (dead features at 0)")
    ax.set_ylim(-0.02, 1.1)
    ax.set_title(title, fontsize=10)
    save(fig, stem)


def plot_loss_curves(
    loss_history: np.ndarray,
    density: np.ndarray,
    sweep_idx: np.ndarray,
    stem: Path,
    title: str = "Training loss (mean over seeds per S)",
    every: int = 20,
) -> None:
    """Per-S mean loss over seeds vs step, log y, colored by sparsity with a colorbar."""
    style()
    sweeps = np.unique(sweep_idx)
    dens = np.array([density[sweep_idx == s][0] for s in sweeps])
    norm = LogNorm(vmin=max(dens.min(), 1e-4), vmax=1.0)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    steps = np.arange(loss_history.shape[0])
    for s, d in zip(sweeps, dens, strict=True):
        cols = np.flatnonzero(sweep_idx == s)
        y = loss_history[:, cols].mean(axis=1)
        ax.plot(steps[::every], y[::every], color=SEQ_CMAP(1.0 - norm(d)), lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (mean over batch of sum_i I_i (x_i - x_hat_i)^2)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQ_CMAP.reversed())
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("feature density 1 - S (darker = sparser)", color=INK2)
    ax.set_title(title, fontsize=10)
    save(fig, stem)


def plot_count_vs_sparsity(
    density: np.ndarray,
    counts: np.ndarray,
    stem: Path,
    ylabel: str = "represented features (norm > 0.5)",
    title: str = "Setting B: represented features vs sparsity",
) -> None:
    style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = 1.0 / density
    jitter = np.exp(np.random.default_rng(1).normal(0, 0.03, size=x.shape))
    ax.scatter(x * jitter, counts, s=20, color=CATEGORICAL[0], alpha=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("1 / (1 - S)   (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    save(fig, stem)

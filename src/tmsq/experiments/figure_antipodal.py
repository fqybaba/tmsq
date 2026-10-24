"""Write-up figure for the Phase 3b result (descriptive; reads the registered K2 sweep, changes no number).

``python -m tmsq.experiments.figure_antipodal --sweep results/quant_sweep/20261024-142123-C_confirm2``

Left: FP bias vs D for eligible features (S = 0.7488), coloured by the registered damaged label at per-row 3-bit
RTN. Right: per-instance damaged fraction, antipodal (D >= 0.48) vs other, at per-row 3 and 4 bits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..plots import CATEGORICAL, GRID, INK, INK2, MUTED, save, style
from .common import FIGURES_ROOT
from .phase3b import D_ANTIPODAL, cell, load


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3b write-up figure")
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=FIGURES_ROOT)
    args = ap.parse_args()
    df = load(args.sweep)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={"width_ratios": [1.3, 1.0]})
    c3 = cell(df, "row", 3)
    ax = axes[0]
    for lab, col, name in ((0.0, CATEGORICAL[0], "survived"), (1.0, CATEGORICAL[1], "damaged")):
        d = c3[c3.damaged == lab]
        ax.scatter(d.D, d.bias, s=9, color=col, alpha=0.55, linewidths=0, label=f"{name} at 3 bits (n={len(d)})")
    ax.axvline(D_ANTIPODAL, color=MUTED, lw=0.8, ls="--")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("feature dimensionality D_i (FP)")
    ax.set_ylabel("bias b_i (FP)")
    ax.set_title("eligible features, 20 instances, S = 0.7488", fontsize=9, color=INK2)
    ax.text(D_ANTIPODAL + 0.003, ax.get_ylim()[0] * 0.9, "antipodal\n(D ≥ 0.48)", fontsize=7, color=INK2)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, color=GRID)
    ax = axes[1]
    for j, bits in enumerate((3, 4)):
        c = cell(df, "row", bits)
        rows = []
        for _, d in c.groupby("instance"):
            a, o = d[d.D >= D_ANTIPODAL].damaged.dropna(), d[d.D < D_ANTIPODAL].damaged.dropna()
            if len(a) >= 2 and len(o) >= 10:
                rows.append((o.mean(), a.mean()))
        rows = np.array(rows)
        x0 = 3 * j
        for o, a in rows:
            ax.plot([x0, x0 + 1], [o, a], color=MUTED, lw=0.6, alpha=0.6)
        ax.scatter(np.full(len(rows), x0), rows[:, 0], s=14, color=CATEGORICAL[0], zorder=3)
        ax.scatter(np.full(len(rows), x0 + 1), rows[:, 1], s=14, color=CATEGORICAL[1], zorder=3)
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["other\n3 bits", "antipodal\n3 bits", "other\n4 bits", "antipodal\n4 bits"], fontsize=8)
    ax.set_ylabel("damaged fraction (per instance)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("per-row RTN; lines join the same model", fontsize=9, color=INK2)
    ax.grid(True, color=GRID)
    fig.suptitle("Setting C (n=100, m=20): zero-bias antipodal-pair features break first (Phase 3b, fresh models)",
                 fontsize=10, color=INK)  # fmt: skip
    save(fig, args.out / "phase3b_antipodal")
    print(f"written {args.out / 'phase3b_antipodal'}.png/.pdf")


if __name__ == "__main__":
    main()

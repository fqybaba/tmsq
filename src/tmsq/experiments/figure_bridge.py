"""Write-up figures for the real-model bridge (Sections 8): replication vs Duan's Table 1, and the INT6 ablation.

``python -m tmsq.experiments.figure_bridge`` (main venv; reads finished runs, changes no number)

- ``figures/bridge_replication``: survival and perplexity change by bit width for Duan's released code (ours),
  the paper-described method (ours) and Duan's reported Table 1 values.
- ``figures/bridge_ablation``: INT6 survival when one factor at a time is changed, from either end
  (exploratory ablation, ``tmsq.experiments.bridge_ablation``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..plots import CATEGORICAL, GRID, INK, INK2, MUTED, SURFACE, save, style
from .common import FIGURES_ROOT, RESULTS_ROOT

BITS = [8, 7, 6, 5, 4]
# Duan (2026), arXiv 2606.03002 v2, Table 1 (Pythia-70M): survived % and perplexity change %, as recorded in
# notes/LOG.md (Phase 6b plan, verified facts)
DUAN_SURVIVED = {8: 98.312, 7: 86.322, 6: 62.395, 5: 37.957, 4: 14.504}
DUAN_DPPL = {8: 1.020, 7: 2.258, 6: 14.722, 5: 71.208, 4: 371.112}
CODE_RUNS = ["bridge/20261024-230918-code-b8-7-5-4", "bridge/20261024-230636-code-b6"]
STATED_RUN = "bridge/20261024-231452-stated-b8-7-6-5-4"
ABLATION_RUNS = ["bridge_ablation/20261025-024313-r1-int6", "bridge_ablation/20261025-024807-c1-int6",
                 "bridge_ablation/20261025-025332-c2-int6", "bridge_ablation/20261025-025755-r2-int6"]  # fmt: skip
BLUE, ORANGE = CATEGORICAL[0], CATEGORICAL[1]
FROM_CODE = [("A0", "Duan's code as released"), ("A0-Q", "per-channel Q/K/V scales"),
             ("A0-F", "no LayerNorm folding"), ("A0-B", "no value-bias folding"),
             ("A0-C", "no weight centering"), ("A0-CI", "no weight centering, raw SAE input")]  # fmt: skip
FROM_STATED = [("S", "described method"), ("S-Q", "shared Q/K/V scales"), ("S-F", "LayerNorm folded"),
               ("S-B", "value biases folded"), ("S-I", "centered SAE input"),
               ("S-CI", "centered weights and SAE input")]  # fmt: skip


def variants(runs: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in runs:
        out.update(json.loads((RESULTS_ROOT / r / "summary.json").read_text(encoding="utf-8"))["variants"])
    return out


def fig_replication(out: Path) -> None:
    code, stated = variants(CODE_RUNS), variants([STATED_RUN])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    series = [
        ("Duan's code, this replication", BLUE, [code[str(b)] for b in BITS]),
        ("paper-described method", ORANGE, [stated[str(b)] for b in BITS]),
    ]
    panels = [("survived_pct", DUAN_SURVIVED, "features surviving (c_j > 0.9), %"),
              ("dppl_pct", DUAN_DPPL, "perplexity increase, %")]  # fmt: skip
    for ax, (key, duan, ylabel) in zip(axes, panels, strict=True):
        x = np.arange(len(BITS))
        for name, col, vals in series:
            ax.plot(x, [v[key] for v in vals], color=col, lw=2, marker="o", ms=8, label=name, zorder=3)
        ax.scatter(x, [duan[b] for b in BITS], s=150, facecolors="none", edgecolors=INK, lw=1.2, zorder=4,
                   label="Duan, Table 1 (reported)")  # fmt: skip
        ax.set_xticks(x, [f"INT{b}" for b in BITS])
        ax.set_ylabel(ylabel)
        ax.grid(True, color=GRID)
        if key == "dppl_pct":
            ax.set_yscale("log")
            ax.set_title("log scale", fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Pythia-70M + layer-4 SAE: the released code reproduces Duan's Table 1; the described method "
                 "damages less", fontsize=10, color=INK)  # fmt: skip
    fig.tight_layout()
    save(fig, out / "bridge_replication")


def fig_ablation(out: Path) -> None:
    v = variants(ABLATION_RUNS)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    ends = [(FROM_CODE, BLUE, "starting from Duan's code"),
            (FROM_STATED, ORANGE, "starting from the described method")]  # fmt: skip
    for ax, (rows, col, title) in zip(axes, ends, strict=True):
        vals = [v[k]["survived_pct"] for k, _ in rows]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, height=0.62, color=col, edgecolor="white", linewidth=2)
        ax.axvline(vals[0], color=MUTED, lw=1, ls="--", zorder=0)
        for yi, val in zip(y, vals, strict=True):
            ax.text(val + 1, yi, f"{val:.1f}", va="center", fontsize=8, color=INK2, zorder=5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0})  # fmt: skip
        ax.set_yticks(y, [lab for _, lab in rows], fontsize=8)
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.set_title(title, fontsize=9, color=INK2)
        ax.set_xlim(0, 105)
        ax.set_xlabel("features surviving at INT6, %")
        ax.grid(True, axis="x", color=GRID)
        ax.set_axisbelow(True)
    fig.suptitle("One change at a time (exploratory): the shared Q/K/V scales drive the gap; LayerNorm folding "
                 "adds ~9 points with per-channel scales", fontsize=10, color=INK)  # fmt: skip
    fig.tight_layout()
    save(fig, out / "bridge_ablation")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bridge write-up figures")
    ap.add_argument("--out", type=Path, default=FIGURES_ROOT)
    args = ap.parse_args()
    style()
    fig_replication(args.out)
    fig_ablation(args.out)
    print(f"written {args.out / 'bridge_replication'} and {args.out / 'bridge_ablation'} (.png/.pdf)")


if __name__ == "__main__":
    main()

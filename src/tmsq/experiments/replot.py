"""Regenerate the figures of a finished run from its saved results, without retraining.

Usage: ``python -m tmsq.experiments.replot --run results/<exp>/<timestamp>``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..metrics import dims_per_feat, superposition_score
from ..plots import (
    plot_arrows_grid,
    plot_count_vs_sparsity,
    plot_dims_per_feature,
    plot_feature_dims_scatter,
    plot_gram_and_bias,
    plot_loss_curves,
    plot_norms_by_superposition,
)
from .common import FIGURES_ROOT, load_weights


def replot_quant(run: Path, cfg: dict, out: Path) -> None:
    """Phase 2 figures from a quant_sweep results directory."""
    import json

    from ..plots_quant import plot_arrows_fp_vs_q, plot_metric_vs_bits, plot_rtn_vs_noise

    meta = json.load(open(run / "meta.json", encoding="utf-8"))
    setting = meta["setting"]
    models = pd.read_parquet(run / "models.parquet")
    out.mkdir(parents=True, exist_ok=True)
    plot_metric_vs_bits(
        models,
        "loss_increase",
        "total weighted loss increase (symlog)",
        out / f"quant_{setting}_loss_vs_bits",
        title=f"Setting {setting}: loss increase vs bits (RTN)",
        logy=True,
    )
    plot_metric_vs_bits(
        models,
        "survived_frac",
        "survived fraction (eligible features)",
        out / f"quant_{setting}_survival_vs_bits",
        title=f"Setting {setting}: survival vs bits (RTN)",
        logy=False,
    )
    plot_rtn_vs_noise(models, out / f"quant_{setting}_rtn_vs_noise", title=f"Setting {setting}: RTN vs matched noise")
    if setting == "A":
        final = load_weights(Path(meta["source_run"]) / "final.pt")
        bits = [bw for bw in (4, 3, 2) if bw in set(models.bits)]
        plot_arrows_fp_vs_q(
            final["W"],
            final["b"],
            np.asarray(final["density"]),
            np.asarray(final["seed_idx"]),
            bits,
            out / "quant_A_arrows",
        )
    print(f"figures for quant_sweep {setting} written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate figures from a results directory")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=FIGURES_ROOT, help="figures directory (default: figures/)")
    args = ap.parse_args()
    run: Path = args.run
    with open(run / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    exp = cfg["experiment"]
    prefix = cfg.get("figure_prefix", exp)
    if exp == "quant_sweep":
        replot_quant(run, cfg, args.out)
        return
    final = load_weights(run / "final.pt")
    W, b = final["W"].numpy(), final["b"].numpy()
    density, sweep_idx, seed_idx = (np.asarray(final[k]) for k in ("density", "sweep_idx", "seed_idx"))
    loss = np.load(run / "loss_history.npy")
    out = args.out

    if exp == "rep_5in2":
        plot_arrows_grid(W, density, sweep_idx, seed_idx, out / "rep_5in2_arrows")
        plot_loss_curves(loss, density, sweep_idx, out / "rep_5in2_loss", title="Setting A: training loss")
    elif exp == "rep_20in5":
        seed0 = np.flatnonzero(seed_idx == 0)
        inst = pd.read_parquet(run / "instances.parquet")
        plot_gram_and_bias(W, b, density, seed0, out / "rep_20in5_gram_bias")
        plot_norms_by_superposition(W, density, seed0, superposition_score(W), out / "rep_20in5_norms")
        plot_count_vs_sparsity(density, inst["n_represented"].to_numpy(), out / "rep_20in5_count")
        plot_loss_curves(loss, density, sweep_idx, out / "rep_20in5_loss", title="Setting B: training loss")
    elif exp.startswith("rep_dims"):
        df = pd.read_parquet(run / "features.parquet")
        snaps = sorted(run.glob("snapshot_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        dstar_snap, label = None, ""
        if snaps:
            snap = load_weights(snaps[-1])
            dstar_snap = dims_per_feat(snap["W"].numpy())
            label = f"step {snaps[-1].stem.split('_')[1]} snapshot"
        plot_dims_per_feature(
            density, dims_per_feat(W), out / f"{prefix}_dstar", dstar_snapshot=dstar_snap, snapshot_label=label
        )
        plot_feature_dims_scatter(df["density"].to_numpy(), df["D"].to_numpy(), out / f"{prefix}_feature_D")
        plot_loss_curves(loss, density, sweep_idx, out / f"{prefix}_loss", title="Setting C: training loss")
    else:
        raise SystemExit(f"unknown experiment {exp!r}")
    print(f"figures for {exp} written to {out}")


if __name__ == "__main__":
    main()

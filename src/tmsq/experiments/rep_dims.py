"""Setting C (n=100, m=20, uniform importance): the "dimensions per feature" curve D* = m / ||W||_F^2.

Scaled-down version of the paper's figure (n=400, m=30) and of the notebook's (n=200, m=20).
Acceptance (fixed in notes/LOG.md before running): D* sticky near 1 at the dense end and near 1/2 for at
least two adjacent S values, each with >= 3 of 5 seeds within +- 0.05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..metrics import dimensionality, dims_per_feat, feature_norms
from ..plots import plot_dims_per_feature, plot_feature_dims_scatter, plot_loss_curves
from ..train import plateau_ratio, train
from .common import (
    apply_runtime,
    build_model_and_sampler,
    expand_sweep,
    feature_frame,
    importance_curve,
    load_config,
    log_density_grid,
    make_run_dir,
    parse_args,
    publish_figures,
    run_figure_dir,
    save_provenance,
    save_weights,
    set_seed,
    train_config_from,
    write_json,
)


def sticky_runs(dstar: np.ndarray, sweep_idx: np.ndarray, level: float, tol: float, min_seeds: int) -> list[int]:
    """Sweep indices at which >= ``min_seeds`` instances have ``|D* - level| < tol``."""
    hits = []
    for sw in np.unique(sweep_idx):
        ks = np.flatnonzero(sweep_idx == sw)
        if int((np.abs(dstar[ks] - level) < tol).sum()) >= min_seeds:
            hits.append(int(sw))
    return hits


def has_adjacent_pair(idx: list[int]) -> bool:
    return any(b - a == 1 for a, b in zip(idx, idx[1:], strict=False))


def main() -> None:
    args = parse_args("Setting C replication: dimensions per feature")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    apply_runtime(cfg)
    g = cfg["density_grid"]
    dens_grid = log_density_grid(g["start"], g["stop"], g["num"])
    if g.get("keep") is not None:  # subset of grid indices, bitwise identical to the full grid's values
        dens_grid = dens_grid[np.asarray(g["keep"], dtype=int)]
    density, sweep_idx, seed_idx = expand_sweep(dens_grid, cfg["n_seeds"])
    model, sampler = build_model_and_sampler(cfg, density)
    importance = importance_curve(cfg["n_features"], cfg["importance_r"])
    tcfg = train_config_from(cfg, args.steps)
    run_dir = make_run_dir(cfg["experiment"], args.tag)
    save_provenance(
        run_dir,
        cfg,
        {"steps_effective": tcfg.steps, "K": int(len(density)), "density_grid": dens_grid.tolist()},
    )
    print(f"[rep_dims] K={len(density)} instances, {tcfg.steps} steps, snapshots at {tcfg.snapshot_steps} -> {run_dir}")

    result = train(model, sampler, torch.tensor(importance), tcfg, desc="rep_dims")
    W, b = model.W.detach(), model.b.detach()
    save_weights(run_dir, "final", W, b, density=density, sweep_idx=sweep_idx, seed_idx=seed_idx, importance=importance)
    for step, (Ws, bs) in result.snapshots.items():
        save_weights(run_dir, f"snapshot_{step}", Ws, bs, density=density, sweep_idx=sweep_idx, seed_idx=seed_idx)
    np.save(run_dir / "loss_history.npy", result.loss_history)
    df = feature_frame(W, b, density, sweep_idx, seed_idx, importance)
    df.to_parquet(run_dir / "features.parquet", index=False)

    Wn = W.numpy()
    dstar = dims_per_feat(Wn)
    inst = pd.DataFrame(
        {
            "instance": np.arange(len(density)),
            "S": 1.0 - density,
            "density": density,
            "seed_idx": seed_idx,
            "final_loss": result.loss_history[-100:].mean(axis=0).astype(np.float64),
            "plateau_ratio": plateau_ratio(result.loss_history).astype(np.float64),
            "n_represented": (feature_norms(Wn) > 0.5).sum(axis=1),
            "dstar": dstar,
            "sum_D": np.nansum(dimensionality(Wn), axis=1),
        }
    )
    dstar_snap = None
    if result.snapshots:
        step = max(result.snapshots)
        dstar_snap = dims_per_feat(result.snapshots[step][0].numpy())
        inst["dstar_snapshot"] = dstar_snap
        inst["snapshot_step"] = step
    inst.to_parquet(run_dir / "instances.parquet", index=False)

    a = cfg["acceptance"]
    half = sticky_runs(dstar, sweep_idx, 0.5, a["sticky_tol"], a["sticky_min_seeds"])
    one = sticky_runs(dstar, sweep_idx, 1.0, a["sticky_tol"], a["sticky_min_seeds"])
    acc = {
        "sticky_half_sweep_idx": half,
        "sticky_half_S": [float(1 - dens_grid[i]) for i in half],
        "sticky_half_pass": has_adjacent_pair(half),
        "sticky_one_sweep_idx": one,
        "sticky_one_S": [float(1 - dens_grid[i]) for i in one],
        # LOG pre-registration: "the same holds ... at the dense end" = an adjacent pair starting at S = 0.
        "sticky_one_pass": {0, 1} <= set(one),
        "dstar_by_S": {f"{1 - d:.4g}": np.round(dstar[sweep_idx == i], 3).tolist() for i, d in enumerate(dens_grid)},
        "plateau_ratio_median_by_S": {
            f"{1 - d:.4g}": float(np.median(inst.plateau_ratio[sweep_idx == i])) for i, d in enumerate(dens_grid)
        },
        "dstar_final_minus_snapshot_by_S": (
            {
                f"{1 - d:.4g}": np.round(dstar[sweep_idx == i] - dstar_snap[sweep_idx == i], 3).tolist()
                for i, d in enumerate(dens_grid)
            }
            if dstar_snap is not None
            else None
        ),
        "train_seconds": result.seconds,
    }
    write_json(run_dir / "acceptance.json", acc)
    prefix = cfg.get("figure_prefix", cfg["experiment"])
    fig_dir = run_figure_dir(run_dir)

    plot_dims_per_feature(
        density,
        dstar,
        fig_dir / f"{prefix}_dstar",
        dstar_snapshot=dstar_snap,
        snapshot_label=f"step {max(result.snapshots)} snapshot" if result.snapshots else "",
    )
    plot_feature_dims_scatter(df["density"].to_numpy(), df["D"].to_numpy(), fig_dir / f"{prefix}_feature_D")
    plot_loss_curves(
        result.loss_history,
        density,
        sweep_idx,
        fig_dir / f"{prefix}_loss",
        title="Setting C: training loss",
    )
    publish_figures(fig_dir, args.tag)

    print("\n=== Setting C acceptance ===")
    print(f"sticky near 1/2 at S = {acc['sticky_half_S']} (pass: {acc['sticky_half_pass']})")
    print(f"sticky near 1 at S = {acc['sticky_one_S']} (pass: {acc['sticky_one_pass']})")
    for S, v in acc["dstar_by_S"].items():
        print(f"S={S}: D* {v}, plateau {acc['plateau_ratio_median_by_S'][S]:.3f}")
    print(f"train time {result.seconds:.1f}s; results in {run_dir}")


if __name__ == "__main__":
    main()

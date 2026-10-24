"""Setting B (n=20, m=5, importance 0.7^i): the paper's main superposition demonstration.

Acceptance (fixed in notes/LOG.md before running):
- Spearman correlation between S and the number of represented features (mean over seeds) > 0.9.
- Where superposition appears (any represented feature with D_i < 0.9), the mean bias of represented features < 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from ..metrics import dimensionality, dims_per_feat, feature_norms, superposition_score
from ..plots import plot_count_vs_sparsity, plot_gram_and_bias, plot_loss_curves, plot_norms_by_superposition
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


def acceptance(W: np.ndarray, b: np.ndarray, density: np.ndarray, sweep_idx: np.ndarray, acc_cfg: dict) -> dict:
    norms = feature_norms(W)
    D = dimensionality(W)
    rep = norms > 0.5
    sweeps = np.unique(sweep_idx)
    S_list, n_rep_mean, by_S = [], [], {}
    for sw in sweeps:
        ks = np.flatnonzero(sweep_idx == sw)
        S = float(1.0 - density[ks[0]])
        n_rep = rep[ks].sum(axis=1)
        sup = [
            bool(np.nanmin(np.where(rep[k], D[k], np.nan)) < acc_cfg["superposition_d_max"]) if rep[k].any() else False
            for k in ks
        ]
        mean_bias_rep = [float(b[k][rep[k]].mean()) if rep[k].any() else float("nan") for k in ks]
        by_S[f"{S:.4g}"] = {
            "n_represented": n_rep.tolist(),
            "superposition_present": sup,
            "mean_bias_of_represented": [round(v, 4) for v in mean_bias_rep],
            "bias_negative_where_superposition": [bool(mb < 0) for mb, s in zip(mean_bias_rep, sup, strict=True) if s],
        }
        S_list.append(S)
        n_rep_mean.append(float(n_rep.mean()))
    rho = float(spearmanr(S_list, n_rep_mean).statistic)
    flags = [f for v in by_S.values() for f in v["bias_negative_where_superposition"]]
    return {
        "spearman_S_vs_n_represented": rho,
        "spearman_pass": bool(rho > acc_cfg["spearman_min"]),
        "n_instances_with_superposition": len(flags),
        "n_with_negative_mean_bias": int(sum(flags)),
        "bias_pass": bool(len(flags) > 0 and all(flags)),
        "by_S": by_S,
    }


def main() -> None:
    args = parse_args("Setting B replication: 20 features in 5 dimensions")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    apply_runtime(cfg)
    g = cfg["density_grid"]
    dens_grid = log_density_grid(g["start"], g["stop"], g["num"])
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
    print(f"[rep_20in5] K={len(density)} instances, {tcfg.steps} steps -> {run_dir}")

    result = train(model, sampler, torch.tensor(importance), tcfg, desc="rep_20in5")
    W, b = model.W.detach(), model.b.detach()
    save_weights(run_dir, "final", W, b, density=density, sweep_idx=sweep_idx, seed_idx=seed_idx, importance=importance)
    np.save(run_dir / "loss_history.npy", result.loss_history)
    df = feature_frame(W, b, density, sweep_idx, seed_idx, importance)
    df.to_parquet(run_dir / "features.parquet", index=False)

    Wn, bn = W.numpy(), b.numpy()
    inst = pd.DataFrame(
        {
            "instance": np.arange(len(density)),
            "S": 1.0 - density,
            "seed_idx": seed_idx,
            "final_loss": result.loss_history[-100:].mean(axis=0).astype(np.float64),
            "plateau_ratio": plateau_ratio(result.loss_history).astype(np.float64),
            "n_represented": (feature_norms(Wn) > 0.5).sum(axis=1),
            "dstar": dims_per_feat(Wn),
            "sum_D": np.nansum(dimensionality(Wn), axis=1),
        }
    )
    inst.to_parquet(run_dir / "instances.parquet", index=False)

    acc = acceptance(Wn, bn, density, sweep_idx, cfg["acceptance"])
    acc["train_seconds"] = result.seconds
    acc["plateau_ratio_median_by_S"] = {
        f"{S:.4g}": round(float(v), 3) for S, v in inst.groupby("S")["plateau_ratio"].median().items()
    }
    write_json(run_dir / "acceptance.json", acc)

    fig_dir = run_figure_dir(run_dir)
    seed0 = np.flatnonzero(seed_idx == 0)
    plot_gram_and_bias(Wn, bn, density, seed0, fig_dir / "rep_20in5_gram_bias")
    plot_norms_by_superposition(Wn, density, seed0, superposition_score(Wn), fig_dir / "rep_20in5_norms")
    plot_count_vs_sparsity(density, inst["n_represented"].to_numpy(), fig_dir / "rep_20in5_count")
    plot_loss_curves(
        result.loss_history,
        density,
        sweep_idx,
        fig_dir / "rep_20in5_loss",
        title="Setting B: training loss",
    )
    publish_figures(fig_dir, args.tag)

    print("\n=== Setting B acceptance ===")
    print(f"Spearman(S, n_represented) = {acc['spearman_S_vs_n_represented']:.3f} (pass: {acc['spearman_pass']})")
    print(
        f"bias < 0 where superposition: {acc['n_with_negative_mean_bias']}/"
        f"{acc['n_instances_with_superposition']} (pass: {acc['bias_pass']})"
    )
    for S, v in acc["by_S"].items():
        print(
            f"S={S}: n_rep {v['n_represented']}, mean bias(rep) {v['mean_bias_of_represented']}, "
            f"superposition {v['superposition_present']}"
        )
    print(f"train time {result.seconds:.1f}s; results in {run_dir}")


if __name__ == "__main__":
    main()

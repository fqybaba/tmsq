"""Setting A (n=5, m=2, importance 0.9^i): visual sanity check across sparsity values.

Acceptance (fixed in notes/LOG.md before running):
- S = 0: exactly 2 features with norm > 0.5, |cos| < 0.1 between them, the other 3 with norm < 0.1.
- high S: fraction of seeds forming a pentagon (all norms > 0.5 and all D_i within 0.4 +- 0.05). Report only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..metrics import angles_2d, dimensionality, dims_per_feat, feature_norms, is_pentagon
from ..plots import plot_arrows_grid, plot_loss_curves
from ..train import plateau_ratio, train
from .common import (
    apply_runtime,
    build_model_and_sampler,
    expand_sweep,
    feature_frame,
    importance_curve,
    load_config,
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


def acceptance(W: np.ndarray, density: np.ndarray, sweep_idx: np.ndarray, seed_idx: np.ndarray, acc_cfg: dict) -> dict:
    norms = feature_norms(W)
    out: dict = {"dense": {}, "pentagon_fraction_by_S": {}, "n_represented_by_S": {}}
    dense = np.flatnonzero(np.isclose(density, 1.0))
    per_seed = []
    for k in dense:
        rep = np.flatnonzero(norms[k] > 0.5)
        other_ok = bool((norms[k][norms[k] <= 0.5] < acc_cfg["dense_other_norm_max"]).all()) if len(rep) < 5 else False
        orth_ok = False
        if len(rep) == 2:
            u, v = W[k][:, rep[0]], W[k][:, rep[1]]
            cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
            orth_ok = abs(cos) < acc_cfg["dense_orthogonal_cos_max"]
        per_seed.append(
            {
                "seed_idx": int(seed_idx[k]),
                "n_represented": int(len(rep)),
                "represented": [int(i) for i in rep],
                "two_orthogonal": bool(orth_ok),
                "others_near_zero": bool(other_ok),
                "pass": bool(len(rep) == 2 and orth_ok and other_ok),
            }
        )
    out["dense"] = {
        "per_seed": per_seed,
        "n_pass": int(sum(p["pass"] for p in per_seed)),
        "n_seeds": len(per_seed),
    }
    pent = is_pentagon(W, d_tol=acc_cfg["pentagon_d_tol"])
    for sw in np.unique(sweep_idx):
        ks = np.flatnonzero(sweep_idx == sw)
        S = float(1.0 - density[ks[0]])
        out["pentagon_fraction_by_S"][f"{S:g}"] = float(pent[ks].mean())
        out["n_represented_by_S"][f"{S:g}"] = [int((norms[k] > 0.5).sum()) for k in ks]
    return out


def adjacent_gaps_by_S(W: np.ndarray, density: np.ndarray, sweep_idx: np.ndarray) -> dict[str, list[list[float]]]:
    """Sorted angular gaps (degrees) between adjacent REPRESENTED columns, per instance, grouped by S.

    A regular pentagon gives five gaps of 72; this is the angle view that complements the D_i-based
    ``is_pentagon`` (a looser proxy, see CLAUDE.md design note 9).
    """
    ang = np.degrees(angles_2d(W))
    rep = feature_norms(W) > 0.5
    out: dict[str, list[list[float]]] = {}
    for sw in np.unique(sweep_idx):
        ks = np.flatnonzero(sweep_idx == sw)
        S = float(1.0 - density[ks[0]])
        rows = []
        for k in ks:
            a = np.sort(ang[k][rep[k]])
            gaps = np.diff(np.concatenate([a, [a[0] + 360.0]])) if len(a) > 1 else np.array([])
            rows.append([round(float(g), 1) for g in gaps])
        out[f"{S:g}"] = rows
    return out


def main() -> None:
    args = parse_args("Setting A replication: 5 features in 2 dimensions")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    apply_runtime(cfg)
    S_values = np.asarray(cfg["S_values"], dtype=np.float64)
    density, sweep_idx, seed_idx = expand_sweep(1.0 - S_values, cfg["n_seeds"])
    model, sampler = build_model_and_sampler(cfg, density)
    importance = importance_curve(cfg["n_features"], cfg["importance_r"])
    tcfg = train_config_from(cfg, args.steps)
    run_dir = make_run_dir(cfg["experiment"], args.tag)
    save_provenance(run_dir, cfg, {"steps_effective": tcfg.steps, "K": int(len(density))})
    print(f"[rep_5in2] K={len(density)} instances, {tcfg.steps} steps -> {run_dir}")

    result = train(model, sampler, torch.tensor(importance), tcfg, desc="rep_5in2")
    W, b = model.W.detach(), model.b.detach()
    save_weights(run_dir, "final", W, b, density=density, sweep_idx=sweep_idx, seed_idx=seed_idx, importance=importance)
    np.save(run_dir / "loss_history.npy", result.loss_history)
    df = feature_frame(W, b, density, sweep_idx, seed_idx, importance)
    df.to_parquet(run_dir / "features.parquet", index=False)

    Wn = W.numpy()
    inst = pd.DataFrame(
        {
            "instance": np.arange(len(density)),
            "S": 1.0 - density,
            "seed_idx": seed_idx,
            "final_loss": result.loss_history[-100:].mean(axis=0).astype(np.float64),
            "plateau_ratio": plateau_ratio(result.loss_history).astype(np.float64),
            "n_represented": (feature_norms(Wn) > 0.5).sum(axis=1),
            "dstar": dims_per_feat(Wn),
            "pentagon": is_pentagon(Wn, d_tol=cfg["acceptance"]["pentagon_d_tol"]),
            "sum_D": np.nansum(dimensionality(Wn), axis=1),
        }
    )
    inst.to_parquet(run_dir / "instances.parquet", index=False)

    acc = acceptance(Wn, density, sweep_idx, seed_idx, cfg["acceptance"])
    acc["train_seconds"] = result.seconds
    acc["plateau_ratio_by_S"] = {
        f"{S:g}": inst.loc[inst.S.round(6) == round(S, 6), "plateau_ratio"].round(3).tolist() for S in S_values
    }
    acc["adjacent_gaps_deg_by_S"] = adjacent_gaps_by_S(Wn, density, sweep_idx)
    write_json(run_dir / "acceptance.json", acc)

    fig_dir = run_figure_dir(run_dir)
    plot_arrows_grid(Wn, density, sweep_idx, seed_idx, fig_dir / "rep_5in2_arrows")
    plot_loss_curves(
        result.loss_history,
        density,
        sweep_idx,
        fig_dir / "rep_5in2_loss",
        title="Setting A: training loss",
    )
    publish_figures(fig_dir, args.tag)

    print("\n=== Setting A acceptance ===")
    print(f"S=0: {acc['dense']['n_pass']}/{acc['dense']['n_seeds']} seeds pass (2 represented, orthogonal, others ~0)")
    for S, frac in acc["pentagon_fraction_by_S"].items():
        print(
            f"S={S}: pentagon fraction {frac:.2f}, n_represented {acc['n_represented_by_S'][S]}, "
            f"plateau ratio {acc['plateau_ratio_by_S'][S]}"
        )
    print(f"train time {result.seconds:.1f}s; results in {run_dir}")


if __name__ == "__main__":
    main()

"""Train setting D (Phase 4): power-law feature frequencies and Liu, Liu & Gore's weight-decay knob.

``python -m tmsq.experiments.train_setting_d --config configs/setting_d.yaml --gamma 0.1``
One run per gamma, so each stays within the ~10-minute budget. Instances are the product
(alpha, k, seed) for the chosen gamma.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data import FeatureSampler, power_law_frequencies
from ..metrics import dims_per_feat, feature_norms, geometry_table
from ..model import ToyModel
from ..plots import CATEGORICAL, INK, INK2, save, style
from ..train import plateau_ratio, train
from .common import (
    apply_runtime,
    load_config,
    make_run_dir,
    publish_figures,
    run_figure_dir,
    save_provenance,
    save_weights,
    set_seed,
    train_config_from,
    write_json,
)


def plot_norm_vs_rank(W: np.ndarray, meta: pd.DataFrame, stem: Path, title: str) -> None:
    """||W_i|| against frequency rank i (log x), one panel per (alpha, k), all seeds overlaid."""
    import matplotlib.pyplot as plt

    style()
    combos = meta[["alpha", "k"]].drop_duplicates().to_records(index=False)
    fig, axes = plt.subplots(1, len(combos), figsize=(3.4 * len(combos), 3.0), squeeze=False, sharey=True)
    norms = feature_norms(W)
    rank = np.arange(W.shape[2]) + 1
    for ax, (a, k) in zip(axes[0], combos, strict=True):
        for idx in meta.index[(meta.alpha == a) & (meta.k == k)]:
            ax.plot(rank, norms[idx], color=CATEGORICAL[0], alpha=0.5, lw=1)
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("frequency rank i + 1 (1 = most frequent)")
        ax.set_title(f"alpha={a}, k={k}", fontsize=9)
    axes[0, 0].set_ylabel("||W_i||")
    fig.suptitle(title, fontsize=10, color=INK)
    save(fig, stem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train setting D")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--gamma", type=float, default=None, help="one of the config gammas (main runs)")
    ap.add_argument(
        "--gamma-list",
        type=str,
        default=None,
        help="comma-separated gammas for a regime sweep (FP only, tagged runs); base seed 900",
    )
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="override n_seeds (pilot)")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gammas = [float(g) for g in cfg["gammas"]]
    if args.gamma_list:
        run_gammas = [float(x) for x in args.gamma_list.split(",")]
        seed = 900
        label = "gsweep"
    else:
        if args.gamma not in gammas:
            raise SystemExit(f"--gamma must be one of {gammas}")
        run_gammas = [args.gamma]
        seed = int(cfg["seed"]) + 100 * gammas.index(args.gamma)
        label = f"g{args.gamma:+g}"
    cfg_run = {**cfg, "seed": seed, "gamma_run": run_gammas}
    set_seed(seed)
    apply_runtime(cfg)
    n, m = int(cfg["n_features"]), int(cfg["n_hidden"])
    n_seeds = args.seeds or int(cfg["n_seeds"])
    rows, P = [], []
    for gam, alpha, k, s in itertools.product(run_gammas, cfg["alphas"], cfg["ks"], range(n_seeds)):
        p, c = power_law_frequencies(n, float(alpha), float(k))
        rows.append({"alpha": float(alpha), "k": float(k), "gamma": gam, "seed_idx": s, "c": c,
                     "p_max": float(p.max()), "n_clipped": int((p >= 1.0).sum())})  # fmt: skip
        P.append(p)
    meta = pd.DataFrame(rows)
    P = np.stack(P)  # [K, n]
    K = len(meta)
    model = ToyModel(K, n, m, seed=seed, init_std=cfg.get("init_std"))
    sampler = FeatureSampler(n, torch.as_tensor(P, dtype=torch.float32), n_instances=K, seed=seed + 1)
    tcfg = train_config_from(cfg, args.steps)
    run_dir = make_run_dir(cfg["experiment"], f"{label}{'-' + args.tag if args.tag else ''}")
    save_provenance(run_dir, cfg_run, {"steps_effective": tcfg.steps, "K": K, "gammas": run_gammas, "seed": seed})
    print(f"[setting_d {label}] K={K}, gammas {run_gammas}, {tcfg.steps} steps, seed {seed} -> {run_dir}")

    gamma_vec = torch.as_tensor(meta.gamma.to_numpy(), dtype=torch.float32)
    result = train(model, sampler, torch.ones(n), tcfg, desc=f"setting_d {label}", gamma=gamma_vec)
    W, b = model.W.detach(), model.b.detach()
    extra = {"density": P, "alpha": meta.alpha.to_numpy(), "k": meta.k.to_numpy(), "gamma": meta.gamma.to_numpy(),
             "seed_idx": meta.seed_idx.to_numpy(), "importance": np.ones(n)}  # fmt: skip
    save_weights(run_dir, "final", W, b, **extra)
    for step, (Ws, bs) in result.snapshots.items():
        save_weights(run_dir, f"snapshot_{step}", Ws, bs, **extra)
    np.save(run_dir / "loss_history.npy", result.loss_history)

    Wn = W.numpy()
    norms = feature_norms(Wn)
    meta["instance"] = np.arange(K)
    meta["phi_half"] = (norms > 0.5).mean(axis=1)
    meta["n_represented"] = (norms > 0.5).sum(axis=1)
    meta["dstar"] = dims_per_feat(Wn)
    meta["final_loss"] = result.loss_history[-100:].mean(axis=0).astype(np.float64)
    meta["plateau_ratio"] = plateau_ratio(result.loss_history).astype(np.float64)
    if result.snapshots:
        Wh = result.snapshots[max(result.snapshots)][0].numpy()
        meta["phi_half_snapshot"] = (feature_norms(Wh) > 0.5).mean(axis=1)
    meta.to_parquet(run_dir / "instances.parquet", index=False)
    g = geometry_table(W, b)
    feat = pd.DataFrame({"instance": np.repeat(np.arange(K), n), "feature": np.tile(np.arange(n), K),
                         "p": P.reshape(-1), **{key: np.asarray(v).reshape(-1) for key, v in g.items()}})  # fmt: skip
    for col in ("alpha", "k", "gamma", "seed_idx"):
        feat[col] = np.repeat(meta[col].to_numpy(), n)
    feat.to_parquet(run_dir / "features.parquet", index=False)
    summary = meta.groupby(["gamma", "alpha", "k"]).agg(phi_half=("phi_half", "mean"), phi_sd=("phi_half", "std"),
                                               n_rep=("n_represented", "mean"), plateau=("plateau_ratio", "median"),
                                               dstar=("dstar", "mean")).round(4)  # fmt: skip
    write_json(
        run_dir / "summary.json",
        {"train_seconds": result.seconds, "by_alpha_k": summary.reset_index().to_dict(orient="records")},
    )
    fig_dir = run_figure_dir(run_dir)
    for gam in run_gammas:
        sub = meta[meta.gamma == gam]
        plot_norm_vs_rank(Wn[sub.index.to_numpy()], sub.reset_index(drop=True),
                          fig_dir / f"setting_d_g{gam:+g}_norm_vs_rank",
                          title=f"Setting D, gamma = {gam:+g}: feature norm vs frequency rank")  # fmt: skip
    publish_figures(fig_dir, args.tag)
    print(summary.to_string())
    print(f"m/n = {m / n:.3f}; train time {result.seconds:.0f}s; results in {run_dir}")


if __name__ == "__main__":
    main()

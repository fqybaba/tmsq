"""Phase 2: quantization sweep over the trained Phase 1 models of one setting.

``python -m tmsq.experiments.quant_sweep --config configs/quant_sweep.yaml --setting C``
Writes results/quant_sweep/<timestamp>-<setting>/{features.parquet, models.parquet, sanity.json, ...}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..analysis import EvalParams, build_eval_set, eval_seed, evaluate_instance
from ..plots_quant import plot_arrows_fp_vs_q, plot_metric_vs_bits, plot_rtn_vs_noise
from ..quantize import build_configs
from .common import (
    PROJECT_ROOT,
    apply_runtime,
    load_config,
    load_weights,
    make_run_dir,
    publish_figures,
    run_figure_dir,
    save_provenance,
    set_seed,
    write_json,
)


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 2 quantization sweep")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--setting", type=str, required=True, choices=["A", "B", "C"])
    ap.add_argument("--bits", type=str, default=None, help="comma-separated override, e.g. 8,4 (smoke)")
    ap.add_argument("--noise-seeds", type=int, default=None, help="override (smoke)")
    ap.add_argument("--max-instances", type=int, default=None, help="override (smoke)")
    ap.add_argument("--tag", type=str, default="")
    return ap.parse_args()


def select_instances(final: dict, S_values: list[float] | None) -> np.ndarray:
    S = 1.0 - np.asarray(final["density"], dtype=np.float64)
    if S_values is None:
        return np.arange(len(S))
    keep = np.zeros(len(S), dtype=bool)
    for s in S_values:
        keep |= np.abs(S - float(s)) < 2e-3
    return np.flatnonzero(keep)


def sanity_checks(feats: pd.DataFrame, models: pd.DataFrame, cfg: dict) -> dict:
    base_f = feats[(feats.method == "rtn") & (feats.ablation == "none") & feats.eligible]
    base_m = models[(models.method == "rtn") & (models.ablation == "none")]
    out: dict = {}
    if 8 in set(base_f.bits):
        out["bits8_median_abs_dR2"] = float(base_f.loc[base_f.bits == 8, "dR2"].abs().median())
        out["bits8_max_loss_increase_rel"] = float(base_m.loc[base_m.bits == 8, "loss_increase_rel"].max())
        out["bits8_pass"] = bool(
            out["bits8_median_abs_dR2"] < cfg["sanity"]["bits8_median_abs_dR2_max"]
            and out["bits8_max_loss_increase_rel"] < cfg["sanity"]["bits8_loss_increase_rel_max"]
        )
    mono = {}
    for g, grp in base_f.groupby("granularity"):
        by_bits = grp.groupby("bits")["dR2"].mean().sort_index(ascending=False)
        mono[g] = {
            "mean_dR2_by_bits": {int(k): round(float(v), 5) for k, v in by_bits.items()},
            "monotone_nondecreasing_as_bits_decrease": bool(np.all(np.diff(by_bits.to_numpy()) >= -1e-6)),
        }
    out["monotonicity"] = mono
    return out


def main() -> None:
    args = parse()
    cfg = load_config(args.config)
    set_seed(0)
    apply_runtime(cfg)
    s_cfg = cfg["settings"][args.setting]
    src_run = PROJECT_ROOT / s_cfg["run"]
    final = load_weights(src_run / "final.pt")
    src_cfg = load_config(src_run / "config.yaml")
    train_seed = int(src_cfg["seed"])
    W_all, b_all = final["W"], final["b"]
    density = np.asarray(final["density"], dtype=np.float64)
    seed_idx = np.asarray(final["seed_idx"])
    importance = np.asarray(final["importance"], dtype=np.float64)
    inst = select_instances(final, s_cfg.get("S_values"))
    if args.max_instances:
        inst = inst[: args.max_instances]

    bits = [int(x) for x in args.bits.split(",")] if args.bits else [int(x) for x in cfg["bits"]]
    noise_seeds = args.noise_seeds if args.noise_seeds is not None else int(cfg["noise_seeds"])
    configs = build_configs(bits, cfg["granularities"], noise_seeds, cfg["ablations"], cfg["ablation_granularities"])
    params = EvalParams(**cfg["eval"])

    run_dir = make_run_dir(cfg["experiment"], f"{args.setting}{'-' + args.tag if args.tag else ''}")
    save_provenance(
        run_dir,
        cfg,
        {
            "setting": args.setting,
            "source_run": str(src_run),
            "source_meta": json.load(open(src_run / "meta.json", encoding="utf-8")),
            "n_instances": int(len(inst)),
            "n_configs": len(configs),
            "bits": bits,
            "noise_seeds": noise_seeds,
            "eval_params": params.__dict__,
        },
    )
    print(f"[quant_sweep {args.setting}] {len(inst)} instances x {len(configs)} configs -> {run_dir}")

    feat_frames, model_frames = [], []
    t0 = time.perf_counter()
    for k in tqdm(inst, desc=f"quant_sweep {args.setting}", dynamic_ncols=True):
        k = int(k)
        p = float(density[k])
        es = build_eval_set(W_all.shape[2], p, eval_seed(train_seed, k, params), params)
        feats, models = evaluate_instance(W_all[k], b_all[k], es, params, configs, importance, p, instance=k)
        for df in (feats, models):
            df.insert(0, "setting", args.setting)
            df.insert(1, "S", np.float64(1.0 - p))
            df.insert(2, "seed_idx", int(seed_idx[k]))
            df.insert(3, "instance", k)
        feats["n_forced_active"] = es.n_forced_active
        feat_frames.append(feats)
        model_frames.append(models)
    feats = pd.concat(feat_frames, ignore_index=True)
    models = pd.concat(model_frames, ignore_index=True)
    seconds = time.perf_counter() - t0
    feats.to_parquet(run_dir / "features.parquet", index=False)
    models.to_parquet(run_dir / "models.parquet", index=False)

    sanity = sanity_checks(feats, models, cfg)
    sanity["eval_seconds"] = seconds
    sanity["rows"] = int(len(feats))
    write_json(run_dir / "sanity.json", sanity)

    fig_dir = run_figure_dir(run_dir)
    tag = args.setting
    plot_metric_vs_bits(
        models,
        "loss_increase",
        "total weighted loss increase (symlog, linear below 1e-5)",
        fig_dir / f"quant_{tag}_loss_vs_bits",
        title=f"Setting {tag}: loss increase vs bits (RTN)",
        logy=True,
    )
    plot_metric_vs_bits(
        models,
        "survived_frac",
        "survived fraction (eligible features)",
        fig_dir / f"quant_{tag}_survival_vs_bits",
        title=f"Setting {tag}: survival vs bits (RTN)",
        logy=False,
    )
    plot_rtn_vs_noise(models, fig_dir / f"quant_{tag}_rtn_vs_noise", title=f"Setting {tag}: RTN vs matched noise")
    if args.setting == "A":
        plot_arrows_fp_vs_q(
            W_all, b_all, density, seed_idx, [bw for bw in (4, 3, 2) if bw in bits], fig_dir / "quant_A_arrows"
        )
    publish_figures(fig_dir, args.tag)

    print("\n=== Phase 2 sanity ===")
    print(json.dumps({k: v for k, v in sanity.items() if k != "monotonicity"}, indent=1))
    for g, v in sanity["monotonicity"].items():
        print(f"{g}: mean dR2 by bits {v['mean_dR2_by_bits']} monotone={v['monotone_nondecreasing_as_bits_decrease']}")
    print(f"{len(feats)} feature rows, {len(models)} model rows, {seconds:.0f}s; results in {run_dir}")


if __name__ == "__main__":
    main()

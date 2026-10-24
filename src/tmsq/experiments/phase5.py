"""Phase 5 runner: every arm of the pre-registration on one setting's models.

``python -m tmsq.experiments.phase5 --config configs/phase5.yaml --setting C_P5``
Writes results/phase5/<timestamp>-<setting>/{models.parquet, features.parquet, greedy.parquet, sanity.json}.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .. import phase5 as P5
from ..analysis import EvalParams, build_eval_set, build_eval_set_vec, eval_seed
from ..quantize import QuantConfig, apply_quant
from .common import PROJECT_ROOT, load_config, load_weights, make_run_dir, save_provenance, set_seed, write_json
from .quant_sweep import select_instances


def source_run(s_cfg: dict) -> Path:
    if "run" in s_cfg:
        return PROJECT_ROOT / s_cfg["run"]
    found = [p for p in PROJECT_ROOT.glob(s_cfg["run_glob"]) if p.is_dir() and p.name.count("-") == 1]
    if len(found) != 1:
        raise SystemExit(f"expected exactly one untagged run for {s_cfg['run_glob']}, found {[p.name for p in found]}")
    return found[0]


def run_instance(W, b, p, importance, train_seed: int, k: int, params: EvalParams) -> tuple[list, list, list, bool]:
    n = W.shape[1]
    builder = build_eval_set_vec if np.ndim(p) else build_eval_set
    es = builder(n, p, eval_seed(train_seed, k, params), params)
    ref = P5.fp_reference(W, b, es, params, p, importance)
    Xcal = P5.calibration_set(n, p, P5.cal_seed(train_seed, k))
    rep = ref["geom"]["represented"]
    keys = {"margin": ref["geom"]["margin"], "norm": ref["geom"]["norm"],
            "near_thr_cal": P5.near_thr_cal(W, b, Xcal, params.tau, params.near_eps)}  # fmt: skip
    arms: list[tuple[str, int, np.ndarray | None, torch.Tensor]] = []
    for ub in (4, 3, 5):
        bits = np.full(n, ub, dtype=np.int64)
        arms.append((f"U{ub}", -1, bits, P5.rtn_mixed(W, bits)[0]))
    reproduces = bool(torch.equal(arms[0][3], apply_quant(QuantConfig(4, "column"), W[None], b[None]).Wq[0]))
    for arm, (key, desc) in P5.ALLOC_ARMS.items():
        bits = P5.allocation_bits(keys[key], rep, desc)
        arms.append((arm, -1, bits, P5.rtn_mixed(W, bits)[0]))
    for d in range(P5.N_RANDOM):
        bits = P5.random_bits(rep, train_seed, k, d)
        arms.append(("random", d, bits, P5.rtn_mixed(W, bits)[0]))
    greedy_rows = []
    cols = np.flatnonzero(rep)
    for arm, obj in (("leak_round", "leak"), ("loss_round", "loss")):
        t0 = time.perf_counter()
        Q, info = P5.greedy_round(W, b, Xcal, importance, obj, cols, bits=4)
        greedy_rows.append({"instance": k, "arm": arm, "objective": obj, "J0": info["J0"], "J_final": info["J_final"],
                            "J_tracked": info["J_tracked"], "n_flips": info["n_flips"],
                            "n_columns": len(cols), "seconds": time.perf_counter() - t0})  # fmt: skip
        arms.append((arm, -1, np.full(n, 4, dtype=np.int64), Q))
    model_rows, feat_frames = [], []
    for arm, draw, bits, Q in arms:
        mdl, ft = P5.evaluate_q(Q, b, es, params, p, importance, ref)
        model_rows.append({"instance": k, "arm": arm, "draw": draw, **mdl})
        ft.insert(0, "instance", k)
        ft.insert(1, "arm", arm)
        ft.insert(2, "draw", draw)
        ft["bits_col"] = bits
        ft["p"] = np.broadcast_to(np.asarray(p, dtype=np.float64), (n,))
        ft["represented"] = rep
        for key, v in keys.items():
            ft[key] = v
        feat_frames.append(ft)
    return model_rows, feat_frames, greedy_rows, reproduces


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 arms")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(0)
    s_cfg = cfg["settings"][args.setting]
    src = source_run(s_cfg)
    final = load_weights(src / "final.pt")
    train_seed = int(load_config(src / "config.yaml")["seed"])
    density = np.asarray(final["density"], dtype=np.float64)
    importance = np.asarray(final["importance"], dtype=np.float64)
    inst = select_instances(final, s_cfg.get("S_values"))
    if args.max_instances:
        inst = inst[: args.max_instances]
    params = EvalParams(**cfg["eval"])
    run_dir = make_run_dir("phase5", f"{args.setting}{'-' + args.tag if args.tag else ''}")
    save_provenance(run_dir, cfg, {"setting": args.setting, "source_run": str(src), "train_seed": train_seed,
                                   "instances": [int(k) for k in inst]})  # fmt: skip
    print(f"[phase5 {args.setting}] {len(inst)} instances from {src.name} -> {run_dir}")
    models, feats, greedy, repro = [], [], [], {}
    t0 = time.perf_counter()
    for k in tqdm(inst, desc=f"phase5 {args.setting}", dynamic_ncols=True):
        k = int(k)
        p = density[k]
        W, b = final["W"][k].float(), final["b"][k].float()
        m, f, g, ok = run_instance(W, b, p, importance, train_seed, k, params)
        S = np.nan if np.ndim(p) else 1.0 - float(p)
        for row in m:
            row["S"] = S
        for fr in f:
            fr["S"] = S
        for row in g:
            row["S"] = S
        models += m
        feats += f
        greedy += g
        repro[k] = ok
    models_df = pd.DataFrame(models)
    for key in ("alpha", "k", "gamma"):
        if key in final:
            models_df[key] = [float(np.asarray(final[key])[i]) for i in models_df.instance]
    models_df.to_parquet(run_dir / "models.parquet", index=False)
    feats_df = pd.concat(feats, ignore_index=True)
    feats_df.to_parquet(run_dir / "features.parquet", index=False)
    greedy_df = pd.DataFrame(greedy)
    greedy_df.to_parquet(run_dir / "greedy.parquet", index=False)
    sanity = {
        "u4_reproduces_phase2_column_rtn": bool(all(repro.values())),
        "greedy_objective_never_increased": bool((greedy_df.J_final <= greedy_df.J0 + 1e-12).all()),
        "greedy_max_abs_tracking_error": float((greedy_df.J_final - greedy_df.J_tracked).abs().max()),
        "n_instances": len(inst),
        "seconds": time.perf_counter() - t0,
    }
    if "sweep" in s_cfg:  # cross-check U4 against the stored Phase 4 sweep (same models, same eval seeds)
        sw = pd.read_parquet(PROJECT_ROOT / s_cfg["sweep"] / "models.parquet")
        sw = sw[(sw.bits == 4) & (sw.granularity == "column") & (sw.method == "rtn") & (sw.ablation == "none")]
        u4 = models_df[models_df.arm == "U4"].set_index("instance").loss_q
        diff = (sw.set_index("instance").loss_q.reindex(u4.index) - u4).abs() / u4
        sanity["u4_vs_sweep_max_rel_diff"] = float(diff.max())
    sanity["stop"] = not (sanity["u4_reproduces_phase2_column_rtn"] and sanity["greedy_objective_never_increased"])
    write_json(run_dir / "sanity.json", sanity)
    print(sanity)
    print(greedy_df.groupby("arm")[["n_flips", "seconds"]].mean().round(2).to_string())
    print(f"results in {run_dir}")


if __name__ == "__main__":
    main()

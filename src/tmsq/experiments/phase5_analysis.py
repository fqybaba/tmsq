"""Phase 5 analysis (frozen with the pre-registration): primary family (arms vs U4), second family (allocation
vs RANDOM), secondaries (BH), registered mechanism checks, figures.

``python -m tmsq.experiments.phase5_analysis --config configs/phase5.yaml [--precheck-only]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import phase3 as P3
from ..phase5 import ALLOC_ARMS, N_RANDOM
from ..plots import CATEGORICAL, GRID, INK, MUTED, save, style
from .common import (
    PROJECT_ROOT,
    load_config,
    make_run_dir,
    publish_figures,
    run_figure_dir,
    save_provenance,
    write_json,
)
from .phase3 import _clean

PRIMARY_CELL = ("C_P5", 0.7488)
PRIMARY_ARMS = ("margin_low", "near_thr", "leak_round")
SECOND_ARMS = tuple(ALLOC_ARMS)
COMPARED = (*ALLOC_ARMS, "random", "leak_round", "loss_round")
OUTCOMES = ("r", "composite_frac", "worst_dR2", "mean_dR2")
MIN_VALID = 8
EXPECTED_ARMS = {"U4": 1, "U3": 1, "U5": 1, **{a: 1 for a in ALLOC_ARMS}, "random": N_RANDOM, "leak_round": 1,
                 "loss_round": 1}  # fmt: skip
EXPECTED_INSTANCES = {"C_P5": 40, "D_strong": 20}


def load_run(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    m = pd.read_parquet(path / "models.parquet")
    f = pd.read_parquet(path / "features.parquet")
    g = pd.read_parquet(path / "greedy.parquet")
    s = json.load(open(path / "sanity.json", encoding="utf-8"))
    return m, f, g, s


def cells(m: pd.DataFrame, setting: str) -> dict:
    """Cell key -> model rows. Setting C: one cell per S (rounded to 4 decimals); setting D: one cell."""
    if setting == "D_strong":
        return {(setting, "all"): m}
    return {(setting, round(float(S), 4)): d for S, d in m.groupby("S")}


def per_instance(m: pd.DataFrame, arm: str, outcome: str, versus: str) -> dict[int, float]:
    """Per-instance paired difference outcome(arm) - outcome(versus); RANDOM = mean over its draws."""
    a = m[m.arm == arm].groupby("instance")[outcome].mean()
    b = m[m.arm == versus].groupby("instance")[outcome].mean()
    d = (a - b).dropna()
    return {int(k): float(v) for k, v in d.items()}


def category(s: dict, p_holm: float, lower_is_better: bool, ref: str) -> str:
    if not s["summarised"]:
        return "Not testable (too few valid instances)"
    if not p_holm < P3.ALPHA:
        return "no detectable difference"
    better = (s["mean"] < 0) if lower_is_better else (s["mean"] > 0)
    return f"{'better' if better else 'worse'} than {ref}"


def precheck(runs: dict) -> dict:
    out = {}
    for setting, (m, _, g, s) in runs.items():
        counts = m.groupby(["instance", "arm"]).size().unstack(fill_value=0)
        problems = []
        if m.instance.nunique() != EXPECTED_INSTANCES[setting]:
            problems.append(f"{m.instance.nunique()} instances, expected {EXPECTED_INSTANCES[setting]}")
        for arm, c in EXPECTED_ARMS.items():
            if arm not in counts or (counts[arm] != c).any():
                problems.append(f"arm {arm}: expected {c} rows per instance")
        if setting == "C_P5" and (m.groupby("S").instance.nunique() != 10).any():
            problems.append("C_P5: not 10 instances per S")
        out[setting] = {"stop": bool(s.get("stop", True)), "sanity": s, "completeness_problems": problems,
                        "greedy_mean_flips": g.groupby("arm").n_flips.mean().to_dict()}  # fmt: skip
    return out


def mechanism(m: pd.DataFrame, f: pd.DataFrame, g: pd.DataFrame, setting: str) -> list[dict]:
    rows = []
    cell_of = {k: c for c, d in cells(m, setting).items() for k in d.instance.unique()}
    el = f[f.eligible]
    # allocation arms: mean dR2 of eligible features in 5-bit columns minus 3-bit columns, per instance (random:
    # per draw, then averaged over draws)
    for arm in (*ALLOC_ARMS, "random"):
        e = el[el.arm == arm]
        per = {}
        for (inst, _draw), d in e.groupby(["instance", "draw"]):
            h5, h3 = d[d.bits_col == 5].dR2, d[d.bits_col == 3].dR2
            if len(h5) and len(h3):
                per.setdefault(int(inst), []).append(float(h5.mean() - h3.mean()))
        per = {k: float(np.mean(v)) for k, v in per.items()}
        for c in sorted(set(cell_of.values()), key=str):
            s = P3.summarize({k: v for k, v in per.items() if cell_of[k] == c}, MIN_VALID)
            rows.append({"check": "dR2(5-bit half) - dR2(3-bit half)", "setting": c[0], "cell": c[1], "arm": arm,
                         **{k: s[k] for k in ("mean", "ci_lo", "ci_hi", "p", "n_valid")}})  # fmt: skip
    # rounding arms: active vs inactive components of the total loss change, relative to U4 (eval set)
    comp = f.assign(active=f.p * f.dcond_mse, inactive=(1 - f.p) * f.dleak_energy.fillna(0.0))
    tot = comp.groupby(["instance", "arm", "draw"])[["active", "inactive"]].sum().reset_index()
    u4 = tot[tot.arm == "U4"].set_index("instance")
    for arm in ("leak_round", "loss_round"):
        t = tot[tot.arm == arm].set_index("instance")
        for part in ("active", "inactive"):
            per = (t[part] - u4[part]).dropna()
            for c in sorted(set(cell_of.values()), key=str):
                s = P3.summarize({int(k): float(v) for k, v in per.items() if cell_of[int(k)] == c}, MIN_VALID)
                rows.append({"check": f"{part} loss component, arm - U4", "setting": c[0], "cell": c[1], "arm": arm,
                             **{k: s[k] for k in ("mean", "ci_lo", "ci_hi", "p", "n_valid")}})  # fmt: skip
        gg = g[g.arm == arm]
        for c in sorted(set(cell_of.values()), key=str):
            x = gg[gg.instance.map(cell_of) == c]
            rows.append({"check": "greedy: mean flips; calibration objective J_final / J0", "setting": c[0],
                         "cell": c[1], "arm": arm, "mean": float(x.n_flips.mean()),
                         "ci_lo": float((x.J_final / x.J0).mean()), "ci_hi": np.nan, "p": np.nan,
                         "n_valid": len(x)})  # fmt: skip
    return rows


def plot_arms(sec: pd.DataFrame, stem: Path) -> None:
    style()
    t = sec[(sec.outcome == "r") & (sec.versus == "U4")]
    keys = list(dict.fromkeys(zip(t.setting, t.cell, strict=True)))
    fig, axes = plt.subplots(1, len(keys), figsize=(2.9 * len(keys), 3.4), squeeze=False, sharey=False)
    for ax, (st, c) in zip(axes[0], keys, strict=True):
        d = t[(t.setting == st) & (t.cell == c)].set_index("arm").reindex(list(COMPARED))
        x = np.arange(len(d))
        ax.errorbar(x, d["mean"], yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]], fmt="o", ms=4,
                    color=CATEGORICAL[0], capsize=0, elinewidth=1.4)  # fmt: skip
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(list(COMPARED), rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{st} {'S=' + str(c) if st == 'C_P5' else ''}", fontsize=8)
        ax.grid(True, color=GRID)
    axes[0, 0].set_ylabel("r(arm) - r(U4)  [relative loss increase]")
    fig.suptitle("Phase 5: per-column, 4.0 bits/weight; paired difference vs uniform 4-bit, 95% t-interval",
                 fontsize=9, color=INK)  # fmt: skip
    save(fig, stem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 analysis")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir(
        "phase5", f"{'precheck' if args.precheck_only else 'analysis'}{'-' + args.tag if args.tag else ''}"
    )
    save_provenance(run_dir, cfg, {"precheck_only": args.precheck_only})
    runs = {s: load_run(PROJECT_ROOT / p) for s, p in cfg["runs"].items()}
    pre = precheck(runs)
    write_json(run_dir / "precheck.json", _clean(pre))
    for s, v in pre.items():
        print(f"[{s}] stop={v['stop']} problems={v['completeness_problems']} greedy flips={v['greedy_mean_flips']}")
        if v["completeness_problems"]:
            raise SystemExit(f"{s}: incomplete run")
    if args.precheck_only:
        return

    mC = runs["C_P5"][0]
    prim_m = mC[np.isclose(mC.S, PRIMARY_CELL[1], atol=2e-3)]
    stopped = pre["C_P5"]["stop"]
    fam1 = {a: P3.summarize(per_instance(prim_m, a, "r", "U4"), MIN_VALID) for a in PRIMARY_ARMS}
    adj1 = P3.holm({a: fam1[a]["p"] if fam1[a]["summarised"] else float("nan") for a in PRIMARY_ARMS})
    u4_mean = float(prim_m[prim_m.arm == "U4"].r.mean())
    fam2 = {a: P3.summarize(per_instance(prim_m, a, "r", "random"), MIN_VALID) for a in SECOND_ARMS}
    adj2 = P3.holm({a: fam2[a]["p"] if fam2[a]["summarised"] else float("nan") for a in SECOND_ARMS})
    primary = {
        "cell": list(PRIMARY_CELL), "mean_r_U4": u4_mean, "stopped": stopped,
        "family1_vs_U4": {a: {**fam1[a], "p_holm": adj1[a], "relative_size": fam1[a]["mean"] / u4_mean,
                              "category": "STOP" if stopped else category(fam1[a], adj1[a], True, "uniform")}
                          for a in PRIMARY_ARMS},
        "family2_vs_random": {a: {**fam2[a], "p_holm": adj2[a],
                                  "category": "STOP" if stopped else category(fam2[a], adj2[a], True, "random")}
                              for a in SECOND_ARMS},
    }  # fmt: skip
    write_json(run_dir / "primary.json", _clean(primary))

    sec_rows, mech_rows = [], []
    for setting, (m, f, g, _) in runs.items():
        for (st, c), d in cells(m, setting).items():
            for outcome in OUTCOMES:
                for arm in COMPARED:
                    for versus in ("U4", "random"):
                        if versus == "random" and arm not in ALLOC_ARMS:
                            continue
                        s = P3.summarize(per_instance(d, arm, outcome, versus), MIN_VALID)
                        sec_rows.append({"setting": st, "cell": c, "outcome": outcome, "arm": arm, "versus": versus,
                                         **{k: s[k] for k in ("mean", "ci_lo", "ci_hi", "p", "p_signflip", "n_valid",
                                                              "summarised")},
                                         "mean_versus": float(d[d.arm == versus][outcome].mean())})  # fmt: skip
        mech_rows += mechanism(m, f, g, setting)
    sec = pd.DataFrame(sec_rows)
    sec["q_bh"] = np.nan
    ok = sec.summarised.astype(bool)
    sec.loc[ok, "q_bh"] = P3.bh(sec.loc[ok, "p"].to_numpy())
    sec.to_csv(run_dir / "secondary.csv", index=False)
    mech = pd.DataFrame(mech_rows)
    mech.to_csv(run_dir / "mechanism.csv", index=False)
    fig_dir = run_figure_dir(run_dir)
    plot_arms(sec, fig_dir / "phase5_arms_vs_u4")
    publish_figures(fig_dir, args.tag)

    pd.set_option("display.width", 220)
    print(f"\n=== Phase 5 primary cell {PRIMARY_CELL}: mean r(U4) = {u4_mean:.4f} ===")
    for fam, key in (("family 1: arm vs U4", "family1_vs_U4"), ("family 2: allocation vs RANDOM", "family2_vs_random")):
        print(fam)
        for a, v in primary[key].items():
            print(f"  {a:12s} mean d {v['mean']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}] n={v['n_valid']} "
                  f"p={v['p']:.3g} sign-flip p={v['p_signflip']:.3g} Holm p={v['p_holm']:.3g} -> {v['category']}")  # fmt: skip
    print(sec[(sec.outcome == "r")].drop(columns=["outcome", "summarised"]).round(4).to_string(index=False))
    print(mech.round(4).to_string(index=False))
    print(f"results in {run_dir}")


if __name__ == "__main__":
    main()

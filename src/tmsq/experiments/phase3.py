"""Phase 3 entry point: pre-registered tests of H1–H3 on a sweep (E or K), and the E-vs-K comparison.

``python -m tmsq.experiments.phase3 --config configs/phase3.yaml --set E``
``python -m tmsq.experiments.phase3 --config configs/phase3.yaml --set K --precheck-only``
``python -m tmsq.experiments.phase3 --config configs/phase3.yaml --compare results/phase3/<E-run> results/phase3/<K-run>``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import phase3 as P3
from ..plots_phase3 import plot_auc, plot_damage_vs_margin, plot_forest
from .common import (
    PROJECT_ROOT,
    load_config,
    make_run_dir,
    publish_figures,
    run_figure_dir,
    save_provenance,
    write_json,
)


def resolve_sweep(spec: dict) -> Path:
    if spec.get("sweep"):
        return PROJECT_ROOT / spec["sweep"]
    matches = sorted(p for p in PROJECT_ROOT.glob(spec["sweep_glob"]) if p.is_dir() and p.name.count("-") == 2)
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one untagged sweep for {spec['sweep_glob']}, found {[m.name for m in matches]}"
        )
    return matches[0]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _category(verdict: str) -> str:
    return verdict.split(";")[0].split(" (")[0] if verdict else ""


def compare(cfg: dict, e_dir: Path, k_dir: Path, tag: str) -> None:
    """Registered E-vs-K report: categories, mean rho and the disagreement flag, side by side."""
    pe = json.load(open(e_dir / "primary.json", encoding="utf-8"))
    pk = json.load(open(k_dir / "primary.json", encoding="utf-8"))
    rows = []
    for name in ("H1", "H2", "H3", "H2_Dplus"):
        e, k = pe["results"][name], pk["results"][name]
        me, mk = e["summary"]["mean"], k["summary"]["mean"]
        disagree = (_category(e["verdict"]) != _category(k["verdict"])) or (
            me is not None and mk is not None and np.sign(me) != np.sign(mk)
        )
        rows.append({"test": name, "E_mean_rho": me, "E_ci": [e["summary"]["ci_lo"], e["summary"]["ci_hi"]],
                     "E_p_holm": e["p_holm"], "E_verdict": e["verdict"], "K_mean_rho": mk,
                     "K_ci": [k["summary"]["ci_lo"], k["summary"]["ci_hi"]], "K_p_holm": k["p_holm"],
                     "K_verdict": k["verdict"], "disagreement": bool(disagree)})  # fmt: skip
    run_dir = make_run_dir(cfg["experiment"], f"compare{'-' + tag if tag else ''}")
    save_provenance(run_dir, cfg, {"E_run": str(e_dir), "K_run": str(k_dir)})
    pd.DataFrame(rows).to_csv(run_dir / "E_vs_K.csv", index=False)
    fig_dir = run_figure_dir(run_dir)
    plot_forest({"E (exploratory, 5 instances)": pe, "K (confirmatory, 10 instances)": pk}, fig_dir / "phase3_forest_E_vs_K",
                title="Phase 3 primary tests: C, S=0.7488, per-row, 3 bits")  # fmt: skip
    publish_figures(fig_dir, tag)
    for r in rows:
        print(f"{r['test']}: E {r['E_mean_rho']:+.3f} {r['E_verdict']} | K {r['K_mean_rho']:+.3f} {r['K_verdict']} | "
              f"disagreement={r['disagreement']}")  # fmt: skip
    print(f"results in {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 pre-registered tests")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", choices=["E", "K"])
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("E_RUN", "K_RUN"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.compare:
        compare(cfg, PROJECT_ROOT / args.compare[0], PROJECT_ROOT / args.compare[1], args.tag)
        return
    if not args.set:
        raise SystemExit("--set E|K is required unless --compare is given")
    spec = cfg["sets"][args.set]
    sweep = resolve_sweep(spec)
    min_valid = int(spec["min_valid"])
    sanity = json.load(open(sweep / "sanity.json", encoding="utf-8"))
    df = P3.load_features(sweep / "features.parquet")

    tag = f"{args.set}{'-precheck' if args.precheck_only else ''}{'-' + args.tag if args.tag else ''}"
    run_dir = make_run_dir(cfg["experiment"], tag)
    save_provenance(run_dir, cfg, {"set": args.set, "sweep": str(sweep), "min_valid": min_valid,
                                   "precheck_only": args.precheck_only})  # fmt: skip
    pre = P3.precheck(df, sanity, int(spec["instances_per_S"]))
    write_json(run_dir / "precheck.json", _clean(pre))
    print(f"[phase3 {args.set}] sweep {sweep.name} -> {run_dir}")
    if pre["completeness"]["problems"]:
        raise SystemExit(f"sweep is not the registered one: {pre['completeness']['problems']}")
    if pre["label_reproduction_mismatches"]:
        raise SystemExit(f"clarification 6 violated: {pre['label_reproduction_mismatches']} label mismatches")
    print(
        f"precheck: stop={pre['stop']} {pre['stop_reasons']} | bits8_pass={pre['bits8_pass']} monotone={pre['monotone']}"
    )
    print(f"completeness: {pre['completeness']}")
    print(f"ICC_noise primary: {pre['icc_primary']} | rho(norm, D) primary: {pre['rho_norm_D_primary']:.3f} "
          f"({pre['rho_norm_D_n_instances']} instances)")  # fmt: skip
    print(
        "label validity:",
        [(v["S"], v["granularity"], round(v["damaged_frac_8bit"], 3), v["valid"]) for v in pre["label_validity"]],
    )
    print(
        "cells passing:",
        [(c["S"], c["granularity"], c["bits"], c["n_damaged"], c["n_undamaged"]) for c in pre["cells"] if c["passes"]],
    )
    print("eligible per instance (per-row 3b):", pre["eligible_per_instance_row3"])
    if args.precheck_only:
        return

    prim = P3.primary_analysis(df, min_valid, pre)
    out = {k: v for k, v in prim.items() if k != "results"}
    out["results"] = {k: asdict(v) for k, v in prim["results"].items()}
    out_clean = _clean(out)
    write_json(run_dir / "primary.json", out_clean)
    inst_rows = []
    for name, r in prim["results"].items():
        for k, v in r.rhos.items():
            inst_rows.append({"test": name, "instance": k, "rho": v})
        for k, why in r.invalid.items():
            inst_rows.append({"test": name, "instance": k, "rho": np.nan, "invalid_reason": why})
    pd.DataFrame(inst_rows).to_csv(run_dir / "primary_instances.csv", index=False)

    stop = pre["stop"]
    validity = {(v["S"], v["granularity"]): v["valid"] for v in pre["label_validity"]}
    P3.secondary1(df, min_valid, validity, stop=stop).to_csv(run_dir / "secondary1_spearman.csv", index=False)
    auc = P3.secondary3_auc(df, pre["cells"], min_valid)
    logit, coefs = P3.secondary4_logistic(df, pre["cells"])
    s7 = P3.secondary7_sensitivity(df, pre["cells"], min_valid)
    for t in (auc, logit, s7):
        if len(t):
            t["descriptive_only"] = bool(stop)
    auc.to_csv(run_dir / "secondary3_auc.csv", index=False)
    logit.to_csv(run_dir / "secondary4_logistic.csv", index=False)
    coefs.to_csv(run_dir / "secondary4_logistic_coefs.csv", index=False)
    P3.secondary5_noise(df, prim, min_valid).to_csv(run_dir / "secondary5_rtn_vs_noise.csv", index=False)
    P3.secondary6_dn10(df, min_valid).to_csv(run_dir / "secondary6_dn10.csv", index=False)
    s7.to_csv(run_dir / "secondary7_label_sensitivity.csv", index=False)
    ab = [P3.secondary8_descriptive(P3.load_features(PROJECT_ROOT / p / "features.parquet"), s)
          for s, p in cfg["descriptive"].items()]  # fmt: skip
    pd.concat(ab, ignore_index=True).to_csv(run_dir / "secondary8_AB_descriptive.csv", index=False)

    fig_dir = run_figure_dir(run_dir)
    plot_forest({f"set {args.set}": out_clean}, fig_dir / f"phase3_{args.set}_forest",
                title=f"Phase 3, set {args.set}: primary cell C, S=0.7488, per-row, 3 bits")  # fmt: skip
    plot_damage_vs_margin(P3.rows(df, *P3.PRIMARY_CELL), fig_dir / f"phase3_{args.set}_damage_vs_margin",
                          title=f"Set {args.set}, primary cell: damage vs margin_tau (eligible features)")  # fmt: skip
    plot_auc(auc, logit, P3.PRIMARY_CELL, fig_dir / f"phase3_{args.set}_auc",
             title=f"Set {args.set}, primary cell: predicting the composite 'damaged' label")  # fmt: skip
    publish_figures(fig_dir, args.tag)

    print(f"\n=== primary tests (set {args.set}) ===")
    print(
        f"merged H2/H3: {prim['merged_H2_H3']} (rho(norm, D) = {prim['rho_norm_D_primary']:.3f}); Holm family {prim['holm_family']}"
    )
    print(f"primary cell label status: {prim['primary_cell_label_status']}")
    for r in prim["results"].values():
        s = r.summary
        print(f"{r.label}: mean rho {s['mean']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] n={s['n_valid']} "
              f"p={s['p']:.3g} p_holm={r.p_holm:.3g} | sign-flip p={s['p_signflip']:.3g} holm={r.p_holm_signflip:.3g} "
              f"disagrees={r.holm_decision_disagrees} | sqrt(ICC)={r.sqrt_icc:.2f} -> {r.verdict} {r.reasons}")  # fmt: skip
        for d in r.downgrade:
            ds = d["summary"]
            print(f"    downgrade {d['cell']}: mean {ds['mean']:+.3f} n={ds['n_valid']} p={ds['p']:.3g} "
                  f"sqrt(ICC)={d['sqrt_icc_noise']:.2f} [{d['status']}]")  # fmt: skip
    if prim["D_vs_Dplus_note"]:
        print(prim["D_vs_Dplus_note"])
    print("leak-free sensitivity:", {k: round(v["mean"], 3) for k, v in prim["leak_free_sensitivity"].items()})
    print(f"results in {run_dir}")


if __name__ == "__main__":
    main()

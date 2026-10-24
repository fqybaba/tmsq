"""Phase 4 entry point: H4 (head vs tail) on the setting-D sweeps, weak and strong regimes.

``python -m tmsq.experiments.phase4 --config configs/phase4.yaml [--precheck-only]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import phase3 as P3
from .. import phase4 as P4
from ..plots import CATEGORICAL, GRID, INK, INK2, MUTED, save, style
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


def precheck(df: pd.DataFrame, sanity: dict, n_instances: int) -> dict:
    reasons = []
    contingency = None
    if not sanity.get("bits8_pass", False):
        # registered contingency: a failure caused only by rare features (1/p amplification) does not stop H4
        c8 = df[(df.method == "rtn") & (df.ablation == "none") & df.eligible & (df.bits == 8) & (df.p >= 0.05)]
        med = float(c8.dR2.abs().median()) if len(c8) else float("nan")
        contingency = {"median_abs_dR2_8bit_p_ge_0.05": med, "passes": bool(np.isfinite(med) and med < 0.01)}
        if not contingency["passes"]:
            reasons.append("bits8_pass false and the p >= 0.05 contingency also fails")
    for g in P3.GRANULARITIES:
        if not sanity.get("monotonicity", {}).get(g, {}).get("monotone_nondecreasing_as_bits_decrease", False):
            reasons.append(f"non-monotone or missing: {g}")
    problems = []
    if df.instance.nunique() != n_instances:
        problems.append(f"{df.instance.nunique()} instances, expected {n_instances}")
    if set(df.bits.unique()) != P3.EXPECTED_BITS:
        problems.append(f"bits {sorted(df.bits.unique())}")
    if df[df.method == "noise"].noise_seed.nunique() != P3.EXPECTED_NOISE_SEEDS:
        problems.append("noise seeds")
    prim = P4.cell(df, P4.PRIMARY_GRAN, P4.PRIMARY_BITS)
    noise = P4.cell(df, P4.PRIMARY_GRAN, P4.PRIMARY_BITS, method="noise")
    icc = {y: P3.icc1(noise, y)[0] for y in ("rel_damage", "dR2", "dLoss")}
    return {"stop": bool(reasons), "stop_reasons": reasons, "bits8_contingency": contingency,
            "completeness_problems": problems,
            "eligible_per_instance": prim.groupby("instance").size().astype(int).to_dict(),
            "icc_primary": icc, "sqrt_icc_primary": {k: P3.sqrt_icc(v) for k, v in icc.items()}}  # fmt: skip


def plot_reldamage_vs_rank(dfs: dict, stem: Path, stopped: frozenset = frozenset()) -> None:
    style()
    fig, axes = plt.subplots(1, len(dfs), figsize=(5.0 * len(dfs), 3.6), squeeze=False, sharey=True)
    for ax, (name, df) in zip(axes[0], dfs.items(), strict=True):
        c = P4.cell(df, P4.PRIMARY_GRAN, P4.PRIMARY_BITS)
        ax.scatter(c.freq_rank + 1, c.rel_damage, s=8, color=CATEGORICAL[0], alpha=0.4, linewidths=0)
        ax.set_xscale("log")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("frequency rank i + 1 (1 = most frequent)")
        ax.set_title(
            f"{name} regime" + (" (STOP: sanity failed, descriptive only)" if name in stopped else ""), fontsize=9
        )
        ax.grid(True, color=GRID)
    axes[0, 0].set_ylabel("relative damage dR2 / R2_fp (eligible)")
    fig.suptitle("Setting D, per-row 3 bits: relative damage vs frequency rank", fontsize=10, color=INK)
    save(fig, stem)


def plot_head_tail(tab: pd.DataFrame, stem: Path, stopped: frozenset = frozenset()) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), squeeze=False)
    fig.subplots_adjust(wspace=0.38)
    for ax, value in zip(axes[0], ("rel_damage", "dLoss"), strict=True):
        t = tab[(tab.value == value) & (tab.granularity == P4.PRIMARY_GRAN)]
        for j, regime in enumerate(P4.REGIMES):
            tt = t[t.regime == regime].sort_values("bits")
            x = np.arange(len(tt)) + (j - 0.5) * 0.3
            ax.errorbar(x, tt["mean"], yerr=[tt["mean"] - tt.ci_lo, tt.ci_hi - tt["mean"]], fmt="o", ms=5,
                        color=CATEGORICAL[j], label=regime + (" (STOP, descriptive only)" if regime in stopped else ""), capsize=0, elinewidth=1.6)  # fmt: skip
            ax.set_xticks(np.arange(len(tt)))
            ax.set_xticklabels([f"{int(b)} bits" for b in tt.bits])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel(f"head - tail, per-instance mean of {value}")
        ax.legend(fontsize=8)
    fig.suptitle("Setting D, per-row: head (top 10%) minus tail (bottom 50%) of eligible features; 95% t-interval",
                 fontsize=9, color=INK2)  # fmt: skip
    save(fig, stem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 (H4)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir(
        "phase4", f"{'precheck' if args.precheck_only else 'H4'}{'-' + args.tag if args.tag else ''}"
    )
    save_provenance(run_dir, cfg, {"precheck_only": args.precheck_only})
    dfs, pres = {}, {}
    for regime in P4.REGIMES:
        sweep = PROJECT_ROOT / cfg["sweeps"][regime]
        dfs[regime] = P4.load(sweep / "features.parquet")
        sanity = json.load(open(sweep / "sanity.json", encoding="utf-8"))
        pres[regime] = precheck(dfs[regime], sanity, int(cfg["instances_per_regime"]))
        print(f"[{regime}] {sweep.name}: stop={pres[regime]['stop']} {pres[regime]['stop_reasons']} "
              f"problems={pres[regime]['completeness_problems']} ICC={pres[regime]['icc_primary']} "
              f"eligible/instance={sorted(pres[regime]['eligible_per_instance'].values())}")  # fmt: skip
        if pres[regime]["completeness_problems"]:
            raise SystemExit(f"{regime} sweep incomplete: {pres[regime]['completeness_problems']}")
    write_json(run_dir / "precheck.json", _clean(pres))
    if args.precheck_only:
        return

    prim = {r: P4.h4_test(dfs[r], P4.PRIMARY_GRAN, P4.PRIMARY_BITS, P4.MIN_VALID) for r in P4.REGIMES}
    adj = P3.holm({r: (prim[r]["p"] if prim[r]["summarised"] else float("nan")) for r in P4.REGIMES})
    verdicts = {}
    for r in P4.REGIMES:
        icc = pres[r]["icc_primary"]["rel_damage"]
        v = "STOP (sanity failed; descriptive only)" if pres[r]["stop"] else P4.decide(prim[r], adj[r], icc)
        verdicts[r] = {**{k: val for k, val in prim[r].items() if k not in ("rhos",)}, "p_holm": adj[r],
                       "sqrt_icc": P3.sqrt_icc(icc), "verdict": v, "per_instance": prim[r]["rhos"]}  # fmt: skip
    # registered sensitivity: strong regime without the instances whose phi_half < 0.9 (not strong)
    src = json.load(open(PROJECT_ROOT / cfg["sweeps"]["strong"] / "meta.json", encoding="utf-8"))["source_run"]
    inst_tab = pd.read_parquet(Path(src) / "instances.parquet")
    strong_ok = set(inst_tab.loc[inst_tab.phi_half >= P4.PHI_STRONG, "instance"].astype(int))
    sens = P4.h4_test(dfs["strong"][dfs["strong"].instance.isin(strong_ok)], P4.PRIMARY_GRAN, P4.PRIMARY_BITS,
                      P4.MIN_VALID_SENS)  # fmt: skip
    verdicts["strong_sensitivity_phi_ge_0.9"] = {**{k: v for k, v in sens.items() if k != "rhos"},
                                                 "n_instances_kept": len(strong_ok)}  # fmt: skip
    pos = [r for r in P4.REGIMES if verdicts[r]["verdict"].startswith("head loses")]
    neg = [r for r in P4.REGIMES if verdicts[r]["verdict"].startswith("tail loses")]
    hits = int("strong" in pos) + int("weak" in neg)
    verdicts["phase3_mechanism_prediction"] = {
        "predicted": {"strong": "rho > 0", "weak": "rho < 0"},
        "outcome": "confirmed out of sample" if hits == 2 else ("partly" if hits == 1 else "not confirmed"),
        "note": "weak-regime half is overdetermined (rel_damage tilts toward the tail via 1/p); strong half is discriminating",
    }
    write_json(run_dir / "primary.json", _clean(verdicts))

    sec_rows, ht_rows = [], []
    for r in P4.REGIMES:
        for g in P3.GRANULARITIES:
            for b in P4.SECONDARY_BITS:
                noise = P4.cell(dfs[r], g, b, method="noise")
                icc_cache: dict[str, float] = {}
                for x, y, partial in (("p", "rel_damage", None), ("p", "dLoss", None), ("p", "dR2", None),
                                      ("p", "dcond_mse", None), ("p", "dR2_active_term", None),
                                      ("p", "dR2_inactive_term", None), ("p", "rel_damage", "near_thr"),
                                      ("p", "rel_damage", "D"), ("p", "rel_damage", "bias"), ("D", "rel_damage", None),
                                      ("bias", "rel_damage", None), ("near_thr", "rel_damage", None)):  # fmt: skip
                    s = P4.h4_test(dfs[r], g, b, P4.MIN_VALID, x=x, y=y, partial=partial)
                    sec_rows.append({"regime": r, "granularity": g, "bits": b, "x": x, "y": y, "partial_on": partial,
                                     **{k: s[k] for k in ("mean", "ci_lo", "ci_hi", "p", "n_valid", "summarised")},
                                     "sqrt_icc_noise": P3.sqrt_icc(icc_cache.setdefault(y, P3.icc1(noise, y)[0]))})  # fmt: skip
                for value in ("rel_damage", "dLoss"):
                    h = P4.head_tail(dfs[r], g, b, P4.MIN_VALID, value)
                    ht_rows.append({"regime": r, "granularity": g, "bits": b, "value": value,
                                    **{k: h[k] for k in ("mean", "ci_lo", "ci_hi", "p", "n_valid", "head_mean", "tail_mean")}})  # fmt: skip
        nz = P4.cell(dfs[r], P4.PRIMARY_GRAN, P4.PRIMARY_BITS, method="noise")
        per_seed = [P3.instance_rhos(gdf, "p", "rel_damage", min_n=P4.MIN_N)[0] for _, gdf in nz.groupby("noise_seed")]
        noise_k = {k: float(np.mean([ps[k] for ps in per_seed if k in ps])) for k in prim[r]["rhos"]}
        s = P3.summarize(noise_k, P4.MIN_VALID)
        sec_rows.append({"regime": r, "granularity": P4.PRIMARY_GRAN, "bits": P4.PRIMARY_BITS, "x": "p",
                         "y": "rel_damage [noise control, single-draw rho averaged over seeds]", "partial_on": None,
                         **{k: s[k] for k in ("mean", "ci_lo", "ci_hi", "p", "n_valid", "summarised")}})  # fmt: skip
    sec = pd.DataFrame(sec_rows)
    sec["q_bh"] = np.nan
    fam = sec.summarised.astype(bool)
    sec.loc[fam, "q_bh"] = P3.bh(sec.loc[fam, "p"].to_numpy())
    sec.to_csv(run_dir / "secondary_spearman.csv", index=False)
    ht = pd.DataFrame(ht_rows)
    ht.to_csv(run_dir / "head_tail.csv", index=False)
    desc = pd.concat([P4.regime_descriptives(dfs[r]).assign(regime=r) for r in P4.REGIMES], ignore_index=True)
    desc.to_csv(run_dir / "bucket_geometry.csv", index=False)

    fig_dir = run_figure_dir(run_dir)
    stopped = frozenset(r for r in P4.REGIMES if pres[r]["stop"])
    plot_reldamage_vs_rank(dfs, fig_dir / "phase4_reldamage_vs_rank", stopped)
    plot_head_tail(ht, fig_dir / "phase4_head_tail", stopped)
    publish_figures(fig_dir, args.tag)

    print("\n=== H4 primary (per-row, 3 bits; Spearman(p, rel_damage) within instance) ===")
    for r in P4.REGIMES:
        v = verdicts[r]
        print(f"{r}: mean rho {v['mean']:+.3f} [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}] n={v['n_valid']} p={v['p']:.3g} "
              f"p_holm={v['p_holm']:.3g} sqrt(ICC)={v['sqrt_icc']:.2f} -> {v['verdict']}")  # fmt: skip
    s = verdicts["strong_sensitivity_phi_ge_0.9"]
    print(f"strong sensitivity (phi_half >= 0.9, {s['n_instances_kept']} instances): mean rho {s['mean']:+.3f} "
          f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] n={s['n_valid']} p={s['p']:.3g}")  # fmt: skip
    print("Phase 3 mechanism prediction:", verdicts["phase3_mechanism_prediction"]["outcome"])
    print("(weak regime: the head bucket holds about 2 eligible features per instance)")
    print(ht[(ht.granularity == "row") & (ht.bits == 3)].round(4).to_string(index=False))
    print(desc.groupby("regime").mean(numeric_only=True).round(3).T.to_string())
    print(f"results in {run_dir}")


if __name__ == "__main__":
    main()

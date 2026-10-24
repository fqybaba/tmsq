"""Phase 3 POST-HOC exploration (not pre-registered; decided after the K verdicts were seen).

Why are H1 and H2 significant with the sign opposite to CLAUDE.md's direction? Two candidate explanations
are checked descriptively in the primary cell:
- H2: a ceiling effect. dR2 = R2_fp - R2_q cannot exceed R2_fp, and features with larger D may be better
  reconstructed in FP, so they have more to lose. Checked with rho(D, R2_fp), the partial rho(D, dR2 | R2_fp)
  and rho(D, rel_damage) (rel_damage = dR2 / R2_fp).
- H1: the threshold-density mechanism. If near_thr carries the effect, the partial rho(margin_tau, dfalse_act |
  near_thr) should shrink towards 0.

``python -m tmsq.experiments.phase3_posthoc --config configs/phase3.yaml``. Results are exploratory and never
change a verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import phase3 as P3
from .common import load_config, make_run_dir, save_provenance
from .phase3 import resolve_sweep

EXTRA = ["R2_fp", "rel_damage", "false_act_fp", "near_thr", "cond_mse_fp", "leak_energy_fp"]


def analyse(df: pd.DataFrame, set_name: str, min_valid: int) -> list[dict]:
    c = P3.rows(df, *P3.PRIMARY_CELL)
    specs = [
        ("FP-only", "D", "R2_fp", None),
        ("FP-only", "norm", "R2_fp", None),
        ("FP-only", "margin_tau", "false_act_fp", None),
        ("FP-only", "margin_tau", "near_thr", None),
        ("H2 ceiling", "D", "dR2", "R2_fp"),
        ("H2 ceiling", "D", "rel_damage", None),
        ("H2 ceiling", "D_plus", "rel_damage", None),
        ("H3 ceiling", "norm", "rel_damage", None),
        ("H1 mechanism", "margin_tau", "dfalse_act", "near_thr"),
        ("H1 mechanism", "near_thr", "dfalse_act", "margin_tau"),
        ("H1 mechanism", "margin_tau", "dfalse_act", "false_act_fp"),
    ]
    out = []
    for group, x, y, partial in specs:
        rh, inv, _ = P3.instance_rhos(c, x, y, partial=partial)
        s = P3.summarize(rh, min_valid)
        out.append({"set": set_name, "group": group, "x": x, "y": y, "partial_on": partial, "mean": s["mean"],
                    "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "p_uncorrected": s["p"], "n_valid": s["n_valid"]})  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 post-hoc exploration (not pre-registered)")
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir(cfg["experiment"], "posthoc")
    save_provenance(run_dir, cfg, {"status": "POST HOC, exploratory, not pre-registered"})
    recs = []
    for set_name in ("E", "K"):
        spec = cfg["sets"][set_name]
        sweep = resolve_sweep(spec)
        df = P3.load_features(sweep / "features.parquet")
        extra = pd.read_parquet(sweep / "features.parquet", columns=["instance", "feature", "config"] + EXTRA)
        cfgcol = pd.read_parquet(sweep / "features.parquet", columns=["config"])
        df = pd.concat([df, cfgcol, extra[[c for c in EXTRA if c not in df.columns]]], axis=1)
        recs += analyse(df, set_name, int(spec["min_valid"]))
    out = pd.DataFrame(recs)
    out.to_csv(run_dir / "posthoc.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(f"results in {run_dir}")


if __name__ == "__main__":
    main()

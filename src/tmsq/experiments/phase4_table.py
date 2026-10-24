"""Phase 4 descriptive table requested by PROMPTS.md (not a test): head vs tail absolute and relative damage at
every bit width, per regime and (alpha, k) config, mean ± std over seeds, with phi_half and the eligible
fraction of each frequency bucket.

``python -m tmsq.experiments.phase4_table --config configs/phase4.yaml``

Two bucket definitions are reported side by side:
- ``eligible``: head = top 10%, tail = bottom 50% of ELIGIBLE features by p (the H4 definition);
- ``all``: head = top 10% (20), tail = bottom 50% (100) of ALL 200 features by p; ``elig_frac`` is the eligible
  fraction of that bucket (the selection effect of design note 8c), and damage means are over its eligible members.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import phase4 as P4
from .common import PROJECT_ROOT, load_config, make_run_dir, save_provenance

VALUES = ("rel_damage", "dLoss", "dR2")


def bucket_all(d: pd.DataFrame) -> pd.Series:
    """Per instance, over ALL features: head = top 10% by p, tail = bottom 50%."""
    out = pd.Series("middle", index=d.index)
    for _, g in d.groupby("instance"):
        r = g.p.rank(ascending=False, method="first")
        n = len(g)
        out.loc[g.index[r <= int(np.floor(P4.HEAD_FRAC * n))]] = "head"
        out.loc[g.index[r > n - int(np.floor(P4.TAIL_FRAC * n))]] = "tail"
    return out


def per_instance(df: pd.DataFrame, gran: str, bits: int) -> pd.DataFrame:
    """One row per (instance, definition, bucket): eligible count/fraction and mean damage of eligible members."""
    rows = []
    base = df[(df.granularity == gran) & (df.bits == bits) & (df.method == "rtn") & (df.ablation == "none")]
    el = base[base.eligible]
    defs = {"eligible": el.assign(bucket=P4.buckets(el)), "all": base.assign(bucket=bucket_all(base))}
    for name, d in defs.items():
        for (inst, b), g in d[d.bucket != "middle"].groupby(["instance", "bucket"]):
            ge = g[g.eligible]
            rows.append({"instance": int(inst), "definition": name, "bucket": b, "n_bucket": len(g),
                         "n_eligible": len(ge), "elig_frac": len(ge) / len(g),
                         **{v: float(ge[v].mean()) if len(ge) else np.nan for v in VALUES}})  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 descriptive head/tail table")
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir("phase4", "table")
    save_provenance(run_dir, cfg, {})
    out = []
    for regime in P4.REGIMES:
        sweep = PROJECT_ROOT / cfg["sweeps"][regime]
        df = P4.load(sweep / "features.parquet")
        src = Path(json.load(open(sweep / "meta.json", encoding="utf-8"))["source_run"])
        inst = pd.read_parquet(src / "instances.parquet")[["instance", "phi_half", "alpha", "k"]]
        for gran in ("row", "tensor", "column"):
            for bits in sorted(df.bits.unique(), reverse=True):
                t = per_instance(df, gran, int(bits)).merge(inst, on="instance")
                t = t.assign(regime=regime, granularity=gran, bits=int(bits))
                out.append(t)
    full = pd.concat(out, ignore_index=True)
    full.to_csv(run_dir / "head_tail_per_instance.csv", index=False)
    keys = ["regime", "granularity", "bits", "definition", "alpha", "k", "bucket"]
    agg = full.groupby(keys).agg(
        phi_half=("phi_half", "mean"), n_seeds=("instance", "nunique"), n_bucket=("n_bucket", "mean"),
        n_eligible=("n_eligible", "mean"), elig_frac=("elig_frac", "mean"),
        **{f"{v}_mean": (v, "mean") for v in VALUES}, **{f"{v}_std": (v, "std") for v in VALUES},
    ).reset_index()  # fmt: skip
    agg.to_csv(run_dir / "head_tail_by_config.csv", index=False)
    show = agg[(agg.granularity == "row") & (agg.definition == "eligible")]
    cols = ["regime", "alpha", "k", "phi_half", "bits", "bucket", "n_eligible", "rel_damage_mean", "rel_damage_std",
            "dLoss_mean", "dLoss_std"]  # fmt: skip
    pd.set_option("display.width", 220)
    print(show[cols].round(4).to_string(index=False))
    print(f"results in {run_dir}")


if __name__ == "__main__":
    main()

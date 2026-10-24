"""Phase 3b: confirmatory test of the post-hoc "antipodal class is more fragile" finding (notes/LOG.md,
pre-registration commit 326fc7c). Primary test on the fresh set K2; the same statistic on E and K is shown
next to it (post hoc there).

``python -m tmsq.experiments.phase3b --sweep-glob "results/quant_sweep/*-C_confirm2"``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import phase3 as P3
from .common import PROJECT_ROOT, make_run_dir, save_provenance, write_json

S_CELL = 0.7488
D_ANTIPODAL = 0.48
MIN_ANTIPODAL, MIN_OTHER = 2, 10
MIN_VALID_K2 = 16
EFFECT_MIN = 0.20
EXTRA = ["R2_fp", "bias"]
POSTHOC_SETS = {"E": "results/quant_sweep/20261024-093058-C", "K": "results/quant_sweep/20261024-140931-C_confirm"}


def load(sweep: Path) -> pd.DataFrame:
    df = P3.load_features(sweep / "features.parquet")
    ex = pd.read_parquet(sweep / "features.parquet", columns=EXTRA)
    return pd.concat([df, ex], axis=1)


def class_deltas(c: pd.DataFrame, value: str) -> tuple[dict[int, float], dict[int, str]]:
    """Per instance: mean(value | antipodal) - mean(value | other), with the registered validity rule."""
    out, inv = {}, {}
    for inst, d in c.groupby("instance"):
        d = d.dropna(subset=[value])
        a = d[d.D >= D_ANTIPODAL]
        o = d[d.D < D_ANTIPODAL]
        if len(a) < MIN_ANTIPODAL or len(o) < MIN_OTHER:
            inv[int(inst)] = f"{len(a)} antipodal / {len(o)} other"
            continue
        out[int(inst)] = float(a[value].mean() - o[value].mean())
    return out, inv


def cell(df: pd.DataFrame, gran: str, bits: int, method: str = "rtn") -> pd.DataFrame:
    c = P3.rows(df, S_CELL, gran, bits, method=method)
    return c.assign(damaged=P3.damaged(c))


def verdict(s: dict) -> str:
    if not s["summarised"]:
        return "Not testable"
    if s["p"] < 0.05 and s["mean"] < 0:
        return "Not supported (significant negative difference)"
    if s["p"] >= 0.05:
        return "Not supported"
    return "Supported" if s["mean"] >= EFFECT_MIN else "Mixed"


def analyse(df: pd.DataFrame, min_valid: int) -> dict:
    res: dict = {}
    prim = cell(df, "row", 3)
    d, inv = class_deltas(prim, "damaged")
    res["primary"] = {**P3.summarize(d, min_valid), "invalid": inv, "per_instance": d}
    sec = {}
    for value in ("dR2", "dleak_energy", "dfalse_act"):
        dd, _ = class_deltas(prim, value)
        sec[f"row3b_{value}"] = P3.summarize(dd, min_valid)
    for gran, bits in (("column", 3), ("tensor", 3), ("row", 4)):
        dd, _ = class_deltas(cell(df, gran, bits), "damaged")
        sec[f"{gran}{bits}b_damaged"] = P3.summarize(dd, min_valid)
    nz = cell(df, "row", 3, method="noise")
    per_seed = [class_deltas(g, "damaged")[0] for _, g in nz.groupby("noise_seed")]
    noise_k = {k: float(np.mean([ps[k] for ps in per_seed if k in ps])) for k in d}
    sec["noise_row3b_damaged"] = P3.summarize(noise_k, min_valid)
    res["secondary"] = sec
    prim = prim.assign(cls=np.where(prim.D >= D_ANTIPODAL, "antipodal", "other"))
    res["class_means"] = (
        prim.groupby("cls")[
            ["D", "bias", "near_thr", "false_act_fp", "R2_fp", "damaged", "dR2", "dfalse_act", "dleak_energy"]
        ]
        .mean()
        .round(4)
        .to_dict(orient="index")
    )
    res["class_counts"] = prim.cls.value_counts().to_dict()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3b confirmatory test")
    ap.add_argument("--sweep-glob", default="results/quant_sweep/*-C_confirm2")
    args = ap.parse_args()
    found = [p for p in PROJECT_ROOT.glob(args.sweep_glob) if p.is_dir() and p.name.count("-") == 2]
    if len(found) != 1:
        raise SystemExit(f"expected exactly one untagged K2 sweep, found {[p.name for p in found]}")
    sweep = found[0]
    sanity = json.load(open(sweep / "sanity.json", encoding="utf-8"))
    df = load(sweep)
    run_dir = make_run_dir("phase3b", "K2")
    save_provenance(run_dir, {"pre_registration": "notes/LOG.md, commit 326fc7c"}, {"sweep": str(sweep)})

    problems = []
    if not sanity.get("bits8_pass"):
        problems.append("bits8_pass false")
    for g in P3.GRANULARITIES:
        if not sanity.get("monotonicity", {}).get(g, {}).get("monotone_nondecreasing_as_bits_decrease"):
            problems.append(f"non-monotone or missing: {g}")
    n_inst = int(df[df.S4 == S_CELL].instance.nunique())
    if n_inst != 20 or set(df.bits.unique()) != P3.EXPECTED_BITS or df[df.method == "noise"].noise_seed.nunique() != 5:
        problems.append(f"incomplete sweep: {n_inst} instances, bits {sorted(df.bits.unique())}")
    out = {"stop_problems": problems}
    if problems:
        out["verdict"] = "STOP (sanity or completeness failed); descriptive only"
    k2 = analyse(df, MIN_VALID_K2)
    out["K2"] = k2
    if not problems:
        out["verdict"] = verdict(k2["primary"])
    for name, path in POSTHOC_SETS.items():  # same statistic, post hoc on the sets where it was discovered
        s = analyse(load(PROJECT_ROOT / path), 4 if name == "E" else 8)
        out[f"{name}_posthoc"] = {"primary": {k: v for k, v in s["primary"].items() if k != "per_instance"},
                                  "class_means": s["class_means"]}  # fmt: skip
    write_json(run_dir / "phase3b.json", out)
    p = k2["primary"]
    print(f"[phase3b] sweep {sweep.name} -> {run_dir}")
    print(f"stop problems: {problems}")
    print(f"H5 primary: mean Delta(damaged frac) = {p['mean']:+.3f} [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] "
          f"n={p['n_valid']} p={p['p']:.3g} sign-flip p={p['p_signflip']:.3g} -> {out['verdict']}")  # fmt: skip
    print("invalid instances:", p["invalid"])
    for k, s in k2["secondary"].items():
        print(f"  {k}: {s['mean']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] n={s['n_valid']}")
    print("class counts:", k2["class_counts"])
    print("class means:", json.dumps(k2["class_means"], indent=1))
    for name in POSTHOC_SETS:
        q = out[f"{name}_posthoc"]["primary"]
        print(f"{name} (post hoc): {q['mean']:+.3f} [{q['ci_lo']:+.3f}, {q['ci_hi']:+.3f}] n={q['n_valid']}")


if __name__ == "__main__":
    main()

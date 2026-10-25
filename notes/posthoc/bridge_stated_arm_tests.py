"""POST HOC (not registered): the four registered tests re-run on the stated arm (the paper-described pipeline:
raw weights, per-output-channel scales, raw residual into the SAE), plus every test against Duan's full
four-statistic baseline for both arms.

``python notes/posthoc/bridge_stated_arm_tests.py`` (main venv) writes
``results/bridge_posthoc/stated_arm_tests_20261025.{txt,json}``.

Uses the frozen ``registered()`` (with its S1 beyond-rarity secondary) unchanged; the Duan-4 baseline uses the
registered CV design (``FoldDesign``, spline base) and the linear base Duan used, with no permutations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tmsq import bridge as B
from tmsq.experiments.bridge import merged_table, registered

ROOT = Path(__file__).resolve().parents[2]
FP = ROOT / "results/bridge/20261024-224808-fp"
RUNS = {
    "code": ROOT / "results/bridge/20261024-230636-code-b6",
    "stated": ROOT / "results/bridge/20261024-231452-stated-b8-7-6-5-4",
}
OUT = ROOT / "results/bridge_posthoc/stated_arm_tests_20261025"
DUAN4 = ["rarity", "log_mean", "log_peak", "log_conc"]


def beyond_duan4(d, reg: B.Registered) -> dict[str, float]:
    y = (d["corr"].to_numpy() > B.SURVIVE).astype(int)
    splits = B.cv_folds(y, reg.cv_repeats, reg.cv_folds, reg.seed)
    C = d[DUAN4].to_numpy()
    fd = B.FoldDesign(y, C, splits)
    base_s, base_l = fd.auc(), B.cv_auc(y, C, splits)
    out = {"auc_duan4_spline": base_s, "auc_duan4_linear": base_l}
    cells = B.strata(C[:, [2, 0]], reg.n_strata)  # log peak x rarity cells for the leakage partials
    for p in ("near_thr", "margin", "bias_only"):
        v = d[p].to_numpy()
        out[f"dauc_{p}_spline"] = fd.auc(v) - base_s
        out[f"dauc_{p}_linear"] = B.cv_auc(y, np.column_stack([C, v]), splits) - base_l
        out[f"pspearman_{p}_dfalse_given_duan4"] = B.partial_spearman(v, d["dfalse"].to_numpy(), C, cells)
    return out


def main() -> None:
    reg = B.Registered()
    res = {}
    for arm, run in RUNS.items():
        d = merged_table(FP, run, reg.bits)
        res[arm] = {"beyond_duan4": beyond_duan4(d, reg)}
        if arm == "stated":
            res[arm]["registered_procedure"] = registered(d, reg)
        print(arm, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    lines = ["POST HOC (not registered). INT6. code = Duan's released pipeline (R1); stated = paper-described (R3)."]
    for arm, r in res.items():
        lines.append(f"\n[{arm}] beyond Duan's four statistics:")
        lines += [f"  {k}: {v:+.4f}" for k, v in r["beyond_duan4"].items()]
        if "registered_procedure" in r:
            p = r["registered_procedure"]
            lines.append(f"[{arm}] registered procedure (n = {p['n_features']}, survived {p['survived_frac']:.3f}):")
            for k in ("A_T", "A_M", "B_T", "B_M"):
                stat = f"dAUC {p[k]['dauc']:+.4f}" if k.startswith("A") else f"rho {p[k]['rho']:+.3f}"
                extra = f" | {p[k]['interpretation']}" if "interpretation" in p[k] else ""
                lines.append(f"  {k}: {stat}, Holm p {p[k]['p_holm']:.4f} -> {p[k]['verdict']}{extra}")
            for k, v in p["S1_beyond_peak_and_rarity"].items():
                stat = f"dAUC {v['dauc']:+.4f}" if k.startswith("A") else f"rho {v['rho']:+.3f}"
                lines.append(f"  S1 {k}: {stat}, p {v['p']:.4f}")
    OUT.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

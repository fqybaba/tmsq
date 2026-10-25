"""POST HOC (not registered): what carries the bridge's margin result, and is B_M's sign mechanical?

``python notes/posthoc/bridge_margin_decomposition.py`` (main venv) writes
``results/bridge_posthoc/margin_decomposition_20261024.txt``.

Primary cell only (code arm, INT6, R1). Since mu+ = mean / rate, log mu+ = log_mean + rarity: the margin's
denominator is built from two of Duan's own statistics, so (1) test margin and its parts over Duan's four
predictors, and (2) re-check B_M controlling for log mu+ as well (the leak threshold is 0.05 mu+, and margin is
proportional to 1/mu+). Descriptive: dAUC with the registered CV design, no permutations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tmsq import bridge as B

ROOT = Path(__file__).resolve().parents[2]
FP = ROOT / "results/bridge/20261024-224808-fp/features_fp_code.parquet"
R1 = ROOT / "results/bridge/20261024-230636-code-b6/features_q.parquet"
OUT = ROOT / "results/bridge_posthoc/margin_decomposition_20261024.txt"


def main() -> None:
    reg = B.Registered()
    d = pd.read_parquet(R1).query("bits == 6").merge(pd.read_parquet(FP), on="feature", validate="one_to_one")
    d["log_mu_plus"] = np.log(d["mu_plus"])
    d["neg_b_enc"] = -d["b_enc"]
    y = (d["corr"].to_numpy() > B.SURVIVE).astype(int)
    splits = B.cv_folds(y, reg.cv_repeats, reg.cv_folds, reg.seed)
    lines = [
        "POST HOC (not registered). Primary cell: code arm, INT6, n = %d, survived %.3f." % (len(d), y.mean()),
        "log mu+ vs log_mean + rarity: max |diff| = %.2e" % np.abs(d.log_mu_plus - (d.log_mean + d.rarity)).max(),
        "",
    ]
    bases = {
        "peak (spline)": ["log_peak"],
        "peak + log mu+ (spline)": ["log_peak", "log_mu_plus"],
        "Duan-4 (spline)": ["rarity", "log_mean", "log_peak", "log_conc"],
    }
    preds = ["margin", "bias_only", "neg_b_enc", "log_mu_plus", "interf", "near_thr"]
    for name, cols in bases.items():
        fd = B.FoldDesign(y, d[cols].to_numpy(), splits)
        base = fd.auc()
        lines.append(f"(1) base {name}: CV AUC {base:.4f}")
        for p in preds:
            if p in cols:
                continue
            lines.append(f"      + {p:12s} dAUC {fd.auc(d[p].to_numpy()) - base:+.4f}")
    lin = d[["rarity", "log_mean", "log_peak", "log_conc"]].to_numpy()
    base = B.cv_auc(y, lin, splits)
    lines.append(f"(1') base Duan-4 (linear, Duan's model): CV AUC {base:.4f}")
    for p in preds:
        lines.append(f"      + {p:12s} dAUC {B.cv_auc(y, np.column_stack([lin, d[p].to_numpy()]), splits) - base:+.4f}")
    lines.append("")
    lines.append("(2) cell-consistent partial Spearman with dfalse (cells: 10 deciles for 1 control, 5x5 for 2)")
    for cols in (["log_peak"], ["log_peak", "log_mu_plus"], ["log_peak", "rarity"]):
        C = d[cols].to_numpy()
        cells = B.strata(C[:, 0] if len(cols) == 1 else C, reg.n_strata)
        for p in ["margin", "bias_only", "neg_b_enc", "near_thr"]:
            r = B.partial_spearman(d[p].to_numpy(), d["dfalse"].to_numpy(), C, cells)
            lines.append(f"      given {' + '.join(cols):24s} {p:10s} rho {r:+.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

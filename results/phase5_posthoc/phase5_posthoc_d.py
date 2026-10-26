"""POST HOC (not registered). Setting D strong: (a) FP-only Spearman of each allocation key with frequency p over the
represented pool; (b) a frequency-ranked allocation arm (highest p -> 5 bits) through the frozen rtn_mixed +
evaluate_q; (c) loss_round per-instance r vs U4 by (alpha, k)."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tmsq import phase3 as P3
from tmsq import phase5 as P5
from tmsq.analysis import EvalParams, build_eval_set_vec, eval_seed
from tmsq.experiments.common import load_config, load_weights

pd.set_option("display.width", 220)
RUN = "results/phase5/20261024-155642-D_strong"
SRC = "results/setting_d/20261024-145113-g-1"
m = pd.read_parquet(f"{RUN}/models.parquet")
f = pd.read_parquet(f"{RUN}/features.parquet")
u4f = f[(f.arm == "U4") & f.represented]
print("(a) FP-only Spearman(key, p) over represented columns, mean over 20 instances [sign as ranked by the arm]:")
for arm, (key, desc) in P5.ALLOC_ARMS.items():
    rs = []
    for _, d in u4f.groupby("instance"):
        k = d[key].to_numpy()
        ok = np.isfinite(k)
        rs.append(spearmanr(k[ok] if desc else -k[ok], d.p.to_numpy()[ok]).statistic)
    print(f"   {arm:12s} rho(rank key, p) = {np.mean(rs):+.3f}  (min {np.min(rs):+.3f}, max {np.max(rs):+.3f})")

cfg = load_config("configs/phase5.yaml")
params = EvalParams(**cfg["eval"])
final = load_weights(f"{SRC}/final.pt")
train_seed = int(load_config(f"{SRC}/config.yaml")["seed"])
imp = np.asarray(final["importance"], dtype=np.float64)
rows = []
for k in range(final["W"].shape[0]):
    W, b = final["W"][k].float(), final["b"][k].float()
    p = np.asarray(final["density"][k], dtype=np.float64)
    es = build_eval_set_vec(W.shape[1], p, eval_seed(train_seed, k, params), params)
    ref = P5.fp_reference(W, b, es, params, p, imp)
    bits = P5.allocation_bits(p, ref["geom"]["represented"], True)
    mdl, _ = P5.evaluate_q(P5.rtn_mixed(W, bits)[0], b, es, params, p, imp, ref)
    rows.append(
        {
            "instance": k,
            "arm": "freq_high",
            "r": mdl["r"],
            "composite_frac": mdl["composite_frac"],
            "worst_dR2": mdl["worst_dR2"],
        }
    )
fq = pd.DataFrame(rows)
fq["alpha"] = [float(final["alpha"][k]) for k in fq.instance]
fq["k"] = [float(final["k"][k]) for k in fq.instance]
allr = pd.concat([m[m.arm.isin(["U4", "near_thr", "margin_low", "loss_round"])], fq], ignore_index=True)
allr = allr.groupby(["instance", "arm"]).r.mean().unstack("arm")
print("\n(b) frequency-ranked allocation (post hoc) vs registered arms, paired per instance (20):")
for a, bref in (("freq_high", "U4"), ("freq_high", "near_thr"), ("near_thr", "U4")):
    s = P3.summarize({int(i): float(v) for i, v in (allr[a] - allr[bref]).items()}, 8)
    print(f"   r({a}) - r({bref}) = {s['mean']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] p={s['p']:.3g}")
cfgk = m[m.arm == "U4"].set_index("instance")[["alpha", "k"]]
tab = allr.join(cfgk).groupby(["alpha", "k"]).mean()
print(tab[["U4", "near_thr", "freq_high", "margin_low", "loss_round"]].round(4).to_string())
print("\n(c) loss_round - U4 per instance, by (alpha, k):")
d = (allr["loss_round"] - allr["U4"]).to_frame("d").join(cfgk)
for key, g in d.groupby(["alpha", "k"]):
    print(f"   {key}: " + " ".join(f"{v:+.4f}" for v in g.d))
gr = pd.read_parquet(f"{RUN}/greedy.parquet")
gr = gr[gr.arm == "loss_round"].set_index("instance").join(cfgk)
print(
    "   calibration J_final/J0 by (alpha, k):", (gr.J_final / gr.J0).groupby([gr.alpha, gr.k]).mean().round(3).to_dict()
)

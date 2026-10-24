"""POST HOC (not pre-registered) Phase 4 diagnostics, reproducing results/phase4_posthoc/diagnostics_20261024.txt.

(A) why quantization improves the weak-regime model; (B) strong regime: what drives the sign of rho(p, rel_damage);
(C) weak regime, 2-bit per-row RTN: split of d_norm2 into entries rounded to +-s and entries zeroed;
(D) strong regime: per-instance MEDIAN head-tail difference of rel_damage.

Run from the project root: ``python notes/posthoc/phase4_diagnostics.py``
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tmsq import phase3 as P3
from tmsq import phase4 as P4
from tmsq.experiments.common import load_weights
from tmsq.quantize import rtn

pd.set_option("display.width", 220)
WEAK, STRONG = "results/quant_sweep/20261024-151741-D_weak", "results/quant_sweep/20261024-152211-D_strong"
W = pd.read_parquet(f"{WEAK}/features.parquet")
S = pd.read_parquet(f"{STRONG}/features.parquet")
Mw = pd.read_parquet(f"{WEAK}/models.parquet")
Ms = pd.read_parquet(f"{STRONG}/models.parquet")
print(
    "=== (A) WEAK regime: model-level relative loss change (loss_q/loss_fp - 1), mean over instances [and noise seeds]"
)
for name, M in (("weak", Mw), ("strong", Ms)):
    t = M[M.ablation == "none"].groupby(["granularity", "method", "bits"]).loss_increase_rel.mean().unstack("bits")
    print(name)
    print(t[[8, 6, 5, 4, 3, 2]].round(4).to_string())
print("\n(A) weak, eligible features, per-row: mean d_norm2, dcond_mse, dleak_energy, dR2 by bits and method")
e = W[W.eligible & (W.granularity == "row") & (W.ablation == "none")]
print(e.groupby(["method", "bits"])[["d_norm2", "dcond_mse", "dleak_energy", "dR2"]].mean().round(4).to_string())
e8 = e[(e.bits == 8) & (e.method == "rtn")]
print(f"\n(A) FP self-gain of eligible features: weak norm^2 mean {(e8.norm**2).mean():.3f}, bias mean {e8.bias.mean():.3f}, "
      f"clean_resp(x=1)=norm^2+b mean {e8.clean_resp.mean():.3f}")  # fmt: skip
es = S[S.eligible & (S.granularity == "row") & (S.ablation == "none") & (S.bits == 8) & (S.method == "rtn")]
print(
    f"    strong norm^2 mean {(es.norm**2).mean():.3f}, bias mean {es.bias.mean():.3f}, clean_resp mean {es.clean_resp.mean():.3f}"
)
c2 = e[(e.bits == 2) & (e.method == "rtn")]
rho = np.nanmean([spearmanr(d.d_norm2, -d.dR2).statistic for _, d in c2.groupby("instance")])
print(f"(A) weak 2-bit per-row RTN, eligible: fraction with d_norm2 > 0 = {(c2.d_norm2 > 0).mean():.3f}; "
      f"Spearman(d_norm2, -dR2) per instance mean: {rho:.3f}")  # fmt: skip

print("\n=== (B) STRONG regime, primary cell (per-row, 3 bits, RTN, eligible): per-instance Spearman, mean [95% t]")
c = S[(S.granularity == "row") & (S.bits == 3) & (S.method == "rtn") & (S.ablation == "none") & S.eligible].copy()
c["dleak_over_p"] = c.dleak_energy / c.p
for x, y in (("p", "dR2"), ("p", "rel_damage"), ("p", "R2_fp"), ("p", "dleak_energy"), ("p", "dfalse_act"),
             ("p", "dleak_over_p"), ("near_thr", "dleak_energy"), ("near_thr", "dfalse_act"), ("p", "dcond_mse"),
             ("p", "d_norm2"), ("p", "dW")):  # fmt: skip
    rh, _, _ = P3.instance_rhos(c, x, y, min_n=10)
    s = P3.summarize(rh, 16)
    print(f"  rho({x}, {y}) = {s['mean']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]  n={s['n_valid']}")
c["bucket"] = P4.buckets(c)
print("\n(B) bucket means (strong, primary cell):")
cols = ["p", "R2_fp", "dR2", "rel_damage", "dcond_mse", "dleak_energy", "dfalse_act", "near_thr", "bias"]
print(c[c.bucket != "middle"].groupby("bucket")[cols].mean().round(4).to_string())

d = load_weights("results/setting_d/20261024-144457-g+7/final.pt")
Qall, _ = rtn(d["W"].float(), 2, "row")
el = W[(W.bits == 2) & (W.granularity == "row") & (W.method == "rtn") & W.eligible]
up, zero, tot, ok, nz_counts = [], [], [], True, []
for k, g in el.groupby("instance"):
    Wk, Q = d["W"][k].float(), Qall[k]
    nz = Q != 0
    contrib = Q**2 - Wk**2
    t = []
    for i in g.feature:
        up.append(float(contrib[nz[:, i], i].sum()))
        zero.append(float(contrib[~nz[:, i], i].sum()))
        t.append(float(contrib[:, i].sum()))
        nz_counts.append(int(nz[:, i].sum()))
    tot += t
    ok &= np.allclose(t, g.d_norm2.to_numpy(), atol=1e-5)
print(f"\n(C) weak, 2-bit per-row RTN, {len(tot)} eligible columns: mean d_norm2 {np.mean(tot):+.4f} = "
      f"entries rounded to +-s {np.mean(up):+.4f} + entries zeroed {np.mean(zero):+.4f}; matches stored d_norm2: {ok}")  # fmt: skip
print(f"    fraction of eligible columns with net d_norm2 > 0: {np.mean(np.array(tot) > 0):.3f}; "
      f"mean nonzero codes per eligible column: {np.mean(nz_counts):.2f} of 20")  # fmt: skip
SD = P4.load(f"{STRONG}/features.parquet")
for gran in ("row", "tensor", "column"):
    cc = P4.cell(SD, gran, 3)
    cc = cc.assign(bucket=P4.buckets(cc))
    diffs = {int(i): float(x[x.bucket == "head"].rel_damage.median() - x[x.bucket == "tail"].rel_damage.median())
             for i, x in cc.groupby("instance")}  # fmt: skip
    s = P3.summarize(diffs, 16)
    print(f"(D) strong {gran} 3 bits: per-instance MEDIAN(head) - MEDIAN(tail) of rel_damage = {s['mean']:+.4f} "
          f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] n={s['n_valid']} p={s['p']:.2g}")  # fmt: skip

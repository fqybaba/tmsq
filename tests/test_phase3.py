"""Statistics helpers and decision logic of Phase 3, on synthetic data only."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tmsq import phase3 as P3


def test_partial_spearman_removes_shared_driver():
    rng = np.random.default_rng(0)
    z = rng.normal(size=4000)
    x = z + 0.1 * rng.normal(size=4000)
    y = z + 0.1 * rng.normal(size=4000)
    assert stats.spearmanr(x, y).statistic > 0.9
    assert abs(P3.partial_spearman(x, y, z)) < 0.08


def test_instance_rhos_validity_rule():
    rng = np.random.default_rng(1)
    rows = []
    for inst, n in ((0, 30), (1, 10), (2, 30)):
        x = rng.normal(size=n)
        y = x + rng.normal(size=n) if inst != 2 else np.repeat([0.0, 1.0], n // 2)
        rows += [{"instance": inst, "x": a, "y": b} for a, b in zip(x, y, strict=True)]
    rh, inv, dropped = P3.instance_rhos(pd.DataFrame(rows), "x", "y")
    assert set(rh) == {0} and set(inv) == {1, 2}
    assert "n=10" in inv[1] and "distinct" in inv[2]


def test_sign_flip_exact():
    assert P3.sign_flip_p(np.array([1.0, 1.0, 1.0])) == pytest.approx(2 / 8)
    assert P3.sign_flip_p(np.array([0.5, -0.5])) == pytest.approx(1.0)


def test_summarize_t_interval_matches_scipy():
    v = np.array([0.31, 0.22, 0.4, 0.18, 0.27, 0.35, 0.29, 0.25])
    s = P3.summarize(v, min_valid=8)
    ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
    assert s["summarised"] and s["ci_lo"] == pytest.approx(ci[0]) and s["ci_hi"] == pytest.approx(ci[1])
    assert s["p"] == pytest.approx(stats.ttest_1samp(v, 0).pvalue)
    assert not P3.summarize(v[:7], min_valid=8)["summarised"]


def test_holm_and_bh_known_values():
    adj = P3.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert P3.holm({"a": 0.01, "b": float("nan")})["b"] == 1.0
    q = P3.bh(np.array([0.01, 0.04, 0.03, np.nan]))
    assert q[:3] == pytest.approx([0.03, 0.04, 0.04]) and np.isnan(q[3])


def _noise_frame(feature_sd, noise_sd, n_feat=40, seeds=5, inst=3, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(inst):
        mu = rng.normal(scale=feature_sd, size=n_feat)
        for s in range(seeds):
            y = mu + rng.normal(scale=noise_sd, size=n_feat)
            out += [{"instance": k, "feature": f, "noise_seed": s, "y": v} for f, v in enumerate(y)]
    return pd.DataFrame(out)


def test_icc_extremes():
    assert P3.icc1(_noise_frame(1.0, 0.05), "y")[0] > 0.95
    assert abs(P3.icc1(_noise_frame(0.0, 1.0), "y")[0]) < 0.1
    assert P3.icc1(_noise_frame(1.0, 1.0), "y")[0] == pytest.approx(0.5, abs=0.12)


def test_pooled_auc_and_bootstrap_match_direct():
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(6), 50)
    lab = rng.integers(0, 2, size=300)
    sc = lab * 0.8 + rng.normal(size=300)
    sc[:10] = sc[10]  # some ties
    out = P3.pooled_auc_with_bootstrap(sc, lab, g, boot=500)
    assert out["auc_pooled"] == pytest.approx(P3.auc_rank(sc, lab))
    from sklearn.metrics import roc_auc_score

    assert out["auc_pooled"] == pytest.approx(roc_auc_score(lab, sc))
    assert out["boot_lo"] < out["auc_pooled"] < out["boot_hi"]


def _summ(mean, p, n=10):
    return {"summarised": n >= 8, "n_valid": n, "mean": mean, "p": p}


def test_decide_ordered_rules():
    h = P3.HYPOTHESES[0]  # H1, predicted negative, alternative reading registered
    assert P3.decide(h, _summ(-0.4, 0.001, n=7), 0.001, 0.5, [])[0] == "Not testable"
    v, _ = P3.decide(h, _summ(+0.4, 0.001), 0.003, 0.5, [])
    assert v.startswith("Not supported (opposite sign)") and "threshold-density" in v
    assert P3.decide(h, _summ(-0.05, 0.5), 0.9, 0.05, [])[0].startswith("Not testable (outcome")
    assert P3.decide(h, _summ(-0.05, 0.5), 0.9, 0.5, [])[0] == "Not supported"
    v, why = P3.decide(h, _summ(-0.15, 0.001), 0.003, 0.5, [])
    assert v == "Mixed" and why[0].startswith("small")
    dg = [{"cell": "x", "summary": {"summarised": True, "mean": +0.12, "p": 0.01}}]
    v, why = P3.decide(h, _summ(-0.35, 0.001), 0.003, 0.5, dg)
    assert v == "Mixed" and why[0].startswith("inconsistent")
    weak = [{"cell": "x", "summary": {"summarised": True, "mean": +0.08, "p": 0.01}}]
    assert P3.decide(h, _summ(-0.35, 0.001), 0.003, 0.5, weak)[0] == "Supported"
    untestable = [{"cell": "x", "summary": {"summarised": False, "mean": +0.5, "p": 0.0001}}]
    assert P3.decide(h, _summ(-0.35, 0.001), 0.003, 0.5, untestable)[0] == "Supported"
    # low reliability never blocks a positive result
    assert P3.decide(h, _summ(-0.35, 0.001), 0.003, 0.01, [])[0] == "Supported"


def test_break_bits_table_rules():
    recs = []
    pattern = {  # feature -> damaged at bits (8, 6, 5, 4, 3, 2)
        0: (0, 0, 0, 0, 0, 1),  # break 2
        1: (0, 0, 0, 1, 1, 1),  # break 4
        2: (0, 0, 0, 0, 0, 0),  # survives all -> NaN, rank 1
        3: (0, 0, 1, 0, 1, 1),  # non-monotone: break 5, envelope 3
    }
    for f, dm in pattern.items():
        for b, d in zip(P3.ALL_BITS, dm, strict=True):
            recs.append({"S4": 0.7488, "granularity": "row", "method": "rtn", "ablation": "none", "eligible": True,
                         "instance": 0, "feature": f, "bits": b, "composite_survived": 1.0 - d, "D": 0.3, "norm": 1.0})  # fmt: skip
    t = P3.break_bits_table(pd.DataFrame(recs), 0.7488, "row").set_index("feature")
    assert t.break_bits.tolist()[:2] == [2, 4] and np.isnan(t.break_bits[2]) and t.break_bits[3] == 5
    assert t.break_bits_rank[2] == 1.0
    assert t.break_bits_env[3] == 3 and t.non_monotone[3] and not t.non_monotone[1]


def test_unsummarised_cells_report_no_interval_or_p():
    s = P3.summarize(np.array([0.3, 0.4, 0.2]), min_valid=8)
    assert not s["summarised"] and np.isfinite(s["mean"])
    assert all(np.isnan(s[k]) for k in ("ci_lo", "ci_hi", "p", "p_signflip", "boot_lo", "boot_hi"))


def test_sqrt_icc_ceiling():
    assert P3.sqrt_icc(0.25) == pytest.approx(0.5) and P3.sqrt_icc(-0.1) == 0.0 and np.isnan(P3.sqrt_icc(float("nan")))


def _bb_frame(n_feat_by_inst, seed=0):
    """Break-bits input: damaged at 2 bits always, at 3 bits with prob 0.5 -> break_bits in {2, 3}."""
    rng = np.random.default_rng(seed)
    recs = []
    for inst, n in n_feat_by_inst.items():
        for f in range(n):
            d3 = rng.random() < 0.5
            D = rng.random()
            for b in P3.ALL_BITS:
                dmg = 1.0 if b == 2 or (b == 3 and d3) else 0.0
                recs.append({"S4": 0.7488, "granularity": "row", "method": "rtn", "ablation": "none", "eligible": True,
                             "instance": inst, "feature": f, "bits": b, "composite_survived": 1.0 - dmg, "D": D,
                             "norm": D, "p": 0.25, "dcond_mse": 0.0, "dleak_energy": 0.0})  # fmt: skip
    return pd.DataFrame(recs)


def test_rank_biserial_uses_registered_instance_rule():
    df = _bb_frame({0: 30, 1: 30, 2: 15})
    t = P3.break_bits_table(df, 0.7488, "row")
    assert sorted(t.break_bits_rank.unique()) == [2.0, 3.0]
    # run only the break_bits part of secondary1 through a tiny wrapper: other cells are empty and skipped
    validity = {(S, g): (S == 0.7488 and g == "row") for S in P3.S_GRID for g in P3.GRANULARITIES}
    full = df.assign(dR2=0.0, dfalse_act=0.0, abs_dfalse_act=0.0, margin_tau=0.0, D_plus=0.0, near_thr=0.0,
                     p99_out_fp=0.0, peak_out_fp=0.0, noise_seed=0, S=0.7488)  # fmt: skip
    out = P3.secondary1(full, min_valid=2, validity=validity)
    bb = out[(out.outcome == "break_bits") & (out.predictor == "D")].iloc[0]
    assert bb.statistic.startswith("rank_biserial") and not bb.in_family
    assert bb.n_valid == 2 and bb.n_invalid == 1 and "n=15" in bb.invalid_instances


def test_completeness_flags_missing_parts():
    df = _bb_frame({0: 3}).assign(S4=0.7488, noise_seed=0)
    rep = P3.completeness(df, instances_per_S=10)
    assert any("S=0.7488" in p for p in rep["problems"]) and any("noise seeds" in p for p in rep["problems"])


def test_composite_label_reproduces_registered_definition():
    c = pd.DataFrame({"surv_corr": [0.95, 0.85, 0.95, 0.95, 0.95], "dR2": [0.05, 0.05, 0.2, 0.05, 0.05],
                      "dfalse_act": [0.0, 0.0, 0.0, np.float32(0.05), np.nan]})  # fmt: skip
    lab = P3.composite_label(c)
    assert lab.tolist()[:4] == [0.0, 1.0, 1.0, 0.0] and np.isnan(lab.iloc[4])

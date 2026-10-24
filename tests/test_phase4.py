"""Phase 4 helpers on synthetic data."""

import numpy as np
import pandas as pd

from tmsq import phase4 as P4


def _frame(n_inst=3, n_feat=40, slope=1.0, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for inst in range(n_inst):
        p = np.sort(rng.random(n_feat))[::-1] * 0.9 + 0.01
        for f in range(n_feat):
            recs.append({"instance": inst, "feature": f, "p": p[f], "freq_rank": f, "bits": 3, "granularity": "row",
                         "method": "rtn", "ablation": "none", "eligible": f < n_feat - 5,
                         "rel_damage": slope * p[f] + 0.05 * rng.normal(), "dLoss": p[f]})  # fmt: skip
    return pd.DataFrame(recs)


def test_buckets_head_top10_tail_bottom50_of_eligible():
    df = _frame()
    c = P4.cell(df, "row", 3)
    b = P4.buckets(c)
    per = c.assign(b=b).groupby("instance").b.value_counts().unstack()
    assert (per["head"] == 3).all() and (per["tail"] == 17).all()  # 35 eligible -> floor(3.5)=3, floor(17.5)=17
    head_p = c[b == "head"].groupby("instance").p.min()
    tail_p = c[b == "tail"].groupby("instance").p.max()
    assert (head_p > tail_p).all()


def test_h4_test_and_head_tail_detect_a_positive_frequency_effect():
    df = _frame(n_inst=18, slope=1.0)
    s = P4.h4_test(df, "row", 3, min_valid=16)
    assert s["summarised"] and s["mean"] > 0.8 and s["p"] < 1e-6
    h = P4.head_tail(df, "row", 3, 16, "rel_damage")
    assert h["mean"] > 0.3 and h["head_mean"] > h["tail_mean"]


def test_decide_categories():
    ok = {"summarised": True, "mean": 0.35}
    assert P4.decide(ok, 0.001, 0.5).startswith("head loses")
    assert P4.decide({**ok, "mean": -0.3}, 0.001, 0.5).startswith("tail loses")
    assert "small" in P4.decide({**ok, "mean": 0.1}, 0.001, 0.5)
    assert P4.decide(ok, 0.3, 0.5) == "No detectable frequency effect"
    assert P4.decide(ok, 0.3, 0.05).startswith("Not testable (outcome")
    assert P4.decide({"summarised": False, "mean": 0.5}, 0.001, 0.5).startswith("Not testable (too few")

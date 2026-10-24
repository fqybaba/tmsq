"""End-to-end smoke test of the Phase 5 analysis entry point on synthetic runner output (all arms, 20 random draws,
C with 4 S values x 10 instances, D with 20 instances), so the frozen analysis never meets its first real input on
the confirmatory run."""

import json
import sys

import numpy as np
import pandas as pd
import yaml

from tmsq.experiments import common
from tmsq.experiments import phase5_analysis as A
from tmsq.phase5 import ALLOC_ARMS, N_RANDOM


def _run(tmp, name, S_values, n_inst_per, n_feat=12, seed=0):
    rng = np.random.default_rng(seed)
    arms = [("U4", -1), ("U3", -1), ("U5", -1), *[(a, -1) for a in ALLOC_ARMS], *[("random", d) for d in range(N_RANDOM)],
            ("leak_round", -1), ("loss_round", -1)]  # fmt: skip
    models, feats, greedy = [], [], []
    inst = 0
    for S in S_values:
        for _ in range(n_inst_per):
            base = rng.uniform(0.05, 0.2)
            for arm, draw in arms:
                eff = {"near_thr": -0.02, "margin_low": 0.01}.get(arm, 0.0)
                r = base + eff + rng.normal(0, 0.005)
                models.append({"instance": inst, "arm": arm, "draw": draw, "S": S, "r": r, "loss_fp": 1.0,
                               "loss_q": 1 + r, "n_eligible": n_feat - 2, "composite_frac": rng.uniform(0.3, 0.9),
                               "worst_dR2": rng.uniform(0.1, 1), "mean_dR2": rng.uniform(0, 0.2),
                               "frob_err": 0.1})  # fmt: skip
                bits = (
                    np.array([5, 3] * (n_feat // 2))
                    if arm not in ("U4", "U3", "U5", "leak_round", "loss_round")
                    else np.full(n_feat, 4)
                )
                p = np.full(n_feat, 1 - S) if S == S else np.minimum(1, 0.5 / (np.arange(n_feat) + 1))
                feats.append(pd.DataFrame({"instance": inst, "arm": arm, "draw": draw, "feature": np.arange(n_feat),
                                           "eligible": np.arange(n_feat) < n_feat - 2, "dR2": rng.uniform(0, 0.2, n_feat),
                                           "dcond_mse": rng.normal(0, 0.01, n_feat),
                                           "dleak_energy": np.where(p < 1, rng.normal(0, 0.001, n_feat), np.nan),
                                           "dfalse_act": rng.normal(0, 0.01, n_feat), "bits_col": bits, "p": p,
                                           "represented": True, "margin": rng.normal(size=n_feat),
                                           "norm": rng.uniform(0.6, 1.2, n_feat), "near_thr_cal": rng.uniform(size=n_feat),
                                           "S": S}))  # fmt: skip
            for arm in ("leak_round", "loss_round"):
                greedy.append({"instance": inst, "arm": arm, "objective": arm[:4], "J0": 1.0, "J_final": 0.9,
                               "J_tracked": 0.9, "n_flips": 50, "n_columns": n_feat, "seconds": 1.0, "S": S})  # fmt: skip
            inst += 1
    d = tmp / name
    d.mkdir()
    pd.DataFrame(models).to_parquet(d / "models.parquet")
    pd.concat(feats).to_parquet(d / "features.parquet")
    pd.DataFrame(greedy).to_parquet(d / "greedy.parquet")
    (d / "sanity.json").write_text(json.dumps({"stop": False}))
    return d


def test_phase5_analysis_end_to_end(tmp_path, monkeypatch, capsys):
    c = _run(tmp_path, "C", [0.7488, 0.8415, 0.9, 0.9602], 10)
    dd = _run(tmp_path, "D", [np.nan], 20, seed=1)
    cfg_path = tmp_path / "phase5.yaml"
    cfg_path.write_text(yaml.safe_dump({"runs": {"C_P5": str(c), "D_strong": str(dd)}}))
    monkeypatch.setattr(common, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(common, "FIGURES_ROOT", tmp_path / "figures")
    monkeypatch.setattr(sys, "argv", ["phase5_analysis", "--config", str(cfg_path), "--tag", "e2e"])
    A.main()
    run = next(p for p in (tmp_path / "results" / "phase5").iterdir() if p.name.endswith("analysis-e2e"))
    prim = json.loads((run / "primary.json").read_text())
    f1, f2 = prim["family1_vs_U4"], prim["family2_vs_random"]
    assert set(f1) == set(A.PRIMARY_ARMS) and set(f2) == set(A.SECOND_ARMS)
    assert f1["near_thr"]["category"] == "better than uniform" and f1["margin_low"]["category"] == "worse than uniform"
    assert f2["near_thr"]["category"] == "better than random" and f1["leak_round"]["n_valid"] == 10
    sec = pd.read_csv(run / "secondary.csv")
    assert set(sec.setting) == {"C_P5", "D_strong"} and sec.q_bh.notna().any()
    mech = pd.read_csv(run / "mechanism.csv")
    assert mech.check.str.contains("5-bit half").any() and mech.check.str.contains("inactive loss").any()
    assert (run / "figures" / "phase5_arms_vs_u4.png").exists()
    assert "family 2" in capsys.readouterr().out

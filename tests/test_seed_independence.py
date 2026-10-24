"""Seed bookkeeping for the Phase 3 exploration (E, training seed 0) and confirmation (K, training seed 1) sets.

K must share no random stream with E or with itself: model init, training data, eval sets and the
noise-control directions all come from torch generators seeded by integers, so equal integers mean equal
random numbers.
"""

from tmsq.analysis import EvalParams, eval_seed
from tmsq.model import instance_seed
from tmsq.quantize import QuantConfig, build_configs, noise_generator_seed

BITS = [8, 6, 5, 4, 3, 2]
CONFIGS = [
    c
    for c in build_configs(BITS, ["tensor", "column", "row"], 5, ["quant_b", "enc_only", "dec_only"], ["column", "row"])
    if c.method == "noise"
]
E = {"train_seed": 0, "K": 80}  # results/rep_dims/20261024-063236 (sweep uses instances 0..79 subset)
K = {"train_seed": 1, "K": 40}  # results/rep_dims_confirm/20261024-130737


def _init_seeds(s):
    return {instance_seed(s["train_seed"], k) for k in range(s["K"])}


def _train_sampler_seeds(s, n_chunks=4):
    seed = s["train_seed"] + 1  # build_model_and_sampler: FeatureSampler(seed=cfg["seed"] + 1)
    return {seed * 1_000_003 + 17 * c + 3 for c in range(n_chunks)}


def _eval_seeds(s):
    p = EvalParams()
    return {eval_seed(s["train_seed"], k, p) * 1_000_003 + 3 for k in range(s["K"])}  # n_chunks=1 generator


def _noise_seeds(s):
    return {noise_generator_seed(c, k, s["train_seed"]) for c in CONFIGS for k in range(s["K"])}


def test_train_seed_zero_reproduces_phase2_noise_seeds():
    c = QuantConfig(4, "row", "noise", noise_seed=2)
    assert noise_generator_seed(c, 7) == noise_generator_seed(c, 7, 0) == 2 * 1_000_003 + 1009 * 7 + 28 + 2


def test_confirmation_noise_streams_are_new():
    k_noise = _noise_seeds(K)
    assert len(k_noise) == len(CONFIGS) * K["K"]  # no repeats inside K
    other = _noise_seeds(E) | _init_seeds(E) | _init_seeds(K) | _train_sampler_seeds(E) | _train_sampler_seeds(K)
    other |= _eval_seeds(E) | _eval_seeds(K)
    assert not (k_noise & other)


def test_confirmation_init_train_eval_streams_are_new():
    k_all = _init_seeds(K) | _train_sampler_seeds(K) | _eval_seeds(K)
    e_all = _init_seeds(E) | _train_sampler_seeds(E) | _eval_seeds(E)
    assert not (k_all & e_all)


K2 = {"train_seed": 7, "K": 20}  # Phase 3b set, results/rep_dims_confirm2


def test_phase3b_set_shares_no_stream():
    k2 = _init_seeds(K2) | _train_sampler_seeds(K2) | _eval_seeds(K2) | _noise_seeds(K2)
    others = set()
    for s in (E, K):
        others |= _init_seeds(s) | _train_sampler_seeds(s) | _eval_seeds(s) | _noise_seeds(s)
    assert not (k2 & others)


def test_known_preexisting_collision_is_exactly_one():
    """Documented in notes/LOG.md (Phase 3 pre-registration v2): K's training-sampler chunk-2 seed 2,000,043
    equals E's noise seed for (instance 0, 5 bits, per-row, noise seed 2). Both runs already exist; one is a
    torch.rand stream of training data, the other one torch.randn noise direction, so the dependence is
    immaterial. Any further collision is a bug."""
    assert _train_sampler_seeds(K) & _noise_seeds(E) == {2_000_043}
    c = QuantConfig(5, "row", "noise", noise_seed=2)
    assert noise_generator_seed(c, 0, 0) == 2_000_043

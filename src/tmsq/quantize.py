"""Post-training quantization (CLAUDE.md, "Quantization (PTQ) definitions").

Symmetric uniform round-to-nearest (RTN): ``qmax = 2**(bits-1) - 1`` and
``Q(W) = s * clamp(round(W / s), -qmax, qmax)``. The codebook has ``2*qmax + 1 = 2**bits - 1`` values
(2-bit is ternary). Scale shapes for ``W [K, m, n]`` (this contract is reused in Phase 5):

- per-tensor: ``s [K, 1, 1]`` = ``max|W| / qmax`` per instance;
- per-column (one scale per feature): ``s [K, 1, n]``;
- per-row (one scale per hidden dimension): ``s [K, m, 1]``.

Scales are floored at ``SCALE_EPS`` so an all-zero column / row quantizes to exactly 0 without NaN.
``round`` is ``torch.round`` (round-half-to-even); ties have measure zero for trained float weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GRANULARITIES = ("tensor", "column", "row")
METHODS = ("rtn", "noise")
ABLATIONS = ("none", "quant_b", "enc_only", "dec_only")
SCALE_EPS = 1e-8


def qmax_for(bits: int) -> int:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return 2 ** (bits - 1) - 1


def rtn_scales(W: torch.Tensor, bits: int, granularity: str) -> torch.Tensor:
    """Per-instance scales broadcastable to ``W [K, m, n]`` (see module docstring for shapes)."""
    q = qmax_for(bits)
    a = W.abs()
    if granularity == "tensor":
        s = a.amax(dim=(1, 2), keepdim=True)
    elif granularity == "column":
        s = a.amax(dim=1, keepdim=True)
    elif granularity == "row":
        s = a.amax(dim=2, keepdim=True)
    else:
        raise ValueError(f"unknown granularity {granularity!r}")
    return (s / q).clamp_min(SCALE_EPS)


def rtn(W: torch.Tensor, bits: int, granularity: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``W [K, m, n]``; returns ``(Q(W), scales)``."""
    s = rtn_scales(W, bits, granularity)
    q = qmax_for(bits)
    return s * torch.clamp(torch.round(W / s), -q, q), s


def quantize_bias(b: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric RTN of ``b [K, n]`` with one per-tensor scale per instance (``max|b| / qmax``)."""
    q = qmax_for(bits)
    s = (b.abs().amax(dim=1, keepdim=True) / q).clamp_min(SCALE_EPS)
    return s * torch.clamp(torch.round(b / s), -q, q)


def noise_control(W: torch.Tensor, Q: torch.Tensor, seed: int) -> torch.Tensor:
    """Gaussian-direction perturbation with, per column, exactly the realized rounding-error norm.

    ``delta_i = ||Q_i - W_i|| * g_i / ||g_i||`` with ``g_i ~ N(0, I_m)``; returns ``W + delta``. This also
    matches the total Frobenius error ``||Q - W||_F``.
    """
    gen = torch.Generator().manual_seed(int(seed))
    g = torch.randn(W.shape, generator=gen, dtype=W.dtype)
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
    err = (Q - W).norm(dim=1, keepdim=True)
    return W + err * g


@dataclass(frozen=True)
class QuantConfig:
    """One point of the sweep: bit width, scale granularity, RTN or its noise control, ablation."""

    bits: int
    granularity: str
    method: str = "rtn"  # "rtn" | "noise"
    noise_seed: int = 0
    ablation: str = "none"  # "none" | "quant_b" | "enc_only" | "dec_only"

    def __post_init__(self) -> None:
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"granularity {self.granularity!r}")
        if self.method not in METHODS:
            raise ValueError(f"method {self.method!r}")
        if self.ablation not in ABLATIONS:
            raise ValueError(f"ablation {self.ablation!r}")
        if self.method == "noise" and self.ablation != "none":
            raise ValueError("noise controls are only run for the un-ablated tied quantization")

    @property
    def label(self) -> str:
        s = f"{self.bits}b_{self.granularity}_{self.method}"
        if self.method == "noise":
            s += f"{self.noise_seed}"
        if self.ablation != "none":
            s += f"_{self.ablation}"
        return s


@dataclass
class QuantResult:
    """Weights actually used in the forward pass plus the perturbed matrix ``Wq`` for geometry."""

    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b: torch.Tensor
    Wq: torch.Tensor
    scales: torch.Tensor


NOISE_TRAIN_SEED_SALT = 2_147_483_629  # large prime-sized offset per training seed


def noise_generator_seed(cfg: QuantConfig, instance: int, train_seed: int = 0) -> int:
    """Noise directions are independent across (training seed, noise seed, instance, bit width, granularity).

    ``train_seed = 0`` reproduces the Phase 2 exploration sweep exactly; any other training seed (e.g. the
    Phase 3 confirmation set, seed 1) is shifted by ``NOISE_TRAIN_SEED_SALT`` so its directions differ from
    those of instances with the same index in other model sets.
    """
    base = 1_000_003 * cfg.noise_seed + 1009 * int(instance) + 7 * cfg.bits + GRANULARITIES.index(cfg.granularity)
    return base + NOISE_TRAIN_SEED_SALT * int(train_seed)


def apply_quant(
    cfg: QuantConfig, W: torch.Tensor, b: torch.Tensor, instance: int = 0, train_seed: int = 0
) -> QuantResult:
    """Apply a sweep point to ``W [K, m, n]``, ``b [K, n]``.

    Tied by default (the same ``Wq`` as encoder and decoder, ``b`` in FP32). Ablations: ``quant_b`` also
    quantizes ``b``; ``enc_only`` / ``dec_only`` use ``Wq`` on one side and the FP ``W`` on the other.
    ``instance`` and ``train_seed`` only salt the noise-control generator (see :func:`noise_generator_seed`).
    """
    Q, s = rtn(W, cfg.bits, cfg.granularity)
    Wq = noise_control(W, Q, noise_generator_seed(cfg, instance, train_seed)) if cfg.method == "noise" else Q
    if cfg.ablation == "enc_only":
        W_enc, W_dec = Wq, W
    elif cfg.ablation == "dec_only":
        W_enc, W_dec = W, Wq
    else:
        W_enc = W_dec = Wq
    b_used = quantize_bias(b, cfg.bits) if cfg.ablation == "quant_b" else b
    return QuantResult(W_enc, W_dec, b_used, Wq, s)


def build_configs(
    bits: list[int],
    granularities: list[str],
    noise_seeds: int,
    ablations: list[str],
    ablation_granularities: list[str],
) -> list[QuantConfig]:
    """The full sweep grid: RTN + ``noise_seeds`` controls per granularity, plus RTN ablations."""
    out: list[QuantConfig] = []
    for bw in bits:
        for g in granularities:
            out.append(QuantConfig(bw, g, "rtn"))
            for ns in range(noise_seeds):
                out.append(QuantConfig(bw, g, "noise", noise_seed=ns))
        for g in ablation_granularities:
            for ab in ablations:
                out.append(QuantConfig(bw, g, "rtn", ablation=ab))
    return out

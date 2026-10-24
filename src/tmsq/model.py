"""Batched tied-weight toy model (CLAUDE.md, "Model and data").

``h = W x`` with ``W in R^{m x n}``; ``x_hat = ReLU(W^T h + b)``. ``K`` independent instances are stored along
a leading instance dimension: ``W [K, m, n]``, ``b [K, n]``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def instance_seed(base_seed: int, k: int) -> int:
    """Deterministic per-instance seed derived from the base seed and the instance index."""
    return (base_seed * 100_003 + 7_919 * k + 1) % (2**31 - 1)


class ToyModel(nn.Module):
    """``K`` tied-weight ReLU-output autoencoders trained in parallel.

    Init: per-instance Xavier normal, ``std = sqrt(2 / (n + m))`` for each ``[m, n]`` matrix (the official
    notebook applies ``xavier_normal_`` to the whole ``[K, n, m]`` tensor, which makes the std depend on K;
    we do not reproduce that quirk). ``b = 0``.
    """

    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        seed: int = 0,
        init_std: float | None = None,
    ) -> None:
        super().__init__()
        self.K, self.n, self.m = n_instances, n_features, n_hidden
        self.init_std = init_std
        self.W = nn.Parameter(torch.empty(n_instances, n_hidden, n_features))
        self.b = nn.Parameter(torch.zeros(n_instances, n_features))
        self.reset_parameters(seed)

    def xavier_std(self) -> float:
        """Xavier-normal std for one ``[m, n]`` matrix: ``sqrt(2 / (n + m))``."""
        return math.sqrt(2.0 / (self.n + self.m))

    def reset_parameters(self, seed: int) -> None:
        std = self.init_std if self.init_std is not None else self.xavier_std()
        with torch.no_grad():
            for k in range(self.K):
                gen = torch.Generator().manual_seed(instance_seed(seed, k))
                self.W[k] = torch.randn(self.m, self.n, generator=gen) * std
            self.b.zero_()

    @staticmethod
    def encode(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``h = W x``; ``x [K, B, n] -> h [K, B, m]``."""
        return torch.einsum("kmn,kbn->kbm", W, x)

    @staticmethod
    def decode(W: torch.Tensor, b: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """``x_hat = ReLU(W^T h + b)``; ``h [K, B, m] -> [K, B, n]``."""
        return F.relu(torch.einsum("kmn,kbm->kbn", W, h) + b[:, None, :])

    @staticmethod
    def run(W_enc: torch.Tensor, W_dec: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with explicit (possibly different) encoder / decoder weights, for Phase 2 ablations."""
        return ToyModel.decode(W_dec, b, ToyModel.encode(W_enc, x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x_hat = ReLU(W^T (W x) + b)``; ``x [K, B, n] -> [K, B, n]``."""
        return self.run(self.W, self.W, self.b, x)

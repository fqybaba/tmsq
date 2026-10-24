"""Sparse feature generator (CLAUDE.md, "Model and data").

Each feature ``x_i`` is 0 with probability ``S_i`` and otherwise ``Uniform[0, 1)``.
``feature_prob = 1 - S`` may differ per instance and per feature (shape ``[K, n]``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.optimize import brentq


def power_law_frequencies(n: int, alpha: float, k: float) -> tuple[np.ndarray, float]:
    """Setting D: ``p_i = min(1, c (i+1)^-alpha)``, i = 0..n-1, with ``c`` solved by root-finding so that
    ``sum_i p_i = k`` (the clip makes the closed form wrong when it binds). Returns ``(p, c)``."""
    if not 0 < k <= n:
        raise ValueError("need 0 < k <= n")
    w = (np.arange(n) + 1.0) ** (-alpha)

    def gap(c: float) -> float:
        return float(np.minimum(1.0, c * w).sum() - k)

    hi = 1.0
    while gap(hi) < 0:
        hi *= 2.0
    c = brentq(gap, 0.0, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    p = np.minimum(1.0, c * w)
    assert abs(p.sum() - k) < 1e-9, p.sum()
    return p, float(c)


class FeatureSampler:
    """Batched sampler returning tensors of shape ``[K, batch, n]``.

    Implementation notes: one float32 uniform ``u`` per entry, active iff ``u < p``, value ``u / p``
    (distributionally exact; the active value lives on a lattice of ``p * 2^24`` levels, ~1.7e4 at
    ``p = 1e-3``, harmless for MSE). Instances are drawn in ``n_chunks`` thread chunks with their own
    generators, so a run is reproducible for a fixed ``(seed, K, n_chunks)``; instance ``k``'s data stream
    is NOT invariant to ``K`` (its init is, see ``model.instance_seed``).

    Args:
        n_features: number of features ``n``.
        feature_prob: ``p_i = 1 - S_i``; scalar, ``[n]``, ``[K, 1]`` or ``[K, n]`` (broadcast to ``[K, n]``).
        n_instances: number of model instances ``K``.
        seed: seed of the internal ``torch.Generator`` (deterministic sequence of batches).
    """

    def __init__(
        self,
        n_features: int,
        feature_prob: torch.Tensor | float,
        n_instances: int = 1,
        seed: int = 0,
        n_chunks: int = 4,
    ) -> None:
        p = torch.as_tensor(feature_prob, dtype=torch.float32)
        if p.ndim == 1 and p.numel() == n_features:
            p = p[None, :]
        self.p = p.expand(n_instances, n_features).clone()
        if not bool(((self.p >= 0) & (self.p <= 1)).all()):
            raise ValueError("feature_prob must lie in [0, 1]")
        self.n = n_features
        self.K = n_instances
        self.seed = seed
        # The CPU RNG is single-threaded and dominates the step time for large K, so instances are split
        # into chunks with their own deterministic generators and drawn in parallel threads.
        self.n_chunks = max(1, min(n_chunks, n_instances))
        bounds = np.linspace(0, n_instances, self.n_chunks + 1).astype(int)
        self._chunks = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]
        self._gens = [torch.Generator().manual_seed(seed * 1_000_003 + 17 * c + 3) for c in range(len(self._chunks))]
        self._pool = ThreadPoolExecutor(max_workers=len(self._chunks)) if len(self._chunks) > 1 else None

    def _draw(self, c: int, batch: int) -> torch.Tensor:
        """One uniform draw per entry; ``u < p`` marks active entries and ``u / p`` is their Uniform[0,1) value."""
        a, b = self._chunks[c]
        u = torch.rand((b - a, batch, self.n), generator=self._gens[c])
        p = self.p[a:b, None, :]
        active = u < p
        return torch.where(active, u / p.clamp_min(torch.finfo(u.dtype).tiny), torch.zeros((), dtype=u.dtype))

    def sample(self, batch: int) -> torch.Tensor:
        """Draw one batch, shape ``[K, batch, n]``."""
        if self._pool is None:
            return self._draw(0, batch)
        parts = list(self._pool.map(lambda c: self._draw(c, batch), range(len(self._chunks))))
        return torch.cat(parts, dim=0)

    def sample_conditional(self, batch: int, feature: int, active: bool) -> torch.Tensor:
        """Draw a batch with feature ``feature`` forced active (``Uniform[0,1)``) or forced to 0.

        Other features follow their natural distribution. Used by the stratified eval set (design note 4).
        """
        x = self.sample(batch)
        if active:
            x[:, :, feature] = torch.rand((self.K, batch), generator=self._gens[0])
        else:
            x[:, :, feature] = 0.0
        return x

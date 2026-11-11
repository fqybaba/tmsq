"""Branch A (scale): the TopK SAEs of Lawson et al. (``tim-lawson/sae-pythia-{size}-deduped-x64-k32-layers-{N}``).

Pure torch / numpy, loaded from the repo's ``config.json`` + ``model.safetensors`` (no ``mlsae`` dependency). The
forward pass reproduces ``mlsae/model/autoencoders/topk.py`` and ``utils.py`` at commit ``f26303f``:

- ``standardize``: per token, ``mu = mean over d_model``, ``sd = std over d_model`` (torch default, unbiased),
  ``x_std = (x - mu) / (sd + 1e-5)``;
- ``z = (x_std - pre_encoder_bias) @ W_enc`` (the encoder has NO bias);
- ``f = ReLU(TopK_k(z))``: the k largest pre-activations are kept, then ReLU is applied to the kept values;
- ``x_hat = (f @ W_dec + pre_encoder_bias) * sd + mu``.

Layout in the safetensors file: ``encoder.weight [n_latents, n_inputs]``, ``decoder.weight [n_inputs, n_latents]``
(columns approximately unit-norm: ``mlsae`` renormalizes before the optimizer step, so the saved weights are one
Adam step off; ``|norm - 1|`` up to about 2e-3 in the 70m layer-4 file), ``pre_encoder_bias [n_inputs]``. Here
``W_enc = encoder.weight.T`` ([d_in, n_latents]) and ``W_dec = decoder.weight.T`` ([n_latents, d_in]) so that
``e_j = W_enc[:, j]`` and ``d_j = W_dec[j]`` as in ``tmsq.bridge``. The stored weights are used as they are (no
renormalization), as in ``mlsae``. The SAE input is the HF ``GPTNeoX`` hidden state after block ``N``
(``mlsae/model/transformers/pythia.py``: embeddings excluded, no final layer norm), i.e. TransformerLens
``blocks.N.hook_resid_post``.

Two forward paths give the same numbers up to float rounding: the dense one (``topk_activations`` / ``decode`` /
``reconstruct``, a transcription of ``mlsae``) and the sparse one (``encode_sparse`` / ``decode_sparse``), which
never builds the [T, n_latents] activation tensor and is what the streamed statistics use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

STD_EPS = 1e-5  # mlsae.model.autoencoders.utils.standardize(eps=1e-5), added to the std
NORM_TOL = 1e-2  # accepted |decoder column norm - 1| (observed up to about 2e-3); the maximum is recorded
NEAR_FRACS = (0.05, 0.1, 0.25)  # near-cut bands, as fractions of the per-token cut |z_(k)|


@dataclass(frozen=True)
class TopKSAE:
    W_enc: torch.Tensor  # [d_in, n_latents]
    W_dec: torch.Tensor  # [n_latents, d_in]
    b_pre: torch.Tensor  # [d_in], ``pre_encoder_bias``
    k: int
    standardize: bool = True
    dec_norm_max_dev: float = 0.0  # max |decoder column norm - 1| of the loaded file

    @property
    def d_in(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def n_latents(self) -> int:
        return int(self.W_enc.shape[1])


def load_topk_sae(config_path: Path | str, weights_path: Path | str) -> TopKSAE:
    """Load a Lawson TopK SAE from ``config.json`` + ``model.safetensors``; checks the shapes and decoder norms."""
    from safetensors.torch import load_file

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    w = load_file(str(weights_path))
    expected = {"encoder.weight", "decoder.weight", "pre_encoder_bias"}
    extra = set(w) - expected - {"last_nonzero"}
    if extra or not expected <= set(w):
        raise ValueError(f"unexpected safetensors keys: have {sorted(w)}, expected {sorted(expected)}")
    enc, dec, b = w["encoder.weight"].float(), w["decoder.weight"].float(), w["pre_encoder_bias"].float()
    n_in, n_lat = int(cfg["n_inputs"]), int(cfg["n_latents"])
    if tuple(enc.shape) != (n_lat, n_in) or tuple(dec.shape) != (n_in, n_lat) or tuple(b.shape) != (n_in,):
        raise ValueError(
            f"shape mismatch: enc {tuple(enc.shape)}, dec {tuple(dec.shape)}, b {tuple(b.shape)}, cfg {cfg}"
        )
    norms = dec.norm(dim=0)
    dev = float((norms - 1).abs().max())
    if not dev <= NORM_TOL:  # also catches NaN
        raise ValueError(f"decoder columns are not unit-norm: min {norms.min():.4f}, max {norms.max():.4f}")
    return TopKSAE(enc.T.contiguous(), dec.T.contiguous(), b, int(cfg["k"]), bool(cfg.get("standardize", True)), dev)


def standardize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mlsae`` standardize: returns ``x_std, mu, sd`` (``sd`` is the unbiased std over the last dim, before eps)."""
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    sd = xc.std(dim=-1, keepdim=True)
    return xc / (sd + STD_EPS), mu, sd


def pre_activation(x: torch.Tensor, sae: TopKSAE) -> torch.Tensor:
    """``z`` for every latent, [..., n_latents] (the TopK cut needs all latents, so no column subset here)."""
    xs = standardize(x)[0] if sae.standardize else x
    return (xs - sae.b_pre) @ sae.W_enc


def topk_activations(z: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense ``f = ReLU(TopK_k(z))`` [..., n_latents], plus the cut ``z_(k)`` (k-th largest) per token."""
    vals, idx = torch.topk(z, k=k, dim=-1, sorted=False)
    f = torch.zeros_like(z).scatter_(-1, idx, torch.relu(vals))
    return f, vals.amin(dim=-1)


def encode(x: torch.Tensor, sae: TopKSAE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense path: ``f`` [..., n_latents], ``z`` [..., n_latents] and the per-token cut ``z_(k)``."""
    z = pre_activation(x, sae)
    f, cut = topk_activations(z, sae.k)
    return f, z, cut


def decode(
    f: torch.Tensor, sae: TopKSAE, mu: torch.Tensor | None = None, sd: torch.Tensor | None = None
) -> torch.Tensor:
    """Dense reconstruction in the model's space (``mu``/``sd`` from :func:`standardize` if the SAE standardizes)."""
    return _unstandardize(f @ sae.W_dec + sae.b_pre, sae, mu, sd)


def reconstruct(x: torch.Tensor, sae: TopKSAE) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense ``x_hat`` and ``f`` for the full forward pass, as ``TopKSAE.forward`` in ``mlsae``."""
    xs, mu, sd = standardize(x) if sae.standardize else (x, None, None)
    f, _ = topk_activations((xs - sae.b_pre) @ sae.W_enc, sae.k)
    return decode(f, sae, mu, sd), f


def _unstandardize(xh: torch.Tensor, sae: TopKSAE, mu: torch.Tensor | None, sd: torch.Tensor | None) -> torch.Tensor:
    if not sae.standardize:
        return xh
    if mu is None or sd is None:
        raise ValueError("a standardizing SAE needs mu and sd to decode")
    return xh * sd + mu


@dataclass(frozen=True)
class SparseCode:
    """Top-k code of a batch of tokens: ``idx``/``vals`` [T, k] (``vals`` after ReLU), the cut ``z_(k)`` [T]."""

    idx: torch.Tensor
    vals: torch.Tensor
    cut: torch.Tensor
    mu: torch.Tensor | None
    sd: torch.Tensor | None


def encode_sparse(x: torch.Tensor, sae: TopKSAE) -> tuple[SparseCode, torch.Tensor]:
    """Sparse path for ``x`` [T, d]: the top-k code and the dense pre-activations ``z`` [T, n_latents]."""
    xs, mu, sd = standardize(x) if sae.standardize else (x, None, None)
    z = (xs - sae.b_pre) @ sae.W_enc
    vals, idx = torch.topk(z, k=sae.k, dim=-1, sorted=False)
    return SparseCode(idx, torch.relu(vals), vals.amin(dim=-1), mu, sd), z


def decode_sparse(code: SparseCode, sae: TopKSAE) -> torch.Tensor:
    """``x_hat`` [T, d] from a sparse code: gathers the k decoder rows per token instead of a dense matmul."""
    xh = torch.nn.functional.embedding_bag(code.idx, sae.W_dec, per_sample_weights=code.vals, mode="sum")
    return _unstandardize(xh + sae.b_pre, sae, code.mu, code.sd)


# ----------------------------------------------------------------------------------------- streamed stats


class TopKFPStats:
    """Streamed full-precision statistics of one TopK SAE, per latent and per token, in one pass.

    Per latent j (all O(n_latents) memory): firing count (``f_j > 0``), the same without position-0 tokens,
    sum and max of ``f_j`` (Duan's rate / mean / peak), the count of ``z_j > 0`` (how often a ReLU SAE with the
    same encoder would fire: the gap to ``fire`` is rank competition), sum and sum of squares of ``z_j`` (``sd_z``
    is the population sd; per-batch sums are float32, accumulated in float64), the count of tokens where j has
    rank k+1..2k, and near-cut band counts for each ``frac`` in ``fracs``, with the band width
    ``w_t = frac * |cut_t|``: ``near_off`` = j is not kept and ``z_j > cut_t - w_t``; ``near_on`` = j fires and
    ``z_j < cut_t + w_t`` (the latent at the cut itself is always in ``near_on``).

    Per token: L0, the cut ``z_(k)``; FVU sums in the model's space, separately for position-0 tokens and the rest.
    Each FVU is ``sum |x - x_hat|^2 / sum |x - mean x|^2`` with the mean taken over the tokens of that group, so
    the pooled FVU can be below both group values when the group means differ (position 0 has a large norm).

    The kept set is the first k entries of one sorted top-2k call. ``mlsae`` uses ``topk(k, sorted=False)``; the two
    can pick different members only at exact float ties of the cut (about 1e-5 of tokens on the 70m stream).
    """

    def __init__(self, sae: TopKSAE, fracs: tuple[float, ...] = NEAR_FRACS) -> None:
        n, d = sae.n_latents, sae.d_in
        self.sae, self.fracs, self.n_lat = sae, tuple(fracs), n
        self.n = 0
        self.n_pos0 = 0
        self.fire = np.zeros(n, np.float64)
        self.fire_nopos0 = np.zeros(n, np.float64)
        self.sum = np.zeros(n, np.float64)
        self.max = torch.zeros(n)
        self.n_pos_z = np.zeros(n, np.float64)
        self.sum_z = np.zeros(n, np.float64)
        self.sum_z2 = np.zeros(n, np.float64)
        self.rank2k = np.zeros(n, np.float64)
        self.near_off = np.zeros((len(self.fracs), n), np.float64)
        self.near_on = np.zeros((len(self.fracs), n), np.float64)
        self.l0 = 0.0
        self.below_k = 0.0
        self.cuts: list[torch.Tensor] = []
        # FVU sums for two token groups: 0 = position 0 of a block, 1 = all other positions
        self.g_n = np.zeros(2)
        self.g_sx = torch.zeros(2, d, dtype=torch.float64)
        self.g_sxx = np.zeros(2)
        self.g_sse = np.zeros(2)

    def _count(self, idx: torch.Tensor, weights: torch.Tensor | None = None) -> np.ndarray:
        w = None if weights is None else weights.double()
        return torch.bincount(idx, weights=w, minlength=self.n_lat).double().numpy()

    @torch.no_grad()
    def update(self, x: torch.Tensor, pos0: torch.Tensor) -> None:
        """``x``: [T, d] residuals; ``pos0``: [T] bool, True for tokens at position 0 of their block."""
        sae, k = self.sae, self.sae.k
        xs, mu, sd = standardize(x) if sae.standardize else (x, None, None)
        z = (xs - sae.b_pre) @ sae.W_enc
        # one sorted top-2k call gives the kept set (ranks 1..k), the cut and ranks k+1..2k, consistent at exact ties
        v2, i2 = torch.topk(z, k=2 * k, dim=-1, sorted=True)
        raw, idx, cut = v2[:, :k], i2[:, :k], v2[:, k - 1]  # kept pre-activations before ReLU
        fv = torch.relu(raw)
        code = SparseCode(idx, fv, cut, mu, sd)
        on = fv > 0
        self.n += x.shape[0]
        self.n_pos0 += int(pos0.sum())
        self.fire += self._count(idx[on])
        self.fire_nopos0 += self._count(idx[on & ~pos0[:, None]])
        self.sum += self._count(idx.reshape(-1), fv.reshape(-1))
        self.max.scatter_reduce_(0, idx.reshape(-1), fv.reshape(-1), reduce="amax", include_self=True)
        self.n_pos_z += (z > 0).sum(0).double().numpy()
        self.sum_z += z.sum(0).double().numpy()
        self.sum_z2 += torch.linalg.vector_norm(z, dim=0).double().pow(2).numpy()
        self.rank2k += self._count(i2[:, k:].reshape(-1))
        for i, frac in enumerate(self.fracs):
            w = frac * cut.abs()
            lo = (cut - w)[:, None]
            self.near_off[i] += (z > lo).sum(0).double().numpy() - self._count(idx[raw > lo])
            self.near_on[i] += self._count(idx[on & (raw < (cut + w)[:, None])])
        l0 = on.sum(1)
        self.l0 += float(l0.sum())
        self.below_k += float((l0 < k).sum())
        self.cuts.append(cut.clone())
        xh = decode_sparse(code, sae)
        for g, m in ((0, pos0), (1, ~pos0)):
            if m.any():
                xg = x[m].double()
                self.g_n[g] += xg.shape[0]
                self.g_sx[g] += xg.sum(0)
                self.g_sxx[g] += float(xg.pow(2).sum())
                self.g_sse[g] += float((xg - xh[m].double()).pow(2).sum())

    def _fvu(self, groups: tuple[int, ...]) -> float:
        n = self.g_n[list(groups)].sum()
        sx = self.g_sx[list(groups)].sum(0)
        sst = self.g_sxx[list(groups)].sum() - float((sx**2).sum()) / n
        return float(self.g_sse[list(groups)].sum() / sst)

    def table(self) -> dict[str, np.ndarray]:
        """Per-latent table (one entry per latent, in latent order)."""
        n, n_rest = self.n, self.n - self.n_pos0
        mean_z = self.sum_z / n
        out = {
            "fire": self.fire.copy(),
            "rate": self.fire / n,
            "rate_nopos0": self.fire_nopos0 / n_rest,
            "pos0_share": np.divide(
                self.fire - self.fire_nopos0, self.fire, out=np.full(self.n_lat, np.nan), where=self.fire > 0
            ),
            "mean": self.sum / n,
            "mu_plus": np.divide(self.sum, self.fire, out=np.full(self.n_lat, np.nan), where=self.fire > 0),
            "peak": self.max.double().numpy(),
            "rate_pos_z": self.n_pos_z / n,
            "mean_z": mean_z,
            "sd_z": np.sqrt(np.maximum(self.sum_z2 / n - mean_z**2, 0.0)),
            "rate_rank2k": self.rank2k / n,
        }
        for i, frac in enumerate(self.fracs):
            out[f"near_off_{frac:g}"] = self.near_off[i] / n
            out[f"near_on_{frac:g}"] = self.near_on[i] / n
        return out

    def summary(self, active_rate: float) -> dict[str, Any]:
        """Scalar summary: FVU, L0, cut distribution, firing-rate distribution, counts above the rate thresholds."""
        n = self.n
        rate = self.fire / n
        rate_np0 = self.fire_nopos0 / (n - self.n_pos0)
        cut = torch.cat(self.cuts).double().numpy()
        cq = np.quantile(cut, [0.01, 0.1, 0.5, 0.9, 0.99])
        rq = np.quantile(rate, [0.5, 0.9, 0.99])
        thr = {"1x": active_rate, "0.5x": active_rate / 2, "0.1x": active_rate / 10}
        pos0_dom = (rate > active_rate) & ((self.fire - self.fire_nopos0) >= 0.9 * self.fire)
        return {
            "tokens": n, "tokens_pos0": self.n_pos0,
            "fvu": self._fvu((0, 1)), "fvu_nopos0": self._fvu((1,)),
            "fvu_pos0": self._fvu((0,)) if self.g_n[0] > 1 else None,
            "l0_mean": self.l0 / n, "frac_tokens_l0_below_k": self.below_k / n,
            "cut_mean": float(cut.mean()), "cut_std": float(cut.std()), "cut_min": float(cut.min()),
            "cut_q01_q10_q50_q90_q99": [float(v) for v in cq], "cut_nonpositive_frac": float((cut <= 0).mean()),
            "rate_mean": float(rate.mean()), "rate_median": float(rq[0]), "rate_p90": float(rq[1]),
            "rate_p99": float(rq[2]), "rate_max": float(rate.max()), "n_never_fire": int((self.fire == 0).sum()),
            "active_rate": active_rate,
            "n_above_rate": {k: int((rate > v).sum()) for k, v in thr.items()},
            "n_above_rate_nopos0": {k: int((rate_np0 > v).sum()) for k, v in thr.items()},
            "n_active_pos0_dominated": int(pos0_dom.sum()),
            "n_above_rate_pos_z": {k: int((self.n_pos_z / n > v).sum()) for k, v in thr.items()},
            "mean_rate_pos_z": float(self.n_pos_z.sum() / n / self.n_lat),
        }  # fmt: skip

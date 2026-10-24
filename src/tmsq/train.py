"""Training loop (CLAUDE.md, "Model and data").

AdamW (``weight_decay`` on ``W`` and, by default, on ``b``, as in the official notebook), constant lr,
importance-weighted MSE: per instance, mean over the batch of ``sum_i I_i (x_i - x_hat_i)^2``; the K
per-instance losses are summed so that every instance receives its own gradient.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from .data import FeatureSampler
from .model import ToyModel


@dataclass
class TrainConfig:
    steps: int = 10_000
    batch: int = 1024
    lr: float = 1e-3
    weight_decay: float = 0.01
    decay_bias: bool = True
    lr_schedule: str = "constant"  # "constant" | "warmup_cosine" (Phase 4)
    warmup_steps: int = 0
    min_lr_frac: float = 0.0
    decay_mode: str = "adamw"  # "adamw" (settings A-C) | "liu" (setting D, per-instance gamma on W only)
    snapshot_steps: tuple[int, ...] = field(default_factory=tuple)
    progress: bool = True
    log_every: int = 100


@dataclass
class TrainResult:
    loss_history: np.ndarray  # [steps, K], per-instance loss at every step
    snapshots: dict[int, tuple[torch.Tensor, torch.Tensor]]  # step -> (W, b) clones
    seconds: float


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate at ``step`` (0-based)."""
    if cfg.lr_schedule == "constant":
        return cfg.lr
    if cfg.lr_schedule == "warmup_cosine":
        min_lr = cfg.min_lr_frac * cfg.lr
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return cfg.lr * step / cfg.warmup_steps
        t = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return min_lr + (cfg.lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))
    raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")


def importance_weighted_loss(x: torch.Tensor, out: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
    """Per-instance loss ``[K]``: mean over batch of ``sum_i I_i (x_i - x_hat_i)^2``. ``importance [K, n]``."""
    err = importance[:, None, :] * (x - out) ** 2
    return err.sum(dim=-1).mean(dim=-1)


def make_optimizer(model: ToyModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    groups = [
        {"params": [model.W], "weight_decay": cfg.weight_decay},
        {"params": [model.b], "weight_decay": cfg.weight_decay if cfg.decay_bias else 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr)


NORM_EPS = 1e-8  # Liu et al. add the Adam eps (1e-8) to the feature norms


class LiuDecayAdam:
    """Adam without built-in weight decay, plus Liu, Liu & Gore's piecewise decay on ``W`` (CLAUDE.md, Setting D).

    Before each Adam step, per instance k and feature column ``W_i`` (layout ``[K, m, n]``):

    - ``gamma_k >= 0``: decoupled decay ``W_i <- W_i (1 - lr gamma_k)``, exactly as ``torch.optim.AdamW``;
    - ``gamma_k < 0``: norm-targeting term ``W_i <- W_i - lr gamma_k W_i (1/||W_i|| - 1)``, gradient descent on
      ``(||W_i|| - 1)^2`` (their ``exp/adamw.py``; NOT unbounded growth).

    ``b`` is never decayed. ``gamma`` is a ``[K]`` vector, so weak and strong regimes can share one batched run.
    """

    def __init__(self, model: ToyModel, lr: float, gamma) -> None:
        self.W = model.W
        g = torch.as_tensor(gamma, dtype=model.W.dtype).reshape(-1)
        if g.numel() == 1:
            g = g.expand(model.K)
        if g.numel() != model.K:
            raise ValueError(f"gamma needs {model.K} values, got {g.numel()}")
        self.gamma = g.reshape(-1, 1, 1).clone()
        self.inner = torch.optim.Adam([{"params": [model.W]}, {"params": [model.b]}], lr=lr)
        self.param_groups = self.inner.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        lr = self.param_groups[0]["lr"]
        W, g = self.W, self.gamma
        decayed_pos = W * (1 - lr * g)
        norms = W.norm(dim=1, keepdim=True) + NORM_EPS
        decayed_neg = W + lr * g * W * (1 - 1 / norms)
        W.copy_(torch.where(g >= 0, decayed_pos, decayed_neg))
        self.inner.step()


def train(
    model: ToyModel,
    sampler: FeatureSampler,
    importance: torch.Tensor,
    cfg: TrainConfig,
    desc: str = "train",
    gamma=None,
) -> TrainResult:
    """Train all ``K`` instances in parallel; returns the full per-instance loss history and snapshots.

    ``gamma`` (``[K]``) is required when ``cfg.decay_mode == "liu"`` (setting D).
    """
    importance = torch.as_tensor(importance, dtype=torch.float32)
    if importance.ndim == 1:
        importance = importance[None, :]
    importance = importance.expand(model.K, model.n)
    if cfg.decay_mode == "liu":
        if gamma is None:
            raise ValueError("decay_mode 'liu' needs gamma")
        opt = LiuDecayAdam(model, cfg.lr, gamma)
    else:
        opt = make_optimizer(model, cfg)
    history = torch.empty(cfg.steps, model.K)
    snapshots: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    snap_at = set(cfg.snapshot_steps)
    t0 = time.perf_counter()
    pbar = tqdm(range(cfg.steps), desc=desc, disable=not cfg.progress, mininterval=1.0, dynamic_ncols=True)
    for step in pbar:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x = sampler.sample(cfg.batch)
        out = model(x)
        per_inst = importance_weighted_loss(x, out, importance)
        loss = per_inst.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history[step] = per_inst.detach()
        if (step + 1) in snap_at:
            snapshots[step + 1] = (model.W.detach().clone(), model.b.detach().clone())
        if step % cfg.log_every == 0:
            pbar.set_postfix(mean_loss=f"{per_inst.mean().item():.4g}", lr=f"{lr:.2e}", refresh=False)
    return TrainResult(history.numpy(), snapshots, time.perf_counter() - t0)


def plateau_ratio(loss_history: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """Per instance: mean loss over the last ``frac`` of steps divided by the mean over the preceding ``frac``.

    Values near 1 mean the loss has plateaued; values well below 1 mean it was still falling.
    """
    steps = loss_history.shape[0]
    w = max(1, int(frac * steps))
    last = loss_history[-w:].mean(axis=0)
    prev = loss_history[-2 * w : -w].mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return last / prev

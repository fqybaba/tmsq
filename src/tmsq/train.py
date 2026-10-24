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


def train(
    model: ToyModel,
    sampler: FeatureSampler,
    importance: torch.Tensor,
    cfg: TrainConfig,
    desc: str = "train",
) -> TrainResult:
    """Train all ``K`` instances in parallel; returns the full per-instance loss history and snapshots."""
    importance = torch.as_tensor(importance, dtype=torch.float32)
    if importance.ndim == 1:
        importance = importance[None, :]
    importance = importance.expand(model.K, model.n)
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

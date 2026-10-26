"""Shared experiment infrastructure: configs, result directories, provenance, tidy tables."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ..data import FeatureSampler
from ..metrics import geometry_table
from ..model import ToyModel
from ..train import TrainConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_ROOT = PROJECT_ROOT / "figures"
LOSS_NORMALIZATION = "per instance: mean over batch of sum_i I_i (x_i - x_hat_i)^2 (paper's sum over features)"


def parse_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None, help="override config steps (smoke tests)")
    ap.add_argument("--tag", type=str, default="", help="suffix for the results directory")
    return ap.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_runtime(cfg: dict[str, Any]) -> None:
    """Apply runtime settings from the config: ``threads`` (bit-exact reruns need the same thread count)."""
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))


def git_info() -> dict[str, Any]:
    """Commit hash and dirty-file count; the literal "no commit" when the repo has no commits."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--short"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        dirty = len([line for line in status.stdout.splitlines() if line.strip()])
        if commit.returncode != 0:
            return {"commit": "no commit", "dirty_files": dirty}
        return {"commit": commit.stdout.strip(), "dirty_files": dirty}
    except OSError:
        return {"commit": "git unavailable", "dirty_files": -1}


def make_run_dir(exp_name: str, tag: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}{'-' + tag if tag else ''}"
    out = RESULTS_ROOT / exp_name / name
    while out.exists():  # never overwrite previous results
        time.sleep(1)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = RESULTS_ROOT / exp_name / f"{stamp}{'-' + tag if tag else ''}"
    out.mkdir(parents=True)
    return out


def save_provenance(run_dir: Path, config: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    meta = {
        "git": git_info(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "loss_normalization": LOSS_NORMALIZATION,
        "loss_history_layout": "loss_history.npy: float32 [steps, K], one per-instance loss per step",
        "weights_layout": "final.pt / snapshot_*.pt: dict with W [K, m, n], b [K, n] and numpy metadata; "
        "load with tmsq.experiments.common.load_weights (weights_only=True with numpy types allow-listed)",
    }
    if extra:
        meta.update(extra)
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def log_density_grid(start: float, stop: float, num: int) -> np.ndarray:
    """Log-spaced feature densities ``1 - S`` from ``start`` down to ``stop``."""
    return np.geomspace(start, stop, num)


def expand_sweep(densities: np.ndarray, n_seeds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One instance per (density, seed_idx): returns ``density [K]``, ``sweep_idx [K]``, ``seed_idx [K]``."""
    dens = np.repeat(np.asarray(densities, dtype=np.float64), n_seeds)
    sweep_idx = np.repeat(np.arange(len(densities)), n_seeds)
    seed_idx = np.tile(np.arange(n_seeds), len(densities))
    return dens, sweep_idx, seed_idx


def importance_curve(n: int, r: float) -> np.ndarray:
    """``I_i = r**i``, i = 0..n-1 (``r = 1`` gives uniform importance)."""
    return np.asarray(r, dtype=np.float64) ** np.arange(n)


def build_model_and_sampler(cfg: dict[str, Any], density: np.ndarray) -> tuple[ToyModel, FeatureSampler]:
    K = len(density)
    model = ToyModel(K, cfg["n_features"], cfg["n_hidden"], seed=cfg["seed"], init_std=cfg.get("init_std"))
    p = torch.as_tensor(density, dtype=torch.float32)[:, None]
    sampler = FeatureSampler(cfg["n_features"], p, n_instances=K, seed=cfg["seed"] + 1)
    return model, sampler


def train_config_from(cfg: dict[str, Any], steps_override: int | None = None) -> TrainConfig:
    t = cfg["train"]
    steps = int(t["steps"]) if steps_override is None else int(steps_override)
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    snaps = tuple(int(s) for s in t.get("snapshot_steps", []) if int(s) < steps)
    if t.get("snapshot_half", False):
        snaps = tuple(sorted(set(snaps) | {steps // 2}))
    return TrainConfig(
        steps=steps,
        batch=int(t["batch"]),
        lr=float(t["lr"]),
        weight_decay=float(t.get("weight_decay", 0.01)),
        decay_bias=bool(t.get("decay_bias", True)),
        lr_schedule=str(t.get("lr_schedule", "constant")),
        warmup_steps=int(t.get("warmup_steps", 0)),
        min_lr_frac=float(t.get("min_lr_frac", 0.0)),
        decay_mode=str(t.get("decay_mode", "adamw")),
        snapshot_steps=snaps,
        progress=True,
    )


def feature_frame(
    W: torch.Tensor,
    b: torch.Tensor,
    density: np.ndarray,
    sweep_idx: np.ndarray,
    seed_idx: np.ndarray,
    importance: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Tidy per-feature table: one row per (instance, feature) with all geometry metrics."""
    K, _, n = W.shape
    g = geometry_table(W, b)
    rows = {
        "instance": np.repeat(np.arange(K), n),
        "sweep_idx": np.repeat(sweep_idx, n),
        "seed_idx": np.repeat(seed_idx, n),
        "density": np.repeat(density, n),
        "S": np.repeat(1.0 - density, n),
        "feature": np.tile(np.arange(n), K),
        "importance": np.tile(np.asarray(importance), K),
    }
    for key, arr in g.items():
        rows[key] = np.asarray(arr).reshape(-1)
    df = pd.DataFrame(rows)
    if extra:
        for key, val in extra.items():
            df[key] = val
    return df


def save_weights(run_dir: Path, name: str, W: torch.Tensor, b: torch.Tensor, **meta: Any) -> None:
    torch.save({"W": W.detach().cpu(), "b": b.detach().cpu(), **meta}, run_dir / f"{name}.pt")


def numpy_safe_globals() -> list[Any]:
    """The numpy types that ``torch.load(weights_only=True)`` must allow to unpickle the metadata arrays."""
    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import _reconstruct
    globs: list[Any] = [np.ndarray, np.dtype, _reconstruct]
    try:
        import numpy.dtypes as nd

        globs += [getattr(nd, n) for n in dir(nd) if n.endswith("DType")]
    except ImportError:
        pass
    return globs


def load_weights(path: Path) -> dict[str, Any]:
    """Load a ``final.pt`` / ``snapshot_*.pt`` file written by :func:`save_weights`.

    The files hold ``W`` and ``b`` tensors plus numpy metadata arrays (density, seed_idx, importance, ...).
    They are loaded with ``weights_only=True``; the numpy array types are allow-listed explicitly, so no
    arbitrary code can run during unpickling.
    """
    with torch.serialization.safe_globals(numpy_safe_globals()):
        return torch.load(Path(path), weights_only=True, map_location="cpu")


def run_figure_dir(run_dir: Path) -> Path:
    """Per-run figure directory (``<run>/figures``); never overwritten by other runs."""
    d = run_dir / "figures"
    d.mkdir(exist_ok=True)
    return d


def publish_figures(fig_dir: Path, tag: str) -> None:
    """Copy a full run's figures (no ``--tag``) to the shared ``figures/`` folder; tagged runs stay private."""
    if tag:
        return
    FIGURES_ROOT.mkdir(exist_ok=True)
    for p in fig_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, FIGURES_ROOT / p.name)


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")

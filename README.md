# TMS-Q: Superposition x Quantization

Replication of key results from *Toy Models of Superposition* (Elhage et al., 2022) and an original extension:
which features stored in superposition does post-training weight quantization (RTN) break first, and can
superposition geometry predict this in advance?

Status: Phase 0 (setup) and Phase 1 (replication). See `notes/LOG.md` for the running log and `CLAUDE.md` for
all canonical definitions.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest
```

## Running experiments

Every experiment is a Python entry point driven by a YAML config and writes to `results/<exp>/<timestamp>/`
(config copy, git hash, weights, loss curves, per-feature metrics) and `figures/`:

```powershell
python -m tmsq.experiments.rep_5in2  --config configs/rep_5in2.yaml       # setting A, ~1 min
python -m tmsq.experiments.rep_20in5 --config configs/rep_20in5.yaml      # setting B, ~2 min
python -m tmsq.experiments.rep_dims  --config configs/rep_dims.yaml       # setting C, ~10 min
python -m tmsq.experiments.rep_dims  --config configs/rep_dims_fine.yaml  # setting C, fine grid near D* = 1/2
```

Phase 2 (post-training quantization of the Phase 1 models; one invocation per setting, a few minutes for C):

```powershell
python -m tmsq.experiments.quant_sweep --config configs/quant_sweep.yaml --setting A
python -m tmsq.experiments.quant_sweep --config configs/quant_sweep.yaml --setting B
python -m tmsq.experiments.quant_sweep --config configs/quant_sweep.yaml --setting C
```

Each run writes `results/quant_sweep/<timestamp>-<setting>/features.parquet` (one row per model instance x
quantization configuration x feature: FP geometry predictors, damage metrics, realized-perturbation
covariates), `models.parquet` (one row per instance x configuration) and `sanity.json`.

Figures can be regenerated from any finished run without retraining:

```powershell
python -m tmsq.experiments.replot --run results/rep_20in5/<timestamp>
```

`tests/test_blackbox.py` is an independently written suite (38 tests) that checks the package against the
CLAUDE.md definitions without reading the implementation; `pytest` runs it together with the unit tests.

Reproducibility: every run directory holds `config.yaml`, `meta.json` (git state, library versions, thread
count, loss normalization), the weights (`final.pt`, load them with `tmsq.experiments.common.load_weights`),
`loss_history.npy`, the per-feature and per-instance tables and its own `figures/`. Bit-exact reruns need the
same `threads` value as `meta.json` records (set `threads:` in the config); different thread counts give
results equal to about 8 decimals in the loss but not bit-identical weights.

## Layout

```
src/tmsq/{data,model,train,metrics,plots}.py   core library
src/tmsq/experiments/                           entry points
configs/   tests/   results/ (gitignored)   figures/   notes/LOG.md
```

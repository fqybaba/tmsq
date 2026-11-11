"""Phase 6b: download the bridge artifacts into the Hugging Face cache and record what was fetched.

``.venv-bridge\\Scripts\\python -m tmsq.experiments.bridge_fetch``

Fetches only the files the bridge plan (notes/LOG.md, Phase 6b, D7) approves: the Pythia-70M-deduped weights,
config and tokenizer (not the duplicate ``pytorch_model.bin``), the layer-4 residual SAE folder of
``pythia-70m-deduped-res-sm``, and the WikiText-2-raw train parquet. The cache stays at the HF default
(outside OneDrive). The manifest records each file's repo revision, size and SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .common import PROJECT_ROOT, load_config, make_run_dir, save_provenance

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# The Xet transfer backend stalled on this machine (a 302 MB file made no progress for 30 min, twice; with Xet
# disabled it took 53 s, 2026-11-11). Plain HTTPS downloads the same blobs; set HF_HUB_DISABLE_XET=0 to override.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

FILES = [
    ("EleutherAI/pythia-70m-deduped", "model", "model.safetensors"),
    ("EleutherAI/pythia-70m-deduped", "model", "config.json"),
    ("EleutherAI/pythia-70m-deduped", "model", "tokenizer.json"),
    ("EleutherAI/pythia-70m-deduped", "model", "tokenizer_config.json"),
    ("EleutherAI/pythia-70m-deduped", "model", "special_tokens_map.json"),
    ("EleutherAI/pythia-70m-deduped", "model", "generation_config.json"),
    ("ctigges/pythia-70m-deduped__res-sm_processed", "model", "4-res-sm/cfg.json"),
    ("ctigges/pythia-70m-deduped__res-sm_processed", "model", "4-res-sm/sae_weights.safetensors"),
    ("ctigges/pythia-70m-deduped__res-sm_processed", "model", "4-res-sm/sparsity.safetensors"),
    ("Salesforce/wikitext", "dataset", "wikitext-2-raw-v1/train-00000-of-00001.parquet"),
]

# Branch A (scale), downloads approved by the owner (notes/LOG.md, 2026-11-11): Pythia-160M/410M-deduped and the
# Lawson TopK SAEs. The 70M model (already cached by ``duan``) is listed again so that this manifest covers every
# weight file the scale runs load; the WikiText parquet stays in the ``duan`` manifest. Every repo of this set is
# fetched at the revision pinned in configs/bridge_scale.yaml.
OPTIONAL = {"generation_config.json"}  # absent from some repos; any other failed download is an error
_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
SCALE_FILES = [
    *[("EleutherAI/pythia-70m-deduped", "model", f) for f in _MODEL_FILES],  # cached; listed so the manifest has it
    *[("EleutherAI/pythia-160m-deduped", "model", f) for f in _MODEL_FILES],
    *[("EleutherAI/pythia-410m-deduped", "model", f) for f in _MODEL_FILES],
    *[
        (f"tim-lawson/sae-pythia-{size}-deduped-x64-k32-layers-{layer}", "model", f)
        for size, layer in (("70m", 4), ("160m", 4), ("160m", 9), ("410m", 4), ("410m", 19))
        for f in ("config.json", "model.safetensors")
    ],
]
SETS = {"duan": FILES, "scale": SCALE_FILES}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def portable(path: Path) -> str:
    """Cache path with the machine-specific Hugging Face home replaced by ``<HF_HOME>`` (results are public)."""
    from huggingface_hub import constants

    home = Path(constants.HF_HOME).resolve()
    p = Path(path).resolve()
    return str(Path("<HF_HOME>") / p.relative_to(home)) if p.is_relative_to(home) else p.name


def scale_pins() -> dict[str, str]:
    """Repo -> pinned revision for the ``scale`` set, from configs/bridge_scale.yaml."""
    cfg = load_config(PROJECT_ROOT / "configs" / "bridge_scale.yaml")
    pins = {s["model"]: s["model_revision"] for s in cfg["sizes"].values()}
    for key, rev in cfg["sae_revisions"].items():
        size, layer = key.split("-")
        pins[cfg["sae_repo"].format(size=size, layer=layer)] = rev
    return pins


def main() -> None:
    import argparse

    from huggingface_hub import HfApi, hf_hub_download

    ap = argparse.ArgumentParser(description="fetch the bridge artifacts into the HF cache")
    ap.add_argument("--set", choices=sorted(SETS), default="duan")
    args = ap.parse_args()
    files = SETS[args.set]
    pins = scale_pins() if args.set == "scale" else {}
    missing = sorted({repo for repo, _, _ in files} - set(pins)) if pins else []
    if missing:
        raise SystemExit(f"no pinned revision in configs/bridge_scale.yaml for {missing}")
    api = HfApi()
    revisions: dict[tuple[str, str], str] = {}
    rows = []
    for repo, kind, fname in files:
        key = (repo, kind)
        if key not in revisions:
            if repo in pins:
                revisions[key] = pins[repo]
            else:
                info = api.dataset_info(repo) if kind == "dataset" else api.model_info(repo)
                revisions[key] = info.sha
        try:
            path = Path(hf_hub_download(repo, fname, repo_type=kind, revision=revisions[key]))
        except Exception as e:
            if fname not in OPTIONAL:
                raise
            print(f"skip optional {repo}/{fname}: {type(e).__name__}")
            continue
        rows.append(
            {
                "repo": repo,
                "type": kind,
                "file": fname,
                "revision": revisions[key],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "local": portable(path),
            }  # fmt: skip
        )
        print(f"{repo}/{fname}: {path.stat().st_size:,} B")
    out = make_run_dir("bridge", "fetch" if args.set == "duan" else f"fetch-{args.set}")
    save_provenance(out, {"set": args.set, "files": [list(f) for f in files]})
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"total {sum(r['bytes'] for r in rows) / 1e6:.1f} MB; manifest {out / 'manifest.json'}")


if __name__ == "__main__":
    main()

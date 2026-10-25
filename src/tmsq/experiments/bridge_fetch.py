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

from .common import make_run_dir, save_provenance

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    revisions: dict[tuple[str, str], str] = {}
    rows = []
    for repo, kind, fname in FILES:
        key = (repo, kind)
        if key not in revisions:
            info = api.dataset_info(repo) if kind == "dataset" else api.model_info(repo)
            revisions[key] = info.sha
        try:
            path = Path(hf_hub_download(repo, fname, repo_type=kind, revision=revisions[key]))
        except Exception as e:  # optional files (e.g. generation_config.json) may not exist
            print(f"skip {repo}/{fname}: {type(e).__name__}")
            continue
        rows.append(
            {
                "repo": repo,
                "type": kind,
                "file": fname,
                "revision": revisions[key],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "local": str(path),
            }  # fmt: skip
        )
        print(f"{repo}/{fname}: {path.stat().st_size:,} B")
    out = make_run_dir("bridge", "fetch")
    save_provenance(out, {"files": [list(f) for f in FILES]})
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"total {sum(r['bytes'] for r in rows) / 1e6:.1f} MB; manifest {out / 'manifest.json'}")


if __name__ == "__main__":
    main()

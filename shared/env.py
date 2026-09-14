"""Cache / offline environment setup.

MUST run before torch, transformers, sentence_transformers or datasets are
imported. `shared/__init__.py` does that for you, so always import `shared`
(or anything under it) first.

Contract
--------
* Every model weight, dataset and repo clone lives under <project>/data.
* Nothing is downloaded twice: HuggingFace caches into data/models/hf, and a
  state marker in data/.state records each completed download.
* After bootstrap, runs are fully OFFLINE by default. Set
  CODERAG_ALLOW_DOWNLOAD=1 to temporarily re-enable network access.
"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("CODERAG_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]   # shared/env.py -> shared -> root


ROOT = project_root()
DATA = ROOT / "data"
HF_CACHE = DATA / "models" / "hf"
TORCH_CACHE = DATA / "models" / "torch"
DS_CACHE = DATA / "datasets" / "hf"
STATE = DATA / ".state"


def configure(allow_download: bool | None = None) -> None:
    """Point every library's cache at the project-local data dir."""
    for p in (HF_CACHE, TORCH_CACHE, DS_CACHE, STATE):
        p.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CODERAG_HOME", str(ROOT))
    os.environ["HF_HOME"] = str(DATA / "models")
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(HF_CACHE)
    os.environ["HF_DATASETS_CACHE"] = str(DS_CACHE)
    os.environ["TORCH_HOME"] = str(TORCH_CACHE)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if allow_download is None:
        allow_download = os.environ.get("CODERAG_ALLOW_DOWNLOAD", "0") == "1"

    flag = "0" if allow_download else "1"
    os.environ["HF_HUB_OFFLINE"] = flag
    os.environ["TRANSFORMERS_OFFLINE"] = flag
    os.environ["HF_DATASETS_OFFLINE"] = flag


def offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "1") == "1"

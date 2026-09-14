"""One-time acquisition of the evaluation datasets (global, all systems)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from . import repos as repo_utils
from . import state
from .paths import paths


def _ensure_git_dataset(name: str, spec: Dict[str, Any], force: bool) -> Path:
    dest = paths.datasets_raw / name
    key = f"dataset:{name}"
    if force:
        state.clear(key)
    if state.is_done(key, check_path=dest):
        print(f"  [skip] dataset {name} already present")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  [clone] dataset {name}: {spec['url']}")
        repo_utils._run(["git", "clone", "--depth", "1", "--branch",
                         spec.get("ref", "main"), spec["url"], str(dest)])
    sha = repo_utils.head_sha(dest)
    state.save_pin(f"dataset:{name}", url=spec["url"], commit=sha)
    state.mark_done(key, path=str(dest), commit=sha)
    return dest


def _ensure_hf_dataset(name: str, spec: Dict[str, Any], force: bool) -> Path:
    key = f"dataset:{name}"
    if force:
        state.clear(key)
    if state.is_done(key, check_path=paths.datasets_hf):
        print(f"  [skip] dataset {name} already in HF cache")
        return paths.datasets_hf

    from datasets import load_dataset      # late import: caches already configured

    print(f"  [download] HF dataset {spec['repo_id']}")
    ds = load_dataset(spec["repo_id"], spec.get("config"),
                      cache_dir=str(paths.datasets_hf))
    splits = {k: len(v) for k, v in ds.items()} if hasattr(ds, "items") else {"train": len(ds)}
    state.mark_done(key, repo_id=spec["repo_id"], splits=splits,
                    cache_dir=str(paths.datasets_hf))
    print(f"  [ok] {name}: {splits}")
    return paths.datasets_hf


def ensure_dataset(name: str, spec: Dict[str, Any], force: bool = False) -> Path | None:
    kind = spec.get("kind", "git")
    try:
        if kind == "git":
            return _ensure_git_dataset(name, spec, force)
        if kind == "hf":
            return _ensure_hf_dataset(name, spec, force)
    except Exception as exc:
        if spec.get("optional"):
            print(f"  [warn] optional dataset {name} failed: {exc}")
            return None
        raise
    raise ValueError(f"unknown dataset kind: {kind}")


def ensure_all(datasets_cfg: Dict[str, Any], force: bool = False) -> Dict[str, Path | None]:
    return {n: ensure_dataset(n, s, force) for n, s in datasets_cfg.items()}


def codereval_tasks(language: str = "python") -> list:
    """Load CoderEval tasks from the cloned repo (CoderEval4Python.json etc.)."""
    root = paths.datasets_raw / "codereval"
    target = f"codereval4{language}.json".lower()
    hits = [p for p in root.rglob("*.json") if p.name.lower() == target]
    if not hits:
        raise FileNotFoundError(
            f"{target} not found under {root}. Run: python scripts/bootstrap.py --only datasets")
    data = json.loads(hits[0].read_text(encoding="utf-8"))
    return data.get("RECORDS", data if isinstance(data, list) else [])


def repoexec_tasks(split: str = "test") -> list:
    from datasets import load_dataset
    ds = load_dataset("Fsoft-AIC/RepoExec", cache_dir=str(paths.datasets_hf))
    key = split if split in ds else list(ds.keys())[0]
    return [dict(r) for r in ds[key]]

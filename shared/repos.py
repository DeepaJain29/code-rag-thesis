"""Pinned git clones, downloaded once, shared by every system.

A repo is cloned to data/repos/<name> at an exact commit; the resolved sha goes
into configs/pins.lock.json so Systems B and C index the SAME snapshot System A
did - required for the controlled comparison (thesis doc, section 6).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

from . import state
from .paths import paths


def _run(cmd: List[str], cwd: Path | None = None) -> str:
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{res.stderr.strip()}")
    return res.stdout.strip()


def head_sha(repo_dir: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)


def ensure_repo(spec: Dict[str, Any], force: bool = False) -> Path:
    """Clone once, check out the pinned commit. Idempotent."""
    name = spec["name"]
    dest = paths.repo(name)
    key = f"repo:{name}"

    if force:
        state.clear(key)
    if state.is_done(key, check_path=dest):
        print(f"  [skip] repo {name} already present at {dest}")
        return dest

    if not dest.exists():
        print(f"  [clone] {spec['url']} -> {dest}")
        _run(["git", "clone", "--filter=blob:none", "--no-checkout",
              spec["url"], str(dest)])

    ref = spec.get("commit") or spec.get("ref") or "HEAD"
    print(f"  [checkout] {name} @ {ref}")
    _run(["git", "fetch", "--filter=blob:none", "origin", ref], cwd=dest)
    try:
        _run(["git", "checkout", "--force", ref], cwd=dest)
    except RuntimeError:
        _run(["git", "checkout", "--force", "FETCH_HEAD"], cwd=dest)

    sha = head_sha(dest)
    state.save_pin(f"repo:{name}", url=spec["url"], ref=spec.get("ref"), commit=sha)
    state.mark_done(key, path=str(dest), commit=sha, url=spec["url"])
    print(f"  [pinned] {name} = {sha}")
    return dest


def ensure_repos(specs: Iterable[Dict[str, Any]], force: bool = False) -> Dict[str, Path]:
    return {s["name"]: ensure_repo(s, force=force) for s in specs}


def iter_source_files(repo_dir: Path, include_ext: Iterable[str],
                      exclude_dirs: Iterable[str], max_bytes: int = 1_000_000):
    """Deterministic (sorted) walk over a repo's source files."""
    include_ext = tuple(include_ext)
    exclude = set(exclude_dirs)
    repo_dir = Path(repo_dir)
    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file() or path.suffix not in include_ext:
            continue
        # Match exclusions against the REPO-RELATIVE path only: matching absolute
        # parts would skip everything whenever the project itself lives under a
        # directory called build/, dist/, venv/, ...
        if any(part in exclude for part in path.relative_to(repo_dir).parts):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        yield path

#!/usr/bin/env python
"""ONE-TIME setup: download models, clone repos, fetch datasets, build derived snapshots.

This is the ONLY script allowed to touch the network. Everything lands in data/ and
is recorded in data/.state/, so re-running is a no-op and every later run is offline.

    python scripts/bootstrap.py                  # everything
    python scripts/bootstrap.py --only models    # just the weights
    python scripts/bootstrap.py --only repos datasets
    python scripts/bootstrap.py --force models   # deliberate re-download (rare)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CODERAG_ALLOW_DOWNLOAD"] = "1"      # must be set before `import shared`

import shared  # noqa: E402,F401
from shared import datasets_setup, models, repos, state  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.paths import paths  # noqa: E402
from shared.textproc import build_stripped_snapshot  # noqa: E402


def do_models(cfg, force: bool) -> None:
    print("\n=== models ===")
    for key in ("embedding", "generation"):
        m = cfg.get_path(f"models.{key}")
        models.download_model(m["repo_id"], m.get("revision", "main"), force=force)


def do_repos(cfg, force: bool) -> None:
    print("\n=== repos ===")
    for leg in ("leg1", "leg2"):
        for spec in cfg["repos"].get(leg, []):
            repos.ensure_repo(spec, force=force)


def do_datasets(cfg, force: bool) -> None:
    print("\n=== datasets ===")
    datasets_setup.ensure_all(cfg["datasets"], force=force)


def do_derived(cfg, force: bool) -> None:
    """Comment/docstring-stripped snapshots for the RQ4 condition."""
    print("\n=== derived snapshots (comment/docstring-stripped) ===")
    ing = cfg.get_path("ingestion", {})
    for spec in cfg["repos"].get("leg2", []):
        repo_dir = paths.repo(spec["name"])
        if not repo_dir.exists():
            print(f"  [warn] {spec['name']} not cloned yet - skipping strip")
            continue
        build_stripped_snapshot(repo_dir, spec["name"], spec.get("include_ext", [".py"]),
                                ing.get("exclude_dirs", []), force=force)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-time asset bootstrap (global, all systems)")
    stages = ["models", "repos", "datasets", "derived"]
    ap.add_argument("--only", nargs="*", choices=stages, help="run only these stages")
    ap.add_argument("--force", nargs="*", choices=stages, default=[],
                    help="force re-download of these stages")
    args = ap.parse_args()

    cfg = load_config("system_a")
    fns = {"models": do_models, "repos": do_repos,
           "datasets": do_datasets, "derived": do_derived}
    for s in (args.only or stages):
        fns[s](cfg, force=s in args.force)

    print("\n=== state ===")
    print(json.dumps({k: v.get("completed_at") for k, v in state.all_state().items()}, indent=2))
    print(f"\nPins written to {paths.pins_lock}")
    print("Bootstrap complete. All later runs are OFFLINE by default.")


if __name__ == "__main__":
    main()

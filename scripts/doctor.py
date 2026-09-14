#!/usr/bin/env python
"""Health check: what is downloaded, what is indexed, what is still missing."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared  # noqa: E402,F401
from shared import state  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.env import offline  # noqa: E402
from shared.logging_utils import dir_size_mb  # noqa: E402
from shared.paths import paths  # noqa: E402

OK, NO = "  OK   ", "MISSING"


def main() -> None:
    cfg = load_config("system_a")
    print(f"project root : {paths.root}")
    print(f"offline mode : {offline()}  (CODERAG_ALLOW_DOWNLOAD=1 re-enables downloads)")
    print(f"data dir     : {paths.data}  ({dir_size_mb(paths.data)} MB)\n")

    print("--- models ---")
    for key in ("embedding", "generation"):
        rid = cfg.get_path(f"models.{key}.repo_id")
        st = state.read(f"model:{rid}")
        print(f"[{OK if st else NO}] {key:10s} {rid}"
              + (f"  @ {str(st.get('commit', ''))[:12]}" if st else ""))

    print("\n--- repos ---")
    for leg in ("leg1", "leg2"):
        for spec in cfg["repos"].get(leg, []):
            st = state.read(f"repo:{spec['name']}")
            print(f"[{OK if st else NO}] {leg} {spec['name']:14s}"
                  + (f"  @ {str(st.get('commit', ''))[:10]}" if st else ""))

    print("\n--- datasets ---")
    for name in cfg["datasets"]:
        st = state.read(f"dataset:{name}")
        print(f"[{OK if st else NO}] {name}")

    print("\n--- system A indexes ---")
    for leg in ("leg1", "leg2"):
        for spec in cfg["repos"].get(leg, []):
            st = state.read(f"index:system_a:{spec['name']}")
            if st:
                print(f"[{OK}] {spec['name']:14s} {st['chunks']:>7,} chunks  "
                      f"{st['index_size_mb']:>8} MB  {st['indexing_wall_clock_s']}s")
            else:
                print(f"[{NO}] {spec['name']:14s} -> python -m systems.system_a.index_build "
                      f"--repo {spec['name']}")

    print("\n--- results ---")
    runs = [p for p in sorted(paths.results.glob("*/*/*")) if p.is_dir()]
    for p in runs[-10:]:
        print(f"  {p.relative_to(paths.results)}")
    if not runs:
        print("  (none yet)")


if __name__ == "__main__":
    main()

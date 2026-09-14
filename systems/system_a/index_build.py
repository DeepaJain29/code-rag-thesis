"""Build one FAISS index per repo (never mixed across legs - spec section 4.4).

Indexes persist under data/indexes/system_a/<repo>/ and are BUILT ONCE: re-running
skips any repo that already has an index unless --force is passed.

    python -m systems.system_a.index_build --leg all
    python -m systems.system_a.index_build --repo shopizer --force
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List

import shared  # noqa: F401  (configures caches before ML imports)
from shared import state
from shared.config import load_config
from shared.logging_utils import dir_size_mb
from shared.models import embedder_from_config
from shared.paths import paths
from shared.repos import ensure_repo

from .chunking import chunk_repo

SYSTEM = "system_a"


def build_index_for_repo(cfg, repo_spec: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    name = repo_spec["name"]
    out_dir = paths.index_dir(SYSTEM, name)
    key = f"index:{SYSTEM}:{name}"

    if force:
        state.clear(key)
    if state.is_done(key, check_path=out_dir / "index.faiss"):
        print(f"[index] {name}: already built ({out_dir}) - skipping")
        return state.read(key)

    repo_dir = ensure_repo(repo_spec)          # no-op if already cloned
    ing = cfg.get_path("ingestion", {})
    ch = cfg.get_path("chunking")

    print(f"[index] {name}: chunking...")
    t0 = time.perf_counter()
    docs, stats = chunk_repo(name, repo_dir,
                             include_ext=repo_spec.get("include_ext", [".java"]),
                             exclude_dirs=ing.get("exclude_dirs", []),
                             chunk_size=ch["chunk_size"],
                             chunk_overlap=ch["chunk_overlap"],
                             max_file_bytes=ing.get("max_file_bytes", 1_000_000))
    t_chunk = time.perf_counter() - t0
    print(f"[index] {name}: {stats.n_files} files -> {stats.n_chunks} chunks ({t_chunk:.1f}s)")

    embedder = embedder_from_config(cfg)
    texts = [d.page_content for d in docs]
    embed_tokens = embedder.count_tokens(texts)
    print(f"[index] {name}: embedding {len(texts)} chunks "
          f"({embed_tokens:,} tokens) with {embedder.repo_id} ...")

    t1 = time.perf_counter()
    vs = FAISS.from_documents(docs, embedder,
                              distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                              normalize_L2=True)     # cosine
    t_embed = time.perf_counter() - t1

    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))

    meta = {"system": SYSTEM, "repo": name,
            "commit": state.load_pins().get(f"repo:{name}", {}).get("commit"),
            "files": stats.n_files, "chunks": stats.n_chunks, "chars": stats.n_chars,
            "paper_files": repo_spec.get("paper_files"),
            "paper_chunks": repo_spec.get("paper_chunks"),
            "chunk_size": ch["chunk_size"], "chunk_overlap": ch["chunk_overlap"],
            "embedding_model": embedder.repo_id, "embedding_dim": embedder.dim,
            "indexing_chunk_time_s": round(t_chunk, 2),
            "indexing_embed_time_s": round(t_embed, 2),
            "indexing_wall_clock_s": round(t_chunk + t_embed, 2),
            "indexing_embed_tokens": embed_tokens,
            "index_size_mb": dir_size_mb(out_dir), "index_path": str(out_dir)}
    (out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    state.mark_done(key, **meta)
    print(f"[index] {name}: done in {meta['indexing_wall_clock_s']}s, "
          f"{meta['index_size_mb']} MB")
    return meta


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build System A FAISS indexes")
    ap.add_argument("--leg", choices=["leg1", "leg2", "all"], default="all")
    ap.add_argument("--repo", help="build only this repo")
    ap.add_argument("--force", action="store_true", help="rebuild even if present")
    args = ap.parse_args(argv)

    cfg = load_config("system_a")
    legs = ["leg1", "leg2"] if args.leg == "all" else [args.leg]

    all_meta = []
    for leg in legs:
        for spec in cfg["repos"].get(leg, []):
            if args.repo and spec["name"] != args.repo:
                continue
            all_meta.append(build_index_for_repo(cfg, spec, force=args.force))

    out = paths.results_dir(SYSTEM) / "index_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    print(f"\n[index] summary -> {out}")


if __name__ == "__main__":
    main()

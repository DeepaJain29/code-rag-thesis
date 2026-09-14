#!/usr/bin/env python
"""Build a CrossCodeEval-style held-out query set for a Python repo (Leg 2).

Method (spec section 3, Leg 2):
  1. Parse every file with `ast`; collect module-level function/class definitions.
  2. Find call sites whose callee is defined in a DIFFERENT file - this is the static
     verification that the held-out line genuinely needs cross-file context.
  3. Hold out the line containing that call; the preceding lines become the prompt.
  4. Record the defining file's chunk ids as ground-truth relevant chunks, so retrieval
     precision/recall is measured against verified relevance rather than a guess.

Output: data/derived/<repo>_heldout_queries.json - built once, reused by A, B and C.

    python scripts/build_query_set.py --repo frappe --n 200
    python scripts/build_query_set.py --repo frappe --stripped     # RQ4 condition
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared  # noqa: E402,F401
from shared.config import load_config  # noqa: E402
from shared.paths import paths  # noqa: E402
from shared.repos import iter_source_files  # noqa: E402


def collect_definitions(files, repo_dir):
    """-> (name -> {relative file paths defining it}, rel -> (source, tree))"""
    defs = defaultdict(set)
    trees = {}
    for f in files:
        try:
            src = f.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError):
            continue
        rel = str(f.relative_to(repo_dir))
        trees[rel] = (src, tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs[node.name].add(rel)
    return defs, trees


def chunk_ids_for_file(repo: str, rel: str, src_len: int, size: int, overlap: int):
    """Mirror of the chunker's id scheme so ground truth matches retrieved ids."""
    step = max(size - overlap, 1)
    n = max(1, (max(src_len - overlap, 0) + step - 1) // step)
    return [f"{repo}::{rel}::{i}" for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="repo name as in configs/global.yaml")
    ap.add_argument("--n", type=int, default=200, help="number of held-out queries to keep")
    ap.add_argument("--context-lines", type=int, default=25)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--stripped", action="store_true",
                    help="build from the comment-stripped snapshot (RQ4 condition)")
    args = ap.parse_args()

    cfg = load_config("system_a")
    size = cfg.get_path("chunking.chunk_size")
    overlap = cfg.get_path("chunking.chunk_overlap")
    ing = cfg.get_path("ingestion", {})

    repo_dir = (paths.derived / f"{args.repo}-stripped") if args.stripped else paths.repo(args.repo)
    if not repo_dir.exists():
        raise SystemExit(f"{repo_dir} not found - run scripts/bootstrap.py first")

    files = list(iter_source_files(repo_dir, [".py"], ing.get("exclude_dirs", [])))
    print(f"[qset] {len(files)} python files in {repo_dir}")
    defs, trees = collect_definitions(files, repo_dir)

    candidates = []
    for rel, (src, tree) in trees.items():
        lines = src.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if not name or name not in defs:
                continue
            owners = defs[name] - {rel}              # defined elsewhere == cross-file
            if not owners or len(defs[name]) > 3:    # skip ambiguous common names
                continue
            ln = getattr(node, "lineno", 0)
            if ln < args.context_lines + 1 or ln > len(lines):
                continue
            gold = lines[ln - 1].strip()
            if len(gold) < 12:
                continue
            prefix = "\n".join(lines[max(0, ln - 1 - args.context_lines): ln - 1])
            owner = sorted(owners)[0]
            owner_src = trees.get(owner, ("", None))[0]
            candidates.append({
                "id": f"{args.repo}-{len(candidates):05d}",
                "file": rel, "line": ln, "callee": name, "defined_in": owner,
                "query": (f"File: {rel}\nComplete the next line of code.\n\n"
                          f"```python\n{prefix}\n```"),
                "gold": gold,
                "relevant_chunk_ids": chunk_ids_for_file(args.repo, owner, len(owner_src),
                                                         size, overlap),
                "cross_file_verified": True})

    random.Random(args.seed).shuffle(candidates)
    picked = candidates[: args.n]
    out = paths.derived / f"{args.repo}_heldout_queries.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repo": args.repo, "stripped": args.stripped,
                               "seed": args.seed, "n_candidates": len(candidates),
                               "n_selected": len(picked), "queries": picked},
                              indent=2), encoding="utf-8")
    print(f"[qset] {len(candidates)} verified cross-file candidates -> kept {len(picked)} -> {out}")


if __name__ == "__main__":
    main()

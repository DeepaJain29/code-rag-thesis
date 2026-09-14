"""Leg 2 - generation evaluation (this project's RQ1).

    python -m systems.system_a.run_leg2 --dataset codereval --index-repo frappe
    python -m systems.system_a.run_leg2 --dataset repoexec  --index-repo frappe
    python -m systems.system_a.run_leg2 --dataset frappe            # EM/ES + retrieval P/R
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Sequence

import shared  # noqa: F401
from shared.config import load_config
from shared.datasets_setup import codereval_tasks, repoexec_tasks
from shared.exec_eval import evaluate_samples, extract_code_block
from shared.logging_utils import RunLogger
from shared.metrics import (aggregate_pass_at_k, edit_similarity, exact_match,
                            latency_percentiles, mean_dicts, precision_recall_at_k)
from shared.paths import paths

from .pipeline import SystemA

SYSTEM = "system_a"


def _normalise_task(raw: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """Map CoderEval / RepoExec fields onto one internal shape.

    Field names vary by release - print one record and adjust here if a key is
    missing rather than silently generating from an empty prompt.
    """
    if dataset == "codereval":
        return {"id": raw.get("_id") or raw.get("question_id") or raw.get("name"),
                "prompt": raw.get("docstring") or raw.get("human_label") or raw.get("prompt", ""),
                "signature": raw.get("signature", ""),
                "test_code": raw.get("test_code", ""),
                "preamble": raw.get("file_content_context", "") or "",
                "gold": raw.get("code", ""),
                "repo": raw.get("project", "codereval")}
    return {"id": raw.get("task_id") or raw.get("_id"),
            "prompt": raw.get("prompt") or raw.get("instruction", ""),
            "signature": raw.get("function_signature", ""),
            "test_code": raw.get("test_script") or raw.get("test", ""),
            "preamble": raw.get("context", "") or "",
            "gold": raw.get("solution") or raw.get("canonical_solution", ""),
            "repo": raw.get("project_name", "repoexec")}


def _validate_tasks(tasks: List[Dict[str, Any]], dataset: str,
                    raw: Sequence[Dict[str, Any]]) -> None:
    """Stop the run if the field mapping missed.

    A wrong key name in _normalise_task produces empty prompts and empty test code, which
    would otherwise sail through and yield a meaningless pass@k of 0.0 across the board.
    Better to fail here with the actual keys printed than to report a fake number.
    """
    if not tasks:
        raise RuntimeError(f"[leg2:{dataset}] loader returned 0 tasks")

    empty_query = [t["id"] for t in tasks if not (t["prompt"] or t["signature"]).strip()]
    empty_tests = [t["id"] for t in tasks if not t["test_code"].strip()]

    if len(empty_query) > 0.2 * len(tasks) or len(empty_tests) > 0.2 * len(tasks):
        available = sorted(raw[0].keys()) if raw else []
        raise RuntimeError(
            f"[leg2:{dataset}] field mapping looks wrong: "
            f"{len(empty_query)}/{len(tasks)} tasks have no prompt or signature, "
            f"{len(empty_tests)}/{len(tasks)} have no test code.\n"
            f"Keys actually present in the first record:\n  {available}\n"
            f"Fix the mapping in _normalise_task() for dataset '{dataset}' "
            f"(see README section 7, item 3)."
        )
    if empty_query or empty_tests:
        print(f"[leg2:{dataset}] WARNING: {len(empty_query)} tasks without a query, "
              f"{len(empty_tests)} without test code - they will score as failures")


def run_executable_benchmark(cfg, dataset: str, index_repo: str, limit: int | None,
                             logger: RunLogger, timeout: int) -> Dict[str, Any]:
    raw = codereval_tasks("python") if dataset == "codereval" else repoexec_tasks()
    tasks = [_normalise_task(r, dataset) for r in raw][:limit]
    _validate_tasks(tasks, dataset, raw)
    print(f"[leg2:{dataset}] {len(tasks)} tasks, retrieving from index '{index_repo}'")

    rag = SystemA(cfg, index_repo, logger, leg=f"leg2-{dataset}")
    n_samples = int(cfg.get_path("generation.leg2_n_samples", 5))
    per_task, records = [], []

    for t in tasks:
        query = f"{t['signature']}\n\n{t['prompt']}".strip()
        ans = rag.answer(str(t["id"]), query, task="generate", n_samples=n_samples,
                         extra={"dataset": dataset})
        candidates = [extract_code_block(s) for s in ans.samples]
        outcome = evaluate_samples(candidates, t["test_code"], t["preamble"], timeout)
        per_task.append({"n_samples": outcome["n_samples"], "n_correct": outcome["n_correct"]})
        records.append({"id": t["id"], "repo": t["repo"],
                        "n_correct": outcome["n_correct"], "n_samples": outcome["n_samples"],
                        "retrieval_latency_s": ans.record.retrieval_latency_s,
                        "generation_latency_s": ans.record.generation_latency_s,
                        "prompt_tokens": ans.record.prompt_tokens,
                        "completion_tokens": ans.record.completion_tokens,
                        "first_candidate": candidates[0][:2000],
                        "errors": [d["stderr"] for d in outcome["details"]
                                   if not d["passed"]][:2]})
        print(f"  [{t['id']}] {outcome['n_correct']}/{outcome['n_samples']} passed")

    summary = {"dataset": dataset,
               **aggregate_pass_at_k(per_task, ks=(1, 3, 5)),
               "retrieval_latency": latency_percentiles(
                   [r["retrieval_latency_s"] for r in records]),
               "generation_latency": latency_percentiles(
                   [r["generation_latency_s"] for r in records]),
               "total_prompt_tokens": sum(r["prompt_tokens"] for r in records),
               "total_completion_tokens": sum(r["completion_tokens"] for r in records)}
    logger.save_json(f"leg2_{dataset}_results.json", {"records": records, "summary": summary})
    return summary


def run_heldout(cfg, repo: str, limit: int | None, logger: RunLogger) -> Dict[str, Any]:
    query_set = paths.derived / f"{repo}_heldout_queries.json"
    if not query_set.exists():
        raise FileNotFoundError(f"{query_set} missing. Build it first:\n"
                                f"  python scripts/build_query_set.py --repo {repo}")
    queries = json.loads(query_set.read_text(encoding="utf-8"))["queries"][:limit]
    print(f"[leg2:{repo}] {len(queries)} held-out completions")

    rag = SystemA(cfg, repo, logger, leg=f"leg2-{repo}")
    k = int(cfg.get_path("retrieval.top_k", 10))
    records = []

    for q in queries:
        ans = rag.answer(q["id"], q["query"], task="generate", n_samples=1,
                         extra={"heldout": True})
        pred = extract_code_block(ans.text)
        pr = precision_recall_at_k(ans.record.retrieved_chunk_ids,
                                   q.get("relevant_chunk_ids", []), k=k)
        records.append({"id": q["id"], "em": exact_match(pred, q["gold"]),
                        "es": edit_similarity(pred, q["gold"]),
                        "retrieval_precision": pr["precision"],
                        "retrieval_recall": pr["recall"], "retrieval_f1": pr["f1"],
                        "retrieval_latency_s": ans.record.retrieval_latency_s,
                        "generation_latency_s": ans.record.generation_latency_s,
                        "prompt_tokens": ans.record.prompt_tokens,
                        "completion_tokens": ans.record.completion_tokens,
                        "prediction": pred[:2000]})
        print(f"  [{q['id']}] EM={records[-1]['em']} ES={records[-1]['es']:.3f} "
              f"P={pr['precision']:.2f} R={pr['recall']:.2f}")

    n = max(len(records), 1)
    summary = {"repo": repo, "n_queries": len(records),
               "exact_match": round(sum(r["em"] for r in records) / n, 4),
               "edit_similarity": round(sum(r["es"] for r in records) / n, 4),
               "retrieval": mean_dicts([{"precision": r["retrieval_precision"],
                                         "recall": r["retrieval_recall"],
                                         "f1": r["retrieval_f1"]} for r in records]),
               "retrieval_latency": latency_percentiles(
                   [r["retrieval_latency_s"] for r in records]),
               "generation_latency": latency_percentiles(
                   [r["generation_latency_s"] for r in records]),
               "total_prompt_tokens": sum(r["prompt_tokens"] for r in records),
               "total_completion_tokens": sum(r["completion_tokens"] for r in records)}
    logger.save_json(f"leg2_{repo}_results.json", {"records": records, "summary": summary})
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="System A - Leg 2 generation evaluation")
    ap.add_argument("--dataset", required=True,
                    choices=["codereval", "repoexec", "frappe", "erpnext"])
    ap.add_argument("--index-repo", default="frappe",
                    help="which built index to retrieve from (benchmark legs only)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--run-id")
    args = ap.parse_args(argv)

    cfg = load_config("system_a")
    logger = RunLogger(SYSTEM, f"leg2-{args.dataset}", args.run_id)
    logger.write_meta(cfg, dataset=args.dataset)

    if args.dataset in ("codereval", "repoexec"):
        summary = run_executable_benchmark(cfg, args.dataset, args.index_repo,
                                           args.limit, logger, args.timeout)
    else:
        summary = run_heldout(cfg, args.dataset, args.limit, logger)

    print("\n" + json.dumps(summary, indent=2))
    print(f"[leg2] logs -> {logger.dir}")


if __name__ == "__main__":
    main()

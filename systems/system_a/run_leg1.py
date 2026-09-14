"""Leg 1 - QA replication on arXiv:2601.08773's three Java repos.

15 questions x 3 repos = 45. Questions live in eval/leg1_questions.json.
Validates the RELATIVE pattern (A vs B vs C), not absolute parity: the models
differ from the paper's Gemini stack (documented deviation, thesis doc s11).

    python -m systems.system_a.run_leg1
    python -m systems.system_a.run_leg1 --repo shopizer --limit 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import shared  # noqa: F401
from shared.config import load_config
from shared.logging_utils import RunLogger
from shared.metrics import latency_percentiles, mean_dicts, qa_rubric_score

from .pipeline import SystemA

SYSTEM = "system_a"
QUESTIONS = Path(__file__).parent / "eval" / "leg1_questions.json"


def load_questions(path: Path = QUESTIONS) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - see eval/README.md for the format")
    qs = json.loads(path.read_text(encoding="utf-8"))["questions"]
    todo = [q for q in qs if q.get("question", "").startswith("TODO")]
    if todo:
        print(f"[leg1] WARNING: {len(todo)}/{len(qs)} questions are still placeholders")
    return qs


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="System A - Leg 1 QA evaluation")
    ap.add_argument("--repo", help="run only this repo")
    ap.add_argument("--limit", type=int, help="first N questions per repo (smoke test)")
    ap.add_argument("--run-id")
    args = ap.parse_args(argv)

    cfg = load_config("system_a")
    logger = RunLogger(SYSTEM, "leg1", args.run_id)
    logger.write_meta(cfg, models={"embedding": cfg.get_path("models.embedding.repo_id"),
                                   "generation": cfg.get_path("models.generation.repo_id")})

    questions = load_questions()
    repos = [args.repo] if args.repo else sorted({q["repo"] for q in questions})
    n_samples = int(cfg.get_path("generation.leg1_n_samples", 1))
    results: List[Dict[str, Any]] = []

    for repo in repos:
        rag = SystemA(cfg, repo, logger, leg="leg1")
        qs = [q for q in questions if q["repo"] == repo][: args.limit]
        print(f"\n[leg1] {repo}: {len(qs)} questions")
        for q in qs:
            ans = rag.answer(q["id"], q["question"], task="qa", n_samples=n_samples)
            score = qa_rubric_score(ans.text, q.get("reference_answer", ""),
                                    q.get("keywords", []))
            results.append({"id": q["id"], "repo": repo, "category": q.get("category"),
                            "question": q["question"], "answer": ans.text, "scores": score,
                            "retrieval_latency_s": ans.record.retrieval_latency_s,
                            "generation_latency_s": ans.record.generation_latency_s,
                            "prompt_tokens": ans.record.prompt_tokens,
                            "completion_tokens": ans.record.completion_tokens,
                            "retrieved_files": sorted(set(ans.record.retrieved_files)),
                            "human_score": None})     # fill during manual grading
            print(f"  [{q['id']}] coverage={score['keyword_coverage']} "
                  f"({ans.record.retrieval_latency_s:.2f}s retrieve + "
                  f"{ans.record.generation_latency_s:.2f}s generate)")

    summary: Dict[str, Any] = {}
    for repo in repos:
        rs = [r for r in results if r["repo"] == repo]
        if not rs:
            continue
        summary[repo] = {
            "n_questions": len(rs),
            **mean_dicts([r["scores"] for r in rs]),
            "retrieval_latency": latency_percentiles([r["retrieval_latency_s"] for r in rs]),
            "generation_latency": latency_percentiles([r["generation_latency_s"] for r in rs]),
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in rs),
            "total_completion_tokens": sum(r["completion_tokens"] for r in rs)}
    summary["overall"] = {"n_questions": len(results),
                          **mean_dicts([r["scores"] for r in results])}

    logger.save_json("leg1_results.json", {"results": results, "summary": summary})
    print(f"\n[leg1] wrote {logger.dir}/leg1_results.json")
    print(json.dumps(summary, indent=2))
    print("\n[leg1] NOTE: the rubric score is automatic. Grade `human_score` manually "
          "(or with an LLM judge) before reporting accuracy - spec section 5.")


if __name__ == "__main__":
    main()

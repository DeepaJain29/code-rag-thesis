"""Metrics shared by Systems A, B and C.

pass@k  - unbiased estimator (Chen et al., Codex)
EM / ES - CrossCodeEval / RepoBench conventions
retrieval precision/recall@k against statically verified ground truth
latency percentiles for the cost table
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import numpy as np


# ------------------------------- pass@k ------------------------------------ #
def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def aggregate_pass_at_k(results: Sequence[Dict], ks: Iterable[int] = (1, 3, 5)) -> Dict:
    """results: [{'n_samples': int, 'n_correct': int}, ...]"""
    out: Dict[str, float] = {}
    for k in ks:
        vals = [pass_at_k(r["n_samples"], r["n_correct"], k)
                for r in results if r["n_samples"] >= k]
        out[f"pass@{k}"] = round(float(np.mean(vals)), 4) if vals else float("nan")
    out["n_tasks"] = len(results)
    return out


# --------------------------- EM / edit similarity -------------------------- #
def _norm(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip())


def exact_match(pred: str, gold: str) -> int:
    return int(_norm(pred) == _norm(gold))


def edit_similarity(pred: str, gold: str) -> float:
    a, b = _norm(pred), _norm(gold)
    if not a and not b:
        return 1.0
    try:
        from rapidfuzz.distance import Levenshtein
        dist = Levenshtein.distance(a, b)
    except ImportError:
        dist = _levenshtein(a, b)
    return round(1.0 - dist / max(len(a), len(b), 1), 4)


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --------------------------- retrieval quality ----------------------------- #
def precision_recall_at_k(retrieved: Sequence[str], relevant: Sequence[str],
                          k: int | None = None) -> Dict[str, float]:
    r = list(retrieved[:k] if k else retrieved)
    rel = set(relevant)
    if not rel:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    hits = len([x for x in r if x in rel])
    p = hits / len(r) if r else 0.0
    rec = hits / len(rel)
    f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
    return {"precision": round(p, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def mean_dicts(dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    return {k: round(float(np.nanmean([d[k] for d in dicts])), 4) for k in dicts[0]}


# ------------------------------- cost -------------------------------------- #
def latency_percentiles(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
    arr = np.asarray(latencies, dtype=float)
    return {"p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "mean": round(float(arr.mean()), 3)}


def token_cost(tokens: int, usd_per_1k: float = 0.0) -> float:
    """Open-weight local models are $0; kept so System B's paid leg uses one formula."""
    return round(tokens / 1000.0 * usd_per_1k, 6)


# --------------------------- QA scoring (Leg 1) ---------------------------- #
def qa_rubric_score(answer: str, reference: str, keywords: List[str]) -> Dict[str, float]:
    """Deterministic fallback rubric: keyword coverage + similarity to reference.

    Document which scoring path you actually used (this rubric, an LLM judge, or
    human judgement) - spec section 5, Leg 1.
    """
    a = answer.lower()
    covered = [kw for kw in keywords if kw.lower() in a]
    coverage = len(covered) / len(keywords) if keywords else float("nan")
    return {"keyword_coverage": round(coverage, 4),
            "edit_similarity_to_reference": edit_similarity(answer, reference),
            "n_keywords_hit": len(covered), "n_keywords": len(keywords)}

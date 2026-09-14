"""System A end-to-end: query -> top-k cosine retrieval -> prompt -> ONE generation call.

Systems B and C should expose this same `answer(...)` surface and only swap the retriever
underneath. Prompt template, generator, logging and scoring then stay byte-identical across
the three arms, which is what makes the comparison attributable to retrieval design
(thesis doc section 6, "shared / controlled variables").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import shared  # noqa: F401  (configures caches before any ML import)
from shared.logging_utils import QueryRecord, RunLogger
from shared.models import generator_from_config

from .prompts import SYSTEM_PROMPT, build_prompt
from .retriever import Retriever

SYSTEM = "system_a"


@dataclass
class Answer:
    text: str                                   # first sample (the reported answer)
    samples: List[str] = field(default_factory=list)   # all n samples, for pass@k
    record: Optional[QueryRecord] = None
    sources: List[str] = field(default_factory=list)   # retrieved files, rank order


class SystemA:
    """One instance per (repo) per run. The FAISS store and both models are process-wide
    singletons underneath, so constructing this for a second repo does not reload weights."""

    def __init__(self, cfg, repo: str, logger: RunLogger, leg: str = "leg1"):
        self.cfg = cfg
        self.repo = repo
        self.logger = logger
        self.leg = leg
        self.max_context_chars = int(cfg.get_path("prompt.max_context_chars", 24000))
        self.retriever = Retriever(cfg, repo)
        self.generator = generator_from_config(cfg)

    def answer(
        self,
        query_id: str,
        query: str,
        task: str = "qa",
        n_samples: int = 1,
        temperature: float | None = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Answer:
        # 1. retrieve (timed inside the retriever)
        hits = self.retriever.retrieve(query)

        # 2. assemble the prompt from exactly those chunks (spec section 4.6)
        prompt = build_prompt(query, hits.as_context(), task=task,
                              max_context_chars=self.max_context_chars)

        # 3. single generation call per query (spec section 4.7).
        #    Sampling needs temperature > 0, otherwise n samples would be identical
        #    and pass@3 / pass@5 would be meaningless.
        temp = temperature
        if temp is None and n_samples > 1:
            temp = max(float(self.cfg.get_path("models.generation.temperature", 0.2)), 0.2)
        out = self.generator.generate(SYSTEM_PROMPT, prompt,
                                      n_samples=n_samples, temperature=temp)

        try:
            embed_tokens = self.retriever.embedder.count_tokens([query])
        except Exception:
            embed_tokens = 0

        record = QueryRecord(
            run_id=self.logger.run_id,
            system=SYSTEM,
            leg=self.leg,
            repo=self.repo,
            query_id=str(query_id),
            query=query,
            retrieved_chunk_ids=hits.chunk_ids,
            retrieved_files=hits.sources,
            retrieval_scores=hits.scores,
            retrieval_latency_s=round(hits.latency_s, 4),
            generation_latency_s=round(out.latency_s, 4),
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            embed_query_tokens=embed_tokens,
            raw_output=out.text,
            samples=out.samples,
            extra={"task": task, "n_samples": n_samples, "top_k": self.retriever.k,
                   **(extra or {})},
        )
        self.logger.log_query(record)

        return Answer(text=out.text, samples=out.samples or [out.text],
                      record=record, sources=hits.sources)

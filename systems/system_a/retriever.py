"""Top-k cosine retrieval over a persisted FAISS index. No re-ranking."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from shared.models import embedder_from_config
from shared.paths import paths

SYSTEM = "system_a"
_STORES: Dict[str, object] = {}      # repo -> loaded FAISS store (loaded once per process)


@dataclass
class Retrieved:
    chunk_ids: List[str]
    sources: List[str]
    texts: List[str]
    scores: List[float]
    latency_s: float

    def as_context(self) -> List[Tuple[str, str]]:
        return list(zip(self.sources, self.texts))


class Retriever:
    def __init__(self, cfg, repo: str):
        from langchain_community.vectorstores import FAISS

        self.cfg = cfg
        self.repo = repo
        self.k = int(cfg.get_path("retrieval.top_k", 10))
        self.embedder = embedder_from_config(cfg)

        if repo not in _STORES:
            path = paths.index_dir(SYSTEM, repo)
            if not (path / "index.faiss").exists():
                raise FileNotFoundError(
                    f"No index for repo '{repo}' at {path}.\n"
                    f"Run: python -m systems.system_a.index_build --repo {repo}")
            _STORES[repo] = FAISS.load_local(str(path), self.embedder,
                                             allow_dangerous_deserialization=True)
        self.store = _STORES[repo]

    def retrieve(self, query: str, k: int | None = None) -> Retrieved:
        k = k or self.k
        t0 = time.perf_counter()
        hits = self.store.similarity_search_with_score(query, k=k)
        latency = time.perf_counter() - t0
        return Retrieved(
            chunk_ids=[d.metadata.get("chunk_id", "?") for d, _ in hits],
            sources=[d.metadata.get("source", "?") for d, _ in hits],
            texts=[d.page_content for d, _ in hits],
            scores=[float(s) for _, s in hits],
            latency_s=latency)

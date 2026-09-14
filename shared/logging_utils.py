"""Per-query JSONL logging + timers.

Spec section 4.8 makes this mandatory: every query logs retrieved chunk ids,
retrieval latency, generation latency, token counts and raw output. These logs
are the source data for the cost tables (thesis doc section 7).
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .paths import paths


@dataclass
class QueryRecord:
    run_id: str
    system: str
    leg: str
    repo: str
    query_id: str
    query: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_files: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_query_tokens: int = 0
    raw_output: str = ""
    samples: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class RunLogger:
    """Append-only JSONL writer, one directory per run."""

    def __init__(self, system: str, leg: str, run_id: str | None = None):
        self.system, self.leg = system, leg
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.dir = paths.results_dir(system) / leg / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.queries_path = self.dir / "queries.jsonl"
        self.events_path = self.dir / "events.jsonl"
        self.meta_path = self.dir / "run_meta.json"
        self._t0 = time.perf_counter()

    def write_meta(self, cfg: Dict[str, Any], **extra: Any) -> None:
        meta = {"run_id": self.run_id, "system": self.system, "leg": self.leg,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "python": platform.python_version(), "platform": platform.platform(),
                "gpu": gpu_name(), "config": cfg, **extra}
        self.meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def log_query(self, rec: QueryRecord) -> None:
        with open(self.queries_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def log_event(self, kind: str, **payload: Any) -> None:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **payload}
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return p

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True)
        return out.stdout.strip() or "cpu"
    except Exception:
        return "cpu"


def dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)

"""Single source of truth for every path in the project.

    code-rag-thesis/
      configs/                    global.yaml + per-system yaml + pins.lock.json
      shared/                     code shared by systems A, B, C
      systems/system_a|b|c/       per-system code
      data/                       GLOBAL, downloaded once, shared by all systems
        .state/                   download-completion markers (download-once guard)
        models/hf/                HuggingFace weights cache (embed + generation)
        repos/<name>/             pinned git clones
        datasets/raw/<name>/      cloned dataset repos (CoderEval, cceval)
        datasets/hf/              HuggingFace datasets cache (RepoExec)
        derived/                  stripped snapshots, built held-out query sets
        indexes/<system>/<repo>/  vector indexes, one per repo per system
      results/<system>/           run logs, metric tables, final report
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .env import DATA, ROOT


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    configs: Path = ROOT / "configs"
    data: Path = DATA
    state: Path = DATA / ".state"
    models: Path = DATA / "models"
    hf_cache: Path = DATA / "models" / "hf"
    repos: Path = DATA / "repos"
    datasets_raw: Path = DATA / "datasets" / "raw"
    datasets_hf: Path = DATA / "datasets" / "hf"
    derived: Path = DATA / "derived"
    indexes: Path = DATA / "indexes"
    results: Path = ROOT / "results"
    pins_lock: Path = ROOT / "configs" / "pins.lock.json"

    def repo(self, name: str) -> Path:
        return self.repos / name

    def index_dir(self, system: str, repo: str) -> Path:
        return self.indexes / system / repo

    def results_dir(self, system: str) -> Path:
        return self.results / system

    def ensure(self) -> "Paths":
        for p in (self.data, self.state, self.models, self.hf_cache, self.repos,
                  self.datasets_raw, self.datasets_hf, self.derived,
                  self.indexes, self.results):
            p.mkdir(parents=True, exist_ok=True)
        return self


paths = Paths().ensure()

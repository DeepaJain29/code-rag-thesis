"""YAML config loading with attribute + dotted access."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import paths


class Config(dict):
    """dict with attribute access and dotted lookup."""

    def __getattr__(self, item: str) -> Any:
        try:
            val = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(val) if isinstance(val, dict) else val

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node


def _read(path: Path) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh) or {})


def load_config(system: str | None = "system_a") -> Config:
    """Merge configs/global.yaml with configs/<system>.yaml (system wins)."""
    cfg = _read(paths.configs / "global.yaml")
    if system:
        sys_path = paths.configs / f"{system}.yaml"
        if sys_path.exists():
            cfg.update(_read(sys_path))
    return cfg

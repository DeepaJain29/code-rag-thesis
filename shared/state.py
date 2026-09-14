"""Download-once bookkeeping.

Each expensive artefact (model, repo clone, dataset, index) writes a JSON marker
into data/.state/ when it finishes. Later runs check the marker first and do
nothing if it is present and the artefact still exists on disk.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

from .paths import paths


def _marker(key: str) -> Path:
    safe = hashlib.sha1(key.encode()).hexdigest()[:10]
    return paths.state / f"{key.replace('/', '_').replace(':', '_')}.{safe}.json"


def is_done(key: str, check_path: Path | None = None) -> bool:
    m = _marker(key)
    if not m.exists():
        return False
    if check_path is not None and not Path(check_path).exists():
        m.unlink(missing_ok=True)      # artefact deleted -> marker is stale
        return False
    return True


def mark_done(key: str, **meta: Any) -> Dict[str, Any]:
    payload = {"key": key, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta}
    _marker(key).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def read(key: str) -> Dict[str, Any] | None:
    m = _marker(key)
    return json.loads(m.read_text(encoding="utf-8")) if m.exists() else None


def clear(key: str) -> None:
    _marker(key).unlink(missing_ok=True)


def all_state() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in sorted(paths.state.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out[d.get("key", f.stem)] = d
        except json.JSONDecodeError:
            continue
    return out


# --- pin lock: the exact commits actually used, so B and C index the same snapshot
def load_pins() -> Dict[str, Any]:
    if paths.pins_lock.exists():
        return json.loads(paths.pins_lock.read_text(encoding="utf-8"))
    return {}


def save_pin(name: str, **info: Any) -> None:
    pins = load_pins()
    pins[name] = info
    paths.pins_lock.write_text(json.dumps(pins, indent=2, sort_keys=True), encoding="utf-8")

"""Shared helpers for the Flower-based federated module."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Base directories reused across helpers
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]

LOG_PATH = BASE_DIR / "logs" / "fl_events.log.jsonl"


def log_fl_event(role: str, event_type: str, **payload: Any) -> None:
    """Append a single JSON event consumed by the dashboard probes."""

    event = {"role": role, "type": event_type, **payload}
    event.setdefault("ts", datetime.now().isoformat(timespec="seconds"))

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Logging issues must never block FL execution on-device.
        pass


def resolve_dataset_path(path_str: str) -> Path:
    """Return a dataset path, falling back to the ai/data directory if needed."""

    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()

    if candidate.exists():
        return candidate

    fallbacks = [
        (ROOT_DIR / path_str).resolve(),
        (ROOT_DIR / "packages" / "ai" / "data" / candidate.name).resolve(),
    ]

    for fallback in fallbacks:
        if fallback.exists():
            return fallback

    return candidate


__all__ = ["log_fl_event", "resolve_dataset_path", "LOG_PATH"]

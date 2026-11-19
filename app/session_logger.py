from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List


LOG_EVENT_FIELDS = [
    "logged_at",
    "session_id",
    "event_type",
    "mode",
    "speaker",
    "audio_source",
    "speaker_mode",
    "raw_transcript",
    "clean_transcript",
    "interpreted_question",
    "interpretation",
    "route",
    "route_confidence",
    "memory_filters",
    "retrieval_source",
    "response_type",
    "usage",
    "latency",
]


def _logs_root() -> Path:
    return Path(os.getenv("SESSION_LOG_DIR", "logs/sessions"))


def _safe_session_id(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", (session_id or "").strip())
    return safe or "default"


def _log_path(session_id: str) -> Path:
    return _logs_root() / f"{_safe_session_id(session_id)}.jsonl"


def ensure_logs_dir() -> Path:
    root = _logs_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


# Recording raw conversation content is an explicit operator opt-in; never the
# default. The application processes transcript text in memory for its features,
# but its diagnostic audit trail should not persist that text automatically.
SAFE_METADATA_FIELDS = frozenset({
    "mode",
    "audio_source",
    "speaker_mode",
    "route",
    "route_confidence",
    "response_type",
    "usage",
    "latency",
    "chunk_count",
    "char_count",
})


def _include_conversation_content() -> bool:
    return os.getenv("SESSION_LOG_INCLUDE_CONTENT", "false").strip().lower() in {
        "1", "true", "yes",
    }


def log_session_event(session_id: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    ensure_logs_dir()
    event: Dict[str, Any] = {field: None for field in LOG_EVENT_FIELDS}
    event.update({
        "logged_at": time.time(),
        "session_id": session_id,
        "event_type": event_type,
    })
    # Keep only operational metadata unless the operator has explicitly
    # enabled content logging AND obtained appropriate participant consent.
    # Do not store transcript, interpretation, retrieved document identifiers,
    # or uploaded filenames by default.
    supplied = payload or {}
    if _include_conversation_content():
        event.update(supplied)
    else:
        event.update({
            key: supplied[key]
            for key in SAFE_METADATA_FIELDS
            if key in supplied
        })

    path = _log_path(session_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    return event


def read_session_log(session_id: str) -> List[Dict[str, Any]]:
    path = _log_path(session_id)
    if not path.exists():
        return []

    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"event_type": "log_parse_error", "raw": line})
    return events


def session_log_path(session_id: str) -> Path:
    return _log_path(session_id)

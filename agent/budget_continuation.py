"""Durable guard for automatic continuation after an iteration cap."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_MAX_CONTINUATIONS = 3


def _state_path() -> Path:
    home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    path = home / "state" / "budget_continuations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def claim(session_id: str, fingerprint: str, max_continuations: int = _MAX_CONTINUATIONS) -> tuple[bool, int, str]:
    """Claim one fresh continuation; return ``(allowed, count, reason)``.

    State is durable across processes. A changed fingerprint resets the
    consecutive no-progress count; an unchanged fingerprint advances it.
    """
    path = _state_path()
    key = str(session_id or "unknown")
    with _LOCK:
        try:
            state = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            state = {}
        old = state.get(key, {}) if isinstance(state, dict) else {}
        count = int(old.get("count", 0)) if old.get("fingerprint") == fingerprint else 0
        count += 1
        allowed = count <= max(0, int(max_continuations))
        state[key] = {"fingerprint": fingerprint, "count": count, "allowed": allowed}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True) + "\n")
        os.replace(tmp, path)
        return allowed, count, "continuation_limit" if not allowed else "ok"


def clear(session_id: str) -> None:
    path = _state_path()
    key = str(session_id or "unknown")
    with _LOCK:
        try:
            state = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            state = {}
        if isinstance(state, dict) and key in state:
            state.pop(key, None)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, sort_keys=True) + "\n")
            os.replace(tmp, path)

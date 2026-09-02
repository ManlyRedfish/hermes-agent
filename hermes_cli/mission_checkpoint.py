"""Deterministic factual checkpoint emission for governed child missions."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_REQUIRED = {
    "parent_mission_id", "child_run_id", "child_outcome", "termination_reason",
    "forward_progress", "human_escalation_recommended", "observations",
    "recommended_next_action",
}
_AUTHORITY_KEYS = frozenset({
    "authority_token", "allowed_actions", "denied_actions", "scope",
    "provider", "harness", "model", "profile", "credentials", "shell",
})
_OUTCOMES = frozenset({"INCOMPLETE", "COMPLETE"})


@dataclass(frozen=True)
class CheckpointWriteResult:
    path: str
    schema_valid: bool
    factual_observations_present: bool
    recommendation_is_non_authoritative: bool


def _require_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate(payload: Mapping[str, Any]) -> None:
    missing = sorted(_REQUIRED - payload.keys())
    if missing:
        raise ValueError(f"checkpoint missing required fields: {', '.join(missing)}")
    for name in ("parent_mission_id", "child_run_id", "termination_reason"):
        _require_string(payload[name], name)
    if payload["child_outcome"] not in _OUTCOMES:
        raise ValueError("child_outcome must be INCOMPLETE or COMPLETE")
    for name in ("forward_progress", "human_escalation_recommended"):
        if type(payload[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    observations = payload["observations"]
    if not isinstance(observations, Mapping):
        raise ValueError("observations must be an object")
    facts = observations.get("verified_facts")
    unresolved = observations.get("unresolved_gates")
    if not isinstance(facts, list) or not facts or not all(isinstance(item, str) and item.strip() for item in facts):
        raise ValueError("observations.verified_facts must be a non-empty string list")
    if not isinstance(unresolved, list) or not all(isinstance(item, str) for item in unresolved):
        raise ValueError("observations.unresolved_gates must be a string list")
    recommendation = payload["recommended_next_action"]
    if not isinstance(recommendation, Mapping):
        raise ValueError("recommended_next_action must be an object")
    if set(recommendation) & _AUTHORITY_KEYS:
        raise ValueError("recommendation contains authority fields; AI Country must decide authority")
    _require_string(recommendation.get("suggested_capability"), "recommended_next_action.suggested_capability")
    _require_string(recommendation.get("intent"), "recommended_next_action.intent")


def write_checkpoint(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> CheckpointWriteResult:
    """Validate and atomically write one factual child checkpoint.

    This function emits evidence and a recommendation only. It does not select
    a provider, mint authority, dispatch work, or evaluate continuation policy.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must be an object")
    _validate(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), sort_keys=True, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return CheckpointWriteResult(
        path=str(target), schema_valid=True, factual_observations_present=True,
        recommendation_is_non_authoritative=True,
    )

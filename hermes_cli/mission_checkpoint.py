"""Deterministic factual checkpoint emission for governed child missions."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_SCHEMA_VERSION = "1.0"
_REQUIRED = {
    "schema_version", "parent_mission_id", "child_run_id", "child_outcome", "termination_reason",
    "forward_progress", "human_escalation_recommended", "observations", "next_gap",
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
    already_present: bool = False


def _require_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate(payload: Mapping[str, Any]) -> None:
    missing = sorted(_REQUIRED - payload.keys())
    if missing:
        raise ValueError(f"checkpoint missing required fields: {', '.join(missing)}")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
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
    next_gap = payload["next_gap"]
    if not isinstance(next_gap, Mapping):
        raise ValueError("next_gap must be an object")
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
    if target.exists():
        try:
            if target.read_text(encoding="utf-8") == serialized:
                return CheckpointWriteResult(
                    path=str(target), schema_valid=True, factual_observations_present=True,
                    recommendation_is_non_authoritative=True, already_present=True,
                )
        except OSError:
            pass
        raise ValueError("checkpoint path already contains a conflicting checkpoint")
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


def emit_lifecycle_checkpoint(
    agent: Any,
    *,
    completed: bool,
    failed: bool,
    interrupted: bool,
    final_response: Any,
    messages: list,
    api_call_count: int,
    max_iterations: int,
    turn_exit_reason: str,
) -> CheckpointWriteResult | None:
    """Emit a mission checkpoint when this agent is a governed child run.

    Mission identity and destination are explicit runtime inputs. Ordinary
    interactive sessions are unchanged because absent identity/path means no-op.
    This adapter reports lifecycle facts only; AI Country owns authority.
    """
    import os

    parent_id = str(getattr(agent, "parent_mission_id", "") or os.getenv("HERMES_PARENT_MISSION_ID", "")).strip()
    child_id = str(getattr(agent, "child_run_id", "") or os.getenv("HERMES_CHILD_RUN_ID", "")).strip()
    output_path = str(getattr(agent, "mission_checkpoint_path", "") or os.getenv("HERMES_MISSION_CHECKPOINT_PATH", "")).strip()
    if not parent_id or not child_id or not output_path:
        return None
    budget = (
        str(turn_exit_reason).startswith("max_iterations_reached(")
        or "budget" in str(turn_exit_reason).lower()
        or api_call_count >= max_iterations
    )
    if budget:
        outcome, reason = "INCOMPLETE", "ITERATION_BUDGET_EXHAUSTED"
    elif completed and not failed and not interrupted:
        outcome, reason = "COMPLETE", "OBJECTIVE_MET"
    else:
        outcome, reason = "INCOMPLETE", "RECOVERABLE_INCOMPLETE"
    facts = [
        f"Hermes lifecycle exited with {str(turn_exit_reason)}",
        f"Hermes child emitted final response: {bool(final_response)}",
        f"Hermes observed {len(messages)} transcript messages",
    ]
    next_gap = getattr(agent, "checkpoint_next_gap", None)
    if not isinstance(next_gap, dict):
        next_gap = {}
    capability = str(getattr(agent, "checkpoint_suggested_capability", "none") or "none")
    intent = str(getattr(agent, "checkpoint_recommendation_intent", "No continuation recommended.") or "No continuation recommended.")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "parent_mission_id": parent_id,
        "child_run_id": child_id,
        "child_outcome": outcome,
        "termination_reason": reason,
        "forward_progress": bool(final_response or messages),
        "human_escalation_recommended": bool(getattr(agent, "checkpoint_human_escalation_recommended", False)),
        "observations": {"verified_facts": facts, "unresolved_gates": list(getattr(agent, "checkpoint_unresolved_gates", []) or [])},
        "next_gap": next_gap,
        "recommended_next_action": {"suggested_capability": capability, "intent": intent},
    }
    return write_checkpoint(output_path, payload)

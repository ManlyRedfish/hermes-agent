"""Credential-free real Hermes child used by the Phase 3 Dagu proof.

This is a subprocess entrypoint, not a finalizer unit test: it constructs the
production AIAgent and calls run_conversation. The upstream completion client
is replaced in memory with deterministic responses, so no provider/network is
needed. The script writes only the lifecycle checkpoint and a bounded JSON
summary to stdout.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def response(content: str = "Done", finish_reason: str = "stop", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="phase3/deterministic", usage=None)


def tool_call():
    return SimpleNamespace(
        id="phase3-call-budget",
        type="function",
        function=SimpleNamespace(name="synthetic_tool", arguments="{}"),
    )


def build_agent(checkpoint: Path, parent: str, child: str, max_iterations: int) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="",
            base_url="http://127.0.0.1:9",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are a deterministic Phase 3 test child."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.parent_mission_id = parent
    agent.child_run_id = child
    agent.mission_checkpoint_path = str(checkpoint)
    agent.checkpoint_suggested_capability = "repo.repair_and_verify"
    agent.checkpoint_recommendation_intent = "Run the bounded synthetic repair."
    agent.checkpoint_next_gap = {"type": "LOCALIZED_CODE_REPAIR"}
    agent.checkpoint_unresolved_gates = ["synthetic verification"]
    return agent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--child", required=True)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.delay_seconds > 0:
        time.sleep(args.delay_seconds)
    agent = build_agent(args.checkpoint, args.parent, args.child, args.max_iterations)
    if args.max_iterations == 1:
        agent.client.chat.completions.create.side_effect = [
            response(content="", finish_reason="tool_calls", tool_calls=[tool_call()]),
            response(content="Budget summary"),
        ]
    else:
        agent.client.chat.completions.create.return_value = response()
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("complete the bounded Phase 3 child proof")
    payload = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    print(json.dumps({
        "real_aiagent": isinstance(agent, AIAgent),
        "real_run_conversation": True,
        "checkpoint_path": str(args.checkpoint),
        "parent_mission_id": payload["parent_mission_id"],
        "child_run_id": payload["child_run_id"],
        "child_outcome": payload["child_outcome"],
        "termination_reason": payload["termination_reason"],
        "schema_valid": result["mission_checkpoint"]["schema_valid"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

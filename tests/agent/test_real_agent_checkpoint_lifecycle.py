import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

sys.path.insert(0, "/tmp/ai-country-lifecycle-convergence/apps/mission-api")
from mission_api.p0_continuation import SyntheticP0Supervisor


def _response(content="Done", finish_reason="stop", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _tool_call():
    return SimpleNamespace(
        id="call_budget",
        type="function",
        function=SimpleNamespace(name="synthetic_tool", arguments="{}"),
    )


def _agent(checkpoint_path: Path, *, max_iterations: int) -> AIAgent:
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
    agent._cached_system_prompt = "You are a deterministic test child."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.parent_mission_id = "msn-mempalace-governed-capability"
    agent.child_run_id = "real-agent-test"
    agent.mission_checkpoint_path = str(checkpoint_path)
    agent.checkpoint_suggested_capability = "repo.repair_and_verify"
    agent.checkpoint_recommendation_intent = "Run the bounded synthetic repair."
    agent.checkpoint_next_gap = {"type": "LOCALIZED_CODE_REPAIR"}
    agent.checkpoint_unresolved_gates = ["synthetic verification"]
    return agent


def _run(agent: AIAgent, prompt: str):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(prompt)


def _accept_checkpoint(path: Path, database: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    supervisor = SyntheticP0Supervisor(database)
    supervisor.create_parent(parent_mission_id=payload["parent_mission_id"])
    accepted = supervisor.submit_checkpoint(
        parent_mission_id=payload["parent_mission_id"],
        child_run_id=payload["child_run_id"],
        child_outcome=payload["child_outcome"],
        termination_reason=payload["termination_reason"],
        forward_progress=payload["forward_progress"],
        human_escalation_recommended=payload["human_escalation_recommended"],
        next_gap_type=payload["next_gap"].get("type"),
    )
    return supervisor, payload, accepted


class RealAIAgentCheckpointLifecycleTests(unittest.TestCase):
    def test_real_agent_normal_completion_emits_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            agent = _agent(path, max_iterations=10)
            agent.client.chat.completions.create.return_value = _response()
            result = _run(agent, "finish this synthetic child")
            supervisor, payload, accepted = _accept_checkpoint(path, Path(directory) / "p0.sqlite3")
            try:
                self.assertTrue(result["mission_checkpoint"]["schema_valid"])
                self.assertEqual("COMPLETE", payload["child_outcome"])
                self.assertEqual("OBJECTIVE_MET", payload["termination_reason"])
                self.assertEqual("COMPLETE", accepted.decision)
                self.assertEqual("completed", supervisor.parent_state(payload["parent_mission_id"]).status)
                self.assertGreaterEqual(agent.client.chat.completions.create.call_count, 1)
            finally:
                supervisor.close()

    def test_real_agent_budget_exhaustion_emits_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            agent = _agent(path, max_iterations=1)
            agent.client.chat.completions.create.side_effect = [
                _response(content="", finish_reason="tool_calls", tool_calls=[_tool_call()]),
                _response(content="Budget summary"),
            ]
            result = _run(agent, "reach the synthetic budget boundary")
            supervisor, payload, accepted = _accept_checkpoint(path, Path(directory) / "p0.sqlite3")
            try:
                self.assertTrue(result["mission_checkpoint"]["schema_valid"])
                self.assertEqual("INCOMPLETE", payload["child_outcome"])
                self.assertEqual("ITERATION_BUDGET_EXHAUSTED", payload["termination_reason"])
                self.assertEqual("CONTINUE", accepted.decision)
                self.assertEqual("repo.repair_and_verify", accepted.capability)
                self.assertEqual(0, supervisor.parent_state(payload["parent_mission_id"]).human_continuation_prompts)
                self.assertGreaterEqual(agent.client.chat.completions.create.call_count, 2)
            finally:
                supervisor.close()


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from hermes_cli.mission_checkpoint import write_checkpoint


class MissionCheckpointTests(unittest.TestCase):
    def test_writes_valid_factual_checkpoint_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            payload = {
                "parent_mission_id": "msn-mempalace-governed-capability",
                "child_run_id": "run-001",
                "child_outcome": "INCOMPLETE",
                "termination_reason": "ITERATION_BUDGET_EXHAUSTED",
                "forward_progress": True,
                "human_escalation_recommended": False,
                "observations": {
                    "verified_facts": ["focused test passed"],
                    "unresolved_gates": ["next bounded repair"],
                },
                "recommended_next_action": {
                    "suggested_capability": "repo.repair_and_verify",
                    "intent": "Run the bounded repair and verify tests.",
                },
            }
            result = write_checkpoint(path, payload)
            self.assertTrue(result.schema_valid)
            self.assertTrue(result.factual_observations_present)
            self.assertTrue(result.recommendation_is_non_authoritative)
            self.assertEqual(payload, json.loads(path.read_text(encoding="utf-8")))

    def test_rejects_recommendation_as_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            payload = {
                "parent_mission_id": "msn-mempalace-governed-capability",
                "child_run_id": "run-001",
                "child_outcome": "INCOMPLETE",
                "termination_reason": "ITERATION_BUDGET_EXHAUSTED",
                "forward_progress": True,
                "human_escalation_recommended": False,
                "observations": {"verified_facts": ["fact"], "unresolved_gates": []},
                "recommended_next_action": {
                    "suggested_capability": "repo.repair_and_verify",
                    "intent": "do work",
                    "authority_token": "must-not-be-accepted",
                },
            }
            with self.assertRaises(ValueError):
                write_checkpoint(path, payload)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

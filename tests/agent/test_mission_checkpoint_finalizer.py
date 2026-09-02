from types import SimpleNamespace
import json

from agent.turn_finalizer import finalize_turn


class _FakeAgent:
    def __init__(self, checkpoint_path, *, budget=False):
        self.parent_mission_id = "msn-mempalace-governed-capability"
        self.child_run_id = "run-finalizer"
        self.mission_checkpoint_path = str(checkpoint_path)
        self.checkpoint_suggested_capability = "repo.repair_and_verify"
        self.checkpoint_recommendation_intent = "Run the bounded repair."
        self.checkpoint_next_gap = {"type": "LOCALIZED_CODE_REPAIR"}
        self.checkpoint_unresolved_gates = ["verification pending"]
        self.checkpoint_human_escalation_recommended = False
        self.max_iterations = 1 if budget else 10
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = "http://127.0.0.1"
        self.session_id = "session-finalizer"
        self.iteration_budget = SimpleNamespace(remaining=0 if budget else 9, used=1 if budget else 1, max_total=self.max_iterations)
        self.quiet_mode = True
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0, _micro_compact_enabled=False)
        self._tool_guardrail_halt_decision = None
        self.valid_tool_names = set()
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._interrupt_message = None
        self._turn_preflight_display_snapshot = None
        self._turn_received_provider_response = False
        self._turn_failed_file_mutations = {}
        self._persist_disabled = False
        self._stream_callback = None
        self._response_was_previewed = False
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"

    def _save_trajectory(self, *args): pass
    def _cleanup_task_resources(self, *args): pass
    def _drop_trailing_empty_response_scaffolding(self, messages): pass
    def _persist_session(self, *args): pass
    def _apply_persist_user_message_override(self, messages): pass
    def _drain_pending_steer(self): return None
    def _sync_external_memory_for_turn(self, **kwargs): pass
    def _spawn_background_review(self, **kwargs): pass
    def _file_mutation_verifier_enabled(self): return False
    def _turn_completion_explainer_enabled(self): return False
    def _format_file_mutation_failure_footer(self, failed): return ""
    def _format_turn_completion_explanation(self, *args, **kwargs): return ""
    def _emit_status(self, *args): pass
    def _safe_print(self, *args): pass
    def clear_interrupt(self): pass
    def _handle_max_iterations(self, messages, api_call_count): return "Budget summary"


def _finalize(agent, *, response, calls, reason):
    return finalize_turn(
        agent,
        final_response=response,
        api_call_count=calls,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "child"}],
        conversation_history=[],
        effective_task_id=None,
        turn_id="turn-1",
        user_message="child",
        original_user_message="child",
        _should_review_memory=False,
        _turn_exit_reason=reason,
    )


def test_normal_completion_emits_checkpoint_before_finalizer_returns(tmp_path):
    path = tmp_path / "checkpoint.json"
    result = _finalize(_FakeAgent(path), response="Done", calls=1, reason="text_response(done)")
    assert result["mission_checkpoint"]["schema_valid"] is True
    assert path.exists()
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert '"child_outcome": "COMPLETE"' in path.read_text(encoding="utf-8")


def test_normal_completion_checkpoint_is_terminal_complete(tmp_path):
    path = tmp_path / "checkpoint.json"
    agent = _FakeAgent(path)
    result = _finalize(agent, response="Done", calls=1, reason="text_response(done)")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert result["mission_checkpoint"]["schema_valid"] is True
    assert payload["child_outcome"] == "COMPLETE"
    assert payload["termination_reason"] == "OBJECTIVE_MET"


def test_budget_exhaustion_emits_incomplete_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.json"
    result = _finalize(_FakeAgent(path, budget=True), response=None, calls=1, reason="budget_exhausted")
    assert result["mission_checkpoint"]["schema_valid"] is True
    text = path.read_text(encoding="utf-8")
    assert '"child_outcome": "INCOMPLETE"' in text
    assert '"termination_reason": "ITERATION_BUDGET_EXHAUSTED"' in text


def test_repeated_finalization_is_idempotent_and_does_not_conflict(tmp_path):
    path = tmp_path / "checkpoint.json"
    first = _finalize(_FakeAgent(path), response="Done", calls=1, reason="text_response(done)")
    second = _finalize(_FakeAgent(path), response="Done", calls=1, reason="text_response(done)")
    assert first["mission_checkpoint"]["already_present"] is False
    assert second["mission_checkpoint"]["already_present"] is True
    assert path.read_text(encoding="utf-8").count('"child_run_id"') == 1

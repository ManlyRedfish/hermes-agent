from types import SimpleNamespace

from agent import budget_continuation


def test_guard_allows_bounded_continuations_then_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert budget_continuation.claim("s", "checkpoint")[0]
    assert budget_continuation.claim("s", "checkpoint")[0]
    assert budget_continuation.claim("s", "checkpoint")[0]
    allowed, count, reason = budget_continuation.claim("s", "checkpoint")
    assert (allowed, count, reason) == (False, 4, "continuation_limit")


def test_guard_resets_after_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    budget_continuation.claim("s", "old")
    budget_continuation.claim("s", "old")
    assert budget_continuation.claim("s", "new") == (True, 1, "ok")

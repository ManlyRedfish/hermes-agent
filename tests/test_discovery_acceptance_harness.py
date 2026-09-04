"""Acceptance tests for the deterministic discovery-on-miss harness."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


HARNESS = Path(__file__).parents[1] / "harness" / "discovery_acceptance.py"
CANDIDATE = "/tmp/hermes-discovery-on-miss-20260904"
BASE = "/tmp/hermes-discovery-base-20260904"


def run_harness(source_root: str) -> dict:
    env = os.environ.copy()
    env["HERMES_SOURCE_ROOT"] = source_root
    env["PYTHONPATH"] = str(Path(__file__).parents[1])
    proc = subprocess.run(
        [sys.executable, str(HARNESS), "--json"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {"returncode": proc.returncode, "results": json.loads(proc.stdout)}


def test_candidate_all_seven_scenarios_pass():
    run = run_harness(CANDIDATE)
    assert run["returncode"] == 0
    assert all(item["pass"] for item in run["results"].values())


def test_base_hidden_capability_controls_fail_without_policy():
    run = run_harness(BASE)
    assert run["returncode"] != 0
    assert run["results"]["DIRECT_HIT_TEST"]["pass"]
    for name in (
        "DAGU_DISCOVERY_TEST",
        "DEFERRED_REPOSITORY_TEST",
        "ORDINARY_LANGUAGE_MCP_TEST",
        "NO_ASK_BEFORE_DISCOVERY_TEST",
    ):
        assert not run["results"][name]["pass"]


def test_harness_discriminates_repair():
    base = run_harness(BASE)["results"]
    candidate = run_harness(CANDIDATE)["results"]
    assert any(base[name]["pass"] != candidate[name]["pass"] for name in base)


def test_negative_absent_capability_remains_truthful_on_candidate():
    result = run_harness(CANDIDATE)["results"]["ABSENT_CAPABILITY_TEST"]
    assert result["pass"]
    events = [event["event"] for event in result["trace"]]
    assert "DEFERRED_DISCOVERY_REQUEST" in events
    assert "UNAVAILABLE_RESPONSE" in events
    assert "CAPABILITY_SELECTED" not in events

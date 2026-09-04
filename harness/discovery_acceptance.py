"""Deterministic discovery-on-miss acceptance harness.

The tested revision supplies the real Hermes system-prompt assembly. The model,
capability providers, and tool execution are deterministic fixtures; no network,
credentials, or live model are used.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass
class Capability:
    name: str
    description: str
    surface: str
    owner: str
    authority: str
    locality: str
    scope: str
    terms: tuple[str, ...]


@dataclass
class Trace:
    scenario: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, event: str, **data: Any) -> None:
        self.events.append({"event": event, **data})

    def index(self, event: str) -> int:
        return next((i for i, e in enumerate(self.events) if e["event"] == event), -1)


CAPABILITIES = {
    "direct": Capability("read_local_file", "Read a local file", "direct", "Hermes", "read", "local", "read-only", ("read", "file")),
    "workflow": Capability("workflow_engine", "Run a local declarative workflow", "local", "Dagu", "execute", "local", "workflow-run", ("workflow", "scheduled", "declarative")),
    "repository": Capability("remote_repository", "Inspect authoritative remote repository and CI state", "deferred", "GitHub", "read", "remote", "read-only", ("repository", "pull request", "ci", "build")),
    "mcp": Capability("fantasy_schedule", "Read fantasy league schedule and matchup data", "deferred", "MCP", "read", "remote", "read-only", ("fantasy", "matchup", "schedule", "league")),
    "absent": Capability("unused", "", "none", "", "", "", "", ()),
    "local_alt": Capability("local_shell", "Run a local shell command", "direct", "Hermes", "execute", "local", "shell", ("workflow", "scheduled")),
}


class FixtureSurface:
    def __init__(self, direct: list[Capability], deferred: list[Capability], local: list[Capability], trace: Trace):
        self.direct = direct
        self.deferred = deferred
        self.local = local
        self.trace = trace

    def direct_check(self, objective: str) -> list[Capability]:
        self.trace.add("DIRECT_TOOL_CHECK", objective=objective, tools=[c.name for c in self.direct])
        words = set(objective.lower().replace("/", " ").split())
        return [c for c in self.direct if set(c.terms) & words]

    def discover_deferred(self, objective: str) -> list[Capability]:
        self.trace.add("DEFERRED_DISCOVERY_REQUEST", objective=objective)
        words = set(objective.lower().replace("/", " ").split())
        return [c for c in self.deferred if set(c.terms) & words]

    def discover_skills(self, objective: str) -> list[Capability]:
        self.trace.add("SKILL_DISCOVERY_REQUEST", objective=objective)
        return []

    def discover_local(self, objective: str) -> list[Capability]:
        self.trace.add("LOCAL_DISCOVERY_REQUEST", objective=objective)
        words = set(objective.lower().replace("/", " ").split())
        return [c for c in self.local if set(c.terms) & words]


class ScriptedModel:
    """Deterministic policy model: response branch depends on real prompt policy."""

    def __init__(self, system_prompt: str, surface: FixtureSurface, trace: Trace):
        self.policy_enabled = "# Capability discovery on a miss" in system_prompt
        self.surface = surface
        self.trace = trace
        self.turn = 0

    def respond(self, objective: str, discovered: list[Capability], selected: Capability | None) -> str | dict[str, Any]:
        self.turn += 1
        self.trace.add("MODEL_TURN", turn=self.turn, policy_enabled=self.policy_enabled)
        direct = self.surface.direct_check(objective) if self.turn == 1 else []
        if direct:
            return {"action": "select", "capability": direct[0].name, "reason": "direct semantic fit"}
        if selected:
            return {"action": "select", "capability": selected.name, "reason": "discovery result"}
        if not self.policy_enabled:
            self.trace.add("USER_CLARIFICATION", text="Which tool should I use?")
            return "Which tool should I use?"
        return {"action": "discover"}


def build_real_system_prompt(source_root: str, valid_tools: list[str]) -> str:
    root = str(Path(source_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        system_prompt = import_module("agent.system_prompt")

    class FakeCompressor:
        context_length = None

    class FakeAgent:
        load_soul_identity = False
        skip_context_files = True
        valid_tool_names = valid_tools
        context_compressor = FakeCompressor()
        _task_completion_guidance = False
        _parallel_tool_call_guidance = False
        _tool_use_enforcement = False
        _kanban_worker_guidance = ""
        _memory_store = None
        _memory_manager = None
        model = "fixture-model"
        provider = "fixture"
        platform = "linux"
        pass_session_id = False
        session_id = "fixture"
        _environment_probe = False

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import run_agent
    run_agent.load_soul_md = lambda *_a, **_k: ""
    run_agent.build_nous_subscription_prompt = lambda *_a, **_k: ""
    run_agent.build_environment_hints = lambda *_a, **_k: ""
    run_agent.build_context_files_prompt = lambda *_a, **_k: ""
    run_agent.get_toolset_for_tool = lambda _name: "fixture"
    run_agent.build_skills_system_prompt = lambda **_k: ""
    parts = system_prompt.build_system_prompt_parts(FakeAgent())
    return "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)


def choose(candidates: list[Capability], objective: str = "") -> Capability | None:
    if not candidates:
        return None
    words = set(objective.lower().replace("/", " ").split())

    def score(c: Capability) -> tuple[int, int, int, int, int]:
        semantic = len(words.intersection(c.terms))
        return (
            semantic,
            int(c.authority == "read"),
            int(c.locality == "local"),
            int(c.scope == "read-only"),
            -len(c.name),
        )

    return max(candidates, key=score)


def run_scenario(source_root: str, name: str, objective: str, direct: list[Capability], deferred: list[Capability], local: list[Capability]) -> Trace:
    trace = Trace(name)
    prompt = build_real_system_prompt(source_root, ["terminal", "tool_search", "tool_describe", "tool_call", "skills_list", "skill_view"])
    surface = FixtureSurface(direct, deferred, local, trace)
    model = ScriptedModel(prompt, surface, trace)
    discovered: list[Capability] = []
    selected: Capability | None = None
    for _ in range(5):
        response = model.respond(objective, discovered, selected)
        if isinstance(response, str):
            trace.add("FINAL_RESPONSE", text=response)
            break
        if response["action"] == "select":
            selected = next((c for c in discovered + direct if c.name == response["capability"]), None)
            trace.add("CAPABILITY_SELECTED", capability=selected.name if selected else None, reason=response["reason"])
            if selected:
                trace.add("TOOL_CALL", capability=selected.name)
                trace.add("TOOL_RESULT", capability=selected.name, result="fixture-success")
                trace.add("FINAL_RESPONSE", text=f"Selected {selected.name}")
            else:
                trace.add("UNAVAILABLE_RESPONSE", text="Capability unavailable")
            break
        if response["action"] == "discover":
            found = surface.discover_deferred(objective)
            found += surface.discover_skills(objective)
            found += surface.discover_local(objective)
            discovered.extend(found)
            trace.add("CAPABILITY_FOUND", capabilities=[c.name for c in found])
            selected = choose(found, objective)
    else:
        trace.add("UNAVAILABLE_RESPONSE", text="Capability unavailable")
    return trace


def assertions(source_root: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    a = run_scenario(source_root, "direct_hit", "read file", [CAPABILITIES["direct"]], [], [])
    results["DIRECT_HIT_TEST"] = {"pass": a.index("CAPABILITY_SELECTED") >= 0 and a.index("DEFERRED_DISCOVERY_REQUEST") < 0, "trace": a.events}
    b = run_scenario(source_root, "local_workflow", "run the scheduled declarative workflow", [], [], [CAPABILITIES["workflow"]])
    results["DAGU_DISCOVERY_TEST"] = {"pass": b.index("LOCAL_DISCOVERY_REQUEST") >= 0 and b.index("CAPABILITY_FOUND") >= 0 and b.index("USER_CLARIFICATION") < 0, "trace": b.events}
    c = run_scenario(source_root, "remote_repository", "inspect authoritative pull request build status", [], [CAPABILITIES["repository"]], [])
    results["DEFERRED_REPOSITORY_TEST"] = {"pass": c.index("DEFERRED_DISCOVERY_REQUEST") >= 0 and c.index("CAPABILITY_SELECTED") >= 0, "trace": c.events}
    d = run_scenario(source_root, "ordinary_mcp", "check my fantasy matchup schedule", [], [CAPABILITIES["mcp"]], [])
    results["ORDINARY_LANGUAGE_MCP_TEST"] = {"pass": d.index("CAPABILITY_FOUND") >= 0 and d.index("UNAVAILABLE_RESPONSE") < 0, "trace": d.events}
    e = run_scenario(source_root, "absent", "control the imaginary quantum teleporter", [], [], [])
    results["ABSENT_CAPABILITY_TEST"] = {"pass": e.index("DEFERRED_DISCOVERY_REQUEST") >= 0 and e.index("UNAVAILABLE_RESPONSE") >= 0 and e.index("CAPABILITY_SELECTED") < 0, "trace": e.events}
    f = run_scenario(source_root, "ambiguous", "run the scheduled declarative workflow", [], [CAPABILITIES["local_alt"]], [CAPABILITIES["workflow"]])
    selected = next((x.get("capability") for x in f.events if x["event"] == "CAPABILITY_SELECTED"), None)
    results["AMBIGUOUS_CAPABILITY_TEST"] = {"pass": selected == "workflow_engine", "trace": f.events}
    g = run_scenario(source_root, "no_ask_before_discovery", "inspect authoritative pull request build status", [], [CAPABILITIES["repository"]], [])
    disc, ask = g.index("DEFERRED_DISCOVERY_REQUEST"), g.index("USER_CLARIFICATION")
    results["NO_ASK_BEFORE_DISCOVERY_TEST"] = {"pass": disc >= 0 and (ask < 0 or disc < ask), "trace": g.events}
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=os.environ.get("HERMES_SOURCE_ROOT", os.getcwd()))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = assertions(args.source_root)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for name, result in results.items():
            print(f"{name}={'PASS' if result['pass'] else 'FAIL'}")
    return 0 if all(r["pass"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

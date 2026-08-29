"""Structural tests for the model-free Hermes capability endpoint."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from aiohttp.test_utils import make_mocked_request

from gateway.config import PlatformConfig
from gateway.platforms import api_server
from gateway.platforms.api_server import APIServerAdapter


API_KEY = "capability-proof-key"


def _request(payload):
    request = make_mocked_request(
        "POST",
        "/v1/capability",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )

    async def json_body():
        return payload

    request.json = MagicMock(side_effect=json_body)
    return request


def _adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))


def _body(response):
    import json

    return json.loads(response.text)


def test_unknown_capability_fails_before_execution(monkeypatch):
    calls = []
    monkeypatch.setitem(api_server.CAPABILITIES, "proof.mempalace_status", lambda args: calls.append(args))
    response = asyncio.run(
        _adapter()._handle_capability(
            _request({"capability": "definitely.not.registered", "args": {}})
        )
    )

    assert response.status == 404
    assert calls == []


def test_test_echo_is_not_on_the_promoted_capability_surface():
    response = asyncio.run(
        _adapter()._handle_capability(
            _request({
                "capability": "test.echo",
                "args": {"message": "legacy proof scaffolding"},
            })
        )
    )

    assert response.status == 404
    assert _body(response) == {"error": "unknown_capability"}


def test_extra_authority_fields_fail_before_execution(monkeypatch):
    calls = []
    monkeypatch.setitem(api_server.CAPABILITIES, "proof.mempalace_status", lambda args: calls.append(args))

    forbidden_fields = (
        "tool",
        "server",
        "shell_command",
        "provider",
        "model",
        "profile",
        "toolset",
        "permission_override",
        "approval",
        "environment",
        "implementation",
        "instructions",
    )
    for field in forbidden_fields:
        payload = {
            "capability": "proof.mempalace_status",
            "args": {"message": "hello"},
            field: "attacker-controlled",
        }
        response = asyncio.run(_adapter()._handle_capability(_request(payload)))
        assert response.status == 400, field

    assert calls == []


def test_malformed_json_fails_before_execution(monkeypatch):
    calls = []
    monkeypatch.setitem(api_server.CAPABILITIES, "proof.mempalace_status", lambda args: calls.append(args))
    request = make_mocked_request(
        "POST",
        "/v1/capability",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    request.json = MagicMock(side_effect=ValueError("bad json"))

    response = asyncio.run(_adapter()._handle_capability(request))

    assert response.status == 400
    assert calls == []


def test_route_is_registered_and_uses_existing_authentication():
    adapter = _adapter()
    routes = {(method, path) for method, path, _ in adapter._http_route_table()}
    assert ("POST", "/v1/capability") in routes

    request = make_mocked_request(
        "POST",
        "/v1/capability",
        headers={"Authorization": "Bearer wrong-key"},
    )
    async def json_body():
        return {"capability": "proof.mempalace_status", "args": {"message": "hello"}}

    request.json = MagicMock(side_effect=json_body)
    response = asyncio.run(adapter._handle_capability(request))
    assert response.status == 401


def test_capability_path_does_not_construct_or_invoke_an_agent(monkeypatch):
    monkeypatch.setattr(
        api_server,
        "CAPABILITIES",
        {"proof.mempalace_status": lambda args: {"total_drawers": 1}},
    )
    forbidden = MagicMock(side_effect=AssertionError("AIAgent must not be used"))
    monkeypatch.setattr(api_server, "AIAgent", forbidden, raising=False)

    response = asyncio.run(
        _adapter()._handle_capability(
            _request({"capability": "proof.mempalace_status", "args": {}})
        )
    )

    assert response.status == 200
    assert _body(response)["model_invoked"] is False
    forbidden.assert_not_called()


def test_existing_agent_api_route_remains_present():
    routes = {(method, path) for method, path, _ in _adapter()._http_route_table()}
    assert ("POST", "/v1/chat/completions") in routes
    assert ("POST", "/v1/runs") in routes


def test_invalid_json_shape_is_not_silently_accepted(monkeypatch):
    calls = []
    monkeypatch.setitem(api_server.CAPABILITIES, "proof.mempalace_status", lambda args: calls.append(args))
    for payload in (
        [],
        {"capability": "proof.mempalace_status"},
        {"args": {"message": "hello"}},
        {"capability": "proof.mempalace_status", "args": {"message": "hello", "tool": "terminal"}},
    ):
        response = asyncio.run(_adapter()._handle_capability(_request(payload)))
        assert response.status == 400
    assert calls == []


def test_real_capability_uses_fixed_native_mcp_binding(monkeypatch):
    calls = []

    def native_handler(server_name, tool_name, timeout):
        assert (server_name, tool_name) == ("mempalace", "mempalace_status")

        def call(args):
            calls.append((server_name, tool_name, timeout, args.copy()))
            return json.dumps({
                "result": json.dumps({
                    "total_drawers": 7,
                    "wings": {"hermes": 1},
                    "rooms": {"testing": 1},
                    "backend": "fixture",
                    "vector_disabled": False,
                })
            })

        return call

    monkeypatch.setattr("tools.mcp_tool._make_tool_handler", native_handler)
    response = asyncio.run(
        _adapter()._handle_capability(
            _request({
                "capability": "proof.mempalace_status",
                "args": {},
                "correlation_id": "real-mcp-positive",
            })
        )
    )

    assert response.status == 200
    assert _body(response) == {
        "capability": "proof.mempalace_status",
        "executed_capability": "proof.mempalace_status",
        "correlation_id": "real-mcp-positive",
        "status": "completed",
        "model_invoked": False,
        "result": {
            "mcp_server": "mempalace",
            "mcp_tool": "mempalace_status",
            "total_drawers": 7,
            "wing_count": 1,
            "room_count": 1,
            "backend": "fixture",
            "vector_disabled": False,
        },
    }
    assert calls == [("mempalace", "mempalace_status", 60.0, {})]


def test_real_capability_accepts_live_payload_shape_without_vector_disabled(monkeypatch):
    """Regression test for the captured production MemPalace MCP shape.

    Reproduces the real observed CallToolResult (isError=False, a single
    TextContent block decoding to JSON, structuredContent=None) which does
    not include a ``vector_disabled`` key at all.
    """
    calls = []

    def native_handler(server_name, tool_name, timeout):
        def call(args):
            calls.append((server_name, tool_name, timeout, args.copy()))
            return json.dumps({
                "result": json.dumps({
                    "total_drawers": 7,
                    "wings": {"hermes": 1},
                    "rooms": {"testing": 1},
                    "backend": "fixture",
                })
            })

        return call

    monkeypatch.setattr("tools.mcp_tool._make_tool_handler", native_handler)
    response = asyncio.run(
        _adapter()._handle_capability(
            _request({
                "capability": "proof.mempalace_status",
                "args": {},
                "correlation_id": "real-mcp-live-shape",
            })
        )
    )

    assert response.status == 200
    assert _body(response)["result"] == {
        "mcp_server": "mempalace",
        "mcp_tool": "mempalace_status",
        "total_drawers": 7,
        "wing_count": 1,
        "room_count": 1,
        "backend": "fixture",
    }
    assert "vector_disabled" not in _body(response)["result"]
    assert calls == [("mempalace", "mempalace_status", 60.0, {})]


def test_real_capability_rejects_explicit_null_vector_disabled(monkeypatch):
    def native_handler(server_name, tool_name, timeout):
        def call(args):
            return json.dumps({
                "result": json.dumps({
                    "total_drawers": 7,
                    "wings": {"hermes": 1},
                    "rooms": {"testing": 1},
                    "backend": "fixture",
                    "vector_disabled": None,
                })
            })

        return call

    monkeypatch.setattr("tools.mcp_tool._make_tool_handler", native_handler)
    response = asyncio.run(
        _adapter()._handle_capability(
            _request({"capability": "proof.mempalace_status", "args": {}})
        )
    )

    assert response.status == 502
    assert _body(response)["error"] == "mcp_result_invalid"


def test_real_capability_rejects_wrong_type_vector_disabled(monkeypatch):
    for bad_value in ("true", 1, 0, [], {}):
        def native_handler(server_name, tool_name, timeout, bad_value=bad_value):
            def call(args):
                return json.dumps({
                    "result": json.dumps({
                        "total_drawers": 7,
                        "wings": {"hermes": 1},
                        "rooms": {"testing": 1},
                        "backend": "fixture",
                        "vector_disabled": bad_value,
                    })
                })

            return call

        monkeypatch.setattr("tools.mcp_tool._make_tool_handler", native_handler)
        response = asyncio.run(
            _adapter()._handle_capability(
                _request({"capability": "proof.mempalace_status", "args": {}})
            )
        )

        assert response.status == 502, bad_value
        assert _body(response)["error"] == "mcp_result_invalid", bad_value


def test_real_capability_rejects_bool_total_drawers(monkeypatch):
    """``bool`` is a subclass of ``int`` in Python; a bare ``isinstance(x, int)``
    check would wrongly accept ``True``/``False`` for a field that must be a
    genuine integer count."""
    for bad_value in (True, False):
        def native_handler(server_name, tool_name, timeout, bad_value=bad_value):
            def call(args):
                return json.dumps({
                    "result": json.dumps({
                        "total_drawers": bad_value,
                        "wings": {"hermes": 1},
                        "rooms": {"testing": 1},
                        "backend": "fixture",
                    })
                })

            return call

        monkeypatch.setattr("tools.mcp_tool._make_tool_handler", native_handler)
        response = asyncio.run(
            _adapter()._handle_capability(
                _request({"capability": "proof.mempalace_status", "args": {}})
            )
        )

        assert response.status == 502, bad_value
        assert _body(response)["error"] == "mcp_result_invalid", bad_value


def test_real_capability_rejects_authority_fields_and_cross_capability_args(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tools.mcp_tool._make_tool_handler",
        lambda *args: lambda payload: calls.append(payload),
    )

    payloads = [
        {
            "capability": "proof.mempalace_status",
            "args": {},
            field: "attacker-controlled",
        }
        for field in (
            "tool",
            "tool_name",
            "mcp_tool",
            "server",
            "mcp_server",
            "model",
            "provider",
            "profile",
            "toolset",
            "instructions",
            "shell",
            "permission",
            "environment",
            "credentials",
        )
    ] + [
        {
            "capability": "proof.mempalace_status",
            "args": {"message": "hello"},
        },
        {
            "capability": "proof.mempalace_status",
            "args": {"unexpected": True},
        },
    ]
    for payload in payloads:
        response = asyncio.run(_adapter()._handle_capability(_request(payload)))
        assert response.status == 400

    assert calls == []


def test_real_mcp_error_is_bounded_without_agent_fallback(monkeypatch):
    calls = []

    def native_handler(server_name, tool_name, timeout):
        def call(args):
            calls.append((server_name, tool_name, args))
            return json.dumps({"error": "MCP tool returned an error"})

        return call

    monkeypatch.setattr("tools.mcp_tool._make_tool_handler", native_handler)
    monkeypatch.setattr(
        api_server,
        "AIAgent",
        MagicMock(side_effect=AssertionError("agent fallback forbidden")),
        raising=False,
    )
    response = asyncio.run(
        _adapter()._handle_capability(
            _request({"capability": "proof.mempalace_status", "args": {}})
        )
    )

    assert response.status == 502
    assert _body(response) == {
        "capability": "proof.mempalace_status",
        "executed_capability": "proof.mempalace_status",
        "correlation_id": "",
        "status": "failed",
        "model_invoked": False,
        "error": "mcp_call_failed",
    }
    assert calls == [("mempalace", "mempalace_status", {})]

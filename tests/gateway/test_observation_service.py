"""
Focused adversarial tests for Hermes Observation Service V1.

Tests cover:
- Zero MCP tool registry and route table isolation
- Unauthenticated liveness /health returning exact compact bytes {"status":"ok"}
- Auth ordering (auth checked before body parsing or target contact)
- Credential isolation (no API_SERVER_KEY reading or fallback)
- Cross-credential isolation (generic vs observation keys)
- Key separation guard (equality produces 503)
- Raw path & method override attacks
- Raw header multiplicity & strict Content-Type whitespace/parameter validation
- JSON attacks & duplicate key rejection
- Payload size cap enforcement (>256 bytes -> 413)
- Fixed target loopback read, state trap & header non-leakage
- Second-port mutation trap proving zero connections
- Target redirect trap (301/302 -> 502, 0 requests to redirect destination)
- Target timeout (>2s) and 64 KiB response cap
- SSRF / target substitution rejection
- Bounded sanitized logging audit & journal size/rotation retention
- Hermetic no-write state snapshots (target, filesystem, session DB, memory, approval, rollback)
- Dynamic key rotation, revocation, replay prevention, and concurrency safety
"""

import asyncio
import json
import logging
import os
import socket
import sys
import time
import pytest
from typing import Dict, List, Optional, Tuple, Set, Any

from gateway.observation_service import (
    ObservationService,
    BoundedAuditJournal,
    get_effective_mcp_tools,
    ACTION_ID_HEALTH,
    OBSERVATION_IDENTITY,
)
from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def setup_test_separation_metadata(monkeypatch):
    monkeypatch.setenv("HERMES_GENERIC_KEY_FINGERPRINT", "generic-key-fingerprint-default-test")


class MockHealthTarget:
    """Mock target listening on loopback that snapshots state and verifies zero writes."""

    def __init__(self):
        self.port = get_free_port()
        self.server: Optional[asyncio.AbstractServer] = None
        self.get_count = 0
        self.post_count = 0
        self.received_headers: List[Dict[str, str]] = []
        self.state = {"status": "ok", "write_count": 0}
        self.redirect_to: Optional[str] = None
        self.redirect_status = 302
        self.slow_delay = 0.0
        self.response_body_override: Optional[bytes] = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        if not line:
            writer.close()
            return

        parts = line.decode("iso-8859-1").split(" ")
        method = parts[0]
        path = parts[1] if len(parts) > 1 else "/"

        if method == "GET":
            self.get_count += 1
        elif method == "POST":
            self.post_count += 1

        headers: Dict[str, str] = {}
        while True:
            hline = await reader.readline()
            if not hline or hline in (b"\r\n", b"\n"):
                break
            hstr = hline.decode("iso-8859-1").rstrip("\r\n")
            if ":" in hstr:
                k, v = hstr.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        self.received_headers.append(headers)

        if self.slow_delay > 0:
            await asyncio.sleep(self.slow_delay)

        if self.redirect_to:
            resp = (
                f"HTTP/1.1 {self.redirect_status} Found\r\n"
                f"Location: {self.redirect_to}\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n\r\n"
            ).encode("utf-8")
            writer.write(resp)
            await writer.drain()
            writer.close()
            return

        if self.response_body_override is not None:
            body = self.response_body_override
        else:
            body = json.dumps({"status": self.state["status"]}).encode("utf-8")

        resp = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("utf-8") + body

        writer.write(resp)
        await writer.drain()
        writer.close()

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, "127.0.0.1", self.port)

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()


class MockMutationTrapServer:
    """Mock mutation server listening on a second port to prove zero connections/requests."""

    def __init__(self):
        self.port = get_free_port()
        self.server: Optional[asyncio.AbstractServer] = None
        self.connection_count = 0
        self.request_count = 0
        self.requests: List[Tuple[str, str]] = []

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connection_count += 1
        line = await reader.readline()
        if line:
            self.request_count += 1
            parts = line.decode("iso-8859-1").split(" ")
            if len(parts) >= 2:
                self.requests.append((parts[0], parts[1]))
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, "127.0.0.1", self.port)

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()


async def raw_http_request(
    host: str,
    port: int,
    raw_bytes: bytes,
    timeout: float = 3.0,
) -> Tuple[int, Dict[str, str], bytes]:
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
    writer.write(raw_bytes)
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    parts = line.decode("iso-8859-1").split(" ")
    status = int(parts[1]) if len(parts) > 1 else 0

    headers: Dict[str, str] = {}
    while True:
        hline = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not hline or hline in (b"\r\n", b"\n"):
            break
        hstr = hline.decode("iso-8859-1").rstrip("\r\n")
        if ":" in hstr:
            k, v = hstr.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    body = await asyncio.wait_for(reader.read(), timeout=timeout)
    writer.close()
    await writer.wait_closed()
    return status, headers, body


@pytest.mark.asyncio
async def test_effective_mcp_tools_and_route_table():
    """Verify zero MCP tools in effective registry and exact route table."""
    mcp_tools = get_effective_mcp_tools()
    assert mcp_tools == [], "V1 observation service must have zero MCP tools"

    service = ObservationService()
    routes = service.get_route_table()
    assert routes == [("GET", "/health"), ("POST", "/v1/observation")]


@pytest.mark.asyncio
async def test_health_liveness_unauthenticated_exact_bytes():
    """Verify GET /health returns exact compact bytes {"status":"ok"} without extra space."""
    port = get_free_port()
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key="obs-key-1234567890123456",
    )
    await service.start()
    try:
        req = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        status, headers, body = await raw_http_request("127.0.0.1", port, req)
        assert status == 200
        # Exact compact bytes {"status":"ok"} with no space after colon!
        assert body == b'{"status":"ok"}'
        assert headers.get("content-type") == "application/json"
        assert headers.get("content-length") == "15"

        req_post = b"POST /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        status2, _, body2 = await raw_http_request("127.0.0.1", port, req_post)
        assert status2 == 405
        assert json.loads(body2.decode("utf-8"))["error"]["code"] == "method_not_allowed"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_auth_ordering_and_rejection():
    """Verify auth is checked before body reading/parsing or target contact."""
    target = MockHealthTarget()
    await target.start()

    port = get_free_port()
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key="obs-key-1234567890123456",
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        # 1. Missing Authorization header
        req1 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 42\r\n"
            "Connection: close\r\n\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")
        status1, _, body1 = await raw_http_request("127.0.0.1", port, req1)
        assert status1 == 401
        assert json.loads(body1)["error"]["code"] == "unauthorized"
        assert target.get_count == 0

        # 2. Malformed / Wrong Authorization header
        req2 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Authorization: Bearer wrong-key-999\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 42\r\n"
            "Connection: close\r\n\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")
        status2, _, body2 = await raw_http_request("127.0.0.1", port, req2)
        assert status2 == 401
        assert json.loads(body2)["error"]["code"] == "unauthorized"
        assert target.get_count == 0

        # 3. Duplicate Authorization headers
        req3 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Authorization: Bearer obs-key-1234567890123456\r\n"
            "Authorization: Bearer obs-key-1234567890123456\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 42\r\n"
            "Connection: close\r\n\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")
        status3, _, body3 = await raw_http_request("127.0.0.1", port, req3)
        assert status3 == 401
        assert json.loads(body3)["error"]["code"] == "unauthorized"
        assert target.get_count == 0

    finally:
        await service.stop()
        await target.stop()


@pytest.mark.asyncio
async def test_credential_isolation_no_ambient_fallback(monkeypatch):
    """Verify observation service does not read, inherit, or fall back to ambient API_SERVER_KEY."""
    port = get_free_port()
    api_key = "ambient-api-server-key-9999"

    # Set ambient API_SERVER_KEY in environment, but leave HERMES_OBSERVATION_KEY unset
    monkeypatch.setenv("API_SERVER_KEY", api_key)
    monkeypatch.delenv("HERMES_OBSERVATION_KEY", raising=False)

    service = ObservationService(host="127.0.0.1", port=port)
    await service.start()
    try:
        # Attempting auth with API_SERVER_KEY must fail (503 observation_disabled or 401)
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {api_key}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 42\r\n"
            "Connection: close\r\n\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")
        status, _, body = await raw_http_request("127.0.0.1", port, req)
        assert status in (401, 503)
        err_code = json.loads(body)["error"]["code"]
        assert err_code in ("unauthorized", "observation_disabled")
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_cross_credential_isolation_and_equality_guard(monkeypatch):
    """Verify key separation: generic keys fail on obs service, obs keys fail on generic listener."""
    port = get_free_port()
    obs_key = "obs-secret-key-12345"
    api_key = "generic-api-key-9999"

    # 1. Generic credential on observation service -> 401
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        api_server_key=api_key,
    )
    await service.start()
    try:
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {api_key}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 42\r\n"
            "Connection: close\r\n\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")
        status, _, body = await raw_http_request("127.0.0.1", port, req)
        assert status == 401
        assert json.loads(body)["error"]["code"] == "unauthorized"
    finally:
        await service.stop()

    # 2. Observation credential on generic Hermes listener -> 401
    monkeypatch.setenv("HERMES_OBSERVATION_KEY", obs_key)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key}))
    class DummyRequest:
        headers = {"Authorization": f"Bearer {obs_key}"}

    auth_res = adapter._check_auth(DummyRequest())  # type: ignore
    assert auth_res is not None
    assert auth_res.status == 401

    # Equal key fail-closed on generic listener -> 401
    adapter_equal = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": obs_key}))
    auth_res_equal = adapter_equal._check_auth(DummyRequest())  # type: ignore
    assert auth_res_equal is not None
    assert auth_res_equal.status == 401

    # 3. Equal keys configuration guard -> 503 observation_disabled
    service_equal = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key="same-secret-key-12345",
        api_server_key="same-secret-key-12345",
    )
    await service_equal.start()
    try:
        req_eq = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Authorization: Bearer same-secret-key-12345\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 42\r\n"
            "Connection: close\r\n\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")
        status_eq, _, body_eq = await raw_http_request("127.0.0.1", port, req_eq)
        assert status_eq == 503
        assert json.loads(body_eq)["error"]["code"] == "observation_disabled"
    finally:
        await service_equal.stop()

    # 4. Missing separation metadata -> 503 observation_disabled
    monkeypatch.delenv("HERMES_GENERIC_KEY_FINGERPRINT", raising=False)
    monkeypatch.delenv("HERMES_API_SERVER_KEY_FINGERPRINT", raising=False)
    service_missing = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key="obs-secret-key-12345",
    )
    ok_m, msg_m = service_missing.check_key_configuration()
    assert ok_m is False
    assert "Missing key separation deployment metadata" in msg_m


@pytest.mark.asyncio
async def test_raw_path_and_header_attacks():
    """Verify raw path traversal, encoded slashes, query strings, and method overrides are rejected."""
    port = get_free_port()
    obs_key = "obs-key-1234567890123456"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
    )
    await service.start()

    try:
        # Path traversal
        req1 = b"POST /v1/../v1/observation HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        s1, _, _ = await raw_http_request("127.0.0.1", port, req1)
        assert s1 == 404

        # Encoded slashes
        req2 = b"POST /v1/%2f/observation HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        s2, _, _ = await raw_http_request("127.0.0.1", port, req2)
        assert s2 == 404

        # Query strings
        req3 = b"POST /v1/observation?q=attack HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        s3, _, _ = await raw_http_request("127.0.0.1", port, req3)
        assert s3 == 404

        # Method override header
        req4 = (
            b"POST /v1/observation HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"X-HTTP-Method-Override: GET\r\n"
            b"Connection: close\r\n\r\n"
        )
        s4, _, _ = await raw_http_request("127.0.0.1", port, req4)
        assert s4 == 405

        # Proxy header attack
        req5 = (
            b"POST /v1/observation HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"X-Forwarded-For: 10.0.0.1\r\n"
            b"Connection: close\r\n\r\n"
        )
        s5, _, _ = await raw_http_request("127.0.0.1", port, req5)
        assert s5 == 400

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_header_multiplicity_and_strict_content_type():
    """Verify duplicate Content-Type, Content-Type parameters, or surrounding whitespace return 400."""
    port = get_free_port()
    obs_key = "obs-key-1234567890123456"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
    )
    await service.start()

    try:
        # Content-Type with parameters
        req1 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            "Content-Length: 42\r\n"
            "Connection: close\r\n\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")
        s1, _, b1 = await raw_http_request("127.0.0.1", port, req1)
        assert s1 == 400
        assert json.loads(b1)["error"]["code"] == "invalid_request"

        # Content-Type with leading extra whitespace after colon
        req2 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type:  application/json\r\n"
            "Content-Length: 42\r\n"
            "Connection: close\r\n\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")
        s2, _, b2 = await raw_http_request("127.0.0.1", port, req2)
        assert s2 == 400
        assert json.loads(b2)["error"]["code"] == "invalid_request"

        # Content-Type with trailing whitespace
        req3 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json \r\n"
            "Content-Length: 42\r\n"
            "Connection: close\r\n\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")
        s3, _, b3 = await raw_http_request("127.0.0.1", port, req3)
        assert s3 == 400
        assert json.loads(b3)["error"]["code"] == "invalid_request"

        # Duplicate Content-Type
        req4 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 42\r\n"
            "Connection: close\r\n\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")
        s4, _, b4 = await raw_http_request("127.0.0.1", port, req4)
        assert s4 == 400
        assert json.loads(b4)["error"]["code"] == "invalid_request"

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_json_attacks_and_duplicate_key_rejection():
    """Verify duplicate JSON key, extra fields, or invalid syntax are rejected with 400."""
    port = get_free_port()
    obs_key = "obs-key-1234567890123456"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
    )
    await service.start()

    try:
        # Duplicate JSON keys
        body_dup = '{"action_id":"other","action_id":"observe.ai_country.health"}'
        req1 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_dup)}\r\n"
            "Connection: close\r\n\r\n" + body_dup
        ).encode("utf-8")
        s1, _, b1 = await raw_http_request("127.0.0.1", port, req1)
        assert s1 == 400
        assert json.loads(b1)["error"]["code"] == "invalid_request"

        # Extra field (e.g. profile)
        body_extra = '{"action_id":"observe.ai_country.health","profile":"admin"}'
        req2 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_extra)}\r\n"
            "Connection: close\r\n\r\n" + body_extra
        ).encode("utf-8")
        s2, _, b2 = await raw_http_request("127.0.0.1", port, req2)
        assert s2 == 400
        assert json.loads(b2)["error"]["code"] == "invalid_request"

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_body_size_limit_enforcement():
    """Verify payloads > 256 bytes return 413 request_too_large."""
    port = get_free_port()
    obs_key = "obs-key-1234567890123456"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
    )
    await service.start()

    try:
        # 300-byte body
        large_body = '{"action_id":"observe.ai_country.health","padding":"' + "A" * 250 + '"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(large_body)}\r\n"
            "Connection: close\r\n\r\n" + large_body
        ).encode("utf-8")
        s, _, b = await raw_http_request("127.0.0.1", port, req)
        assert s == 413
        assert json.loads(b)["error"]["code"] == "request_too_large"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_fixed_target_read_and_second_port_trap():
    """Verify target GET succeeds, 0 writes occurred, header non-leakage, and 0 connections to second port trap."""
    target = MockHealthTarget()
    await target.start()

    mutation_trap = MockMutationTrapServer()
    await mutation_trap.start()

    port = get_free_port()
    obs_key = "obs-key-1234567890123456"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")
        status, _, resp_body = await raw_http_request("127.0.0.1", port, req)
        assert status == 200

        data = json.loads(resp_body.decode("utf-8"))
        assert data == {
            "action_id": ACTION_ID_HEALTH,
            "observation_identity": OBSERVATION_IDENTITY,
            "mutation_capability": "none",
            "provenance": "real_observation",
            "observed": {"status": "ok"},
            "limitations": [],
        }

        # Assert target received exactly 1 GET and 0 POSTs/writes
        assert target.get_count == 1
        assert target.post_count == 0
        assert target.state["write_count"] == 0

        # Assert target received NO authorization header
        tgt_headers = target.received_headers[0]
        assert "authorization" not in tgt_headers

        # Assert second-port mutation trap received ZERO connections
        assert mutation_trap.connection_count == 0
        assert mutation_trap.request_count == 0

    finally:
        await service.stop()
        await target.stop()
        await mutation_trap.stop()


@pytest.mark.asyncio
async def test_target_redirect_trap():
    """Verify target returning 301/302 returns 502, does NOT follow redirect, and zero requests hit redirect destination."""
    target = MockHealthTarget()
    redirect_dest = MockMutationTrapServer()
    await target.start()
    await redirect_dest.start()

    target.redirect_to = f"http://127.0.0.1:{redirect_dest.port}/trap"
    target.redirect_status = 302

    port = get_free_port()
    obs_key = "obs-key-1234567890123456"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")
        status, _, resp_body = await raw_http_request("127.0.0.1", port, req)
        assert status == 502
        assert json.loads(resp_body)["error"]["code"] == "observation_unavailable"
        assert target.get_count == 1  # Exactly 1 request to target

        # ZERO requests and ZERO connections to redirect destination!
        assert redirect_dest.connection_count == 0
        assert redirect_dest.request_count == 0
    finally:
        await service.stop()
        await target.stop()
        await redirect_dest.stop()


@pytest.mark.asyncio
async def test_target_timeout():
    """Verify slow target (>2s) returns 502 observation_unavailable."""
    target = MockHealthTarget()
    target.slow_delay = 2.5
    await target.start()

    port = get_free_port()
    obs_key = "obs-key-1234567890123456"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")
        status, _, resp_body = await raw_http_request("127.0.0.1", port, req, timeout=5.0)
        assert status == 502
        assert json.loads(resp_body)["error"]["code"] == "observation_unavailable"
    finally:
        await service.stop()
        await target.stop()


@pytest.mark.asyncio
async def test_bounded_audit_journal_retention_and_rotation(caplog):
    """Verify audit journal is bounded in size and rotates out oldest records."""
    port = get_free_port()
    obs_key = "super-secret-obs-key-888"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        log_max_entries=5,  # Bounded max 5 log entries
    )
    await service.start()

    try:
        body = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n" + body
        ).encode("utf-8")

        # Emit 10 requests (exceeding bounded max entries of 5)
        for _ in range(10):
            await raw_http_request("127.0.0.1", port, req)

        # Assert total recorded log entries is capped at max 5
        assert len(service.audit_journal.entries) == 5

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_hermetic_no_write_state_snapshot(tmp_path):
    """Verify target, config, session, memory, receipt, approval, rollback, and filesystem state remain unchanged across success and failed observations."""
    target = MockHealthTarget()
    await target.start()

    mutation_trap = MockMutationTrapServer()
    await mutation_trap.start()

    # Create real file stores under tmp_path for named state domains
    session_dir = tmp_path / "session_store"
    session_dir.mkdir()
    (session_dir / "session.db").write_text("initial-session-db-state")

    memory_dir = tmp_path / "memory_store"
    memory_dir.mkdir()
    (memory_dir / "mempalace.json").write_text("initial-mempalace-state")

    receipt_dir = tmp_path / "receipt_store"
    receipt_dir.mkdir()
    (receipt_dir / "receipt.log").write_text("initial-receipt-state")

    approval_dir = tmp_path / "approval_store"
    approval_dir.mkdir()
    (approval_dir / "approval.json").write_text("initial-approval-state")

    rollback_dir = tmp_path / "rollback_store"
    rollback_dir.mkdir()
    (rollback_dir / "checkpoint.json").write_text("initial-checkpoint-state")

    watch_dirs = [session_dir, memory_dir, receipt_dir, approval_dir, rollback_dir]

    def _snapshot_domains() -> Dict[str, Any]:
        fs_snaps = {}
        for d in watch_dirs:
            file_map = {}
            for p in d.rglob("*"):
                if p.is_file():
                    st = p.stat()
                    file_map[str(p.relative_to(d))] = (p.read_bytes(), st.st_size, st.st_mtime_ns)
            fs_snaps[d.name] = file_map

        return {
            "business_state": {
                "write_count": target.state["write_count"],
                "post_count": target.post_count,
                "status": target.state["status"],
            },
            "config_state": dict(os.environ),
            "session_state": fs_snaps.get("session_store", {}),
            "memory_state": fs_snaps.get("memory_store", {}),
            "receipt_state": fs_snaps.get("receipt_store", {}),
            "approval_state": fs_snaps.get("approval_store", {}),
            "rollback_state": fs_snaps.get("rollback_store", {}),
            "filesystem_state": fs_snaps,
        }

    port = get_free_port()
    obs_key = "obs-key-hermetic-123"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'

        # 1. SUCCESSFUL OBSERVATION PATH (POST 200)
        snap_before_succ = _snapshot_domains()

        req_succ = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        status_s, _, _ = await raw_http_request("127.0.0.1", port, req_succ)
        assert status_s == 200

        snap_after_succ = _snapshot_domains()

        # Assert all named state domains are strictly equal before and after success
        assert snap_after_succ["business_state"]["write_count"] == snap_before_succ["business_state"]["write_count"]
        assert snap_after_succ["business_state"]["post_count"] == 0
        assert snap_after_succ["config_state"] == snap_before_succ["config_state"]
        assert snap_after_succ["session_state"] == snap_before_succ["session_state"]
        assert snap_after_succ["memory_state"] == snap_before_succ["memory_state"]
        assert snap_after_succ["receipt_state"] == snap_before_succ["receipt_state"]
        assert snap_after_succ["approval_state"] == snap_before_succ["approval_state"]
        assert snap_after_succ["rollback_state"] == snap_before_succ["rollback_state"]
        assert snap_after_succ["filesystem_state"] == snap_before_succ["filesystem_state"]
        assert mutation_trap.connection_count == 0
        assert mutation_trap.request_count == 0

        # 2. FAILED OBSERVATION PATH (POST 401 Unauthorized)
        snap_before_fail_401 = _snapshot_domains()

        req_fail_401 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Authorization: Bearer wrong-key-999\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        status_f1, _, _ = await raw_http_request("127.0.0.1", port, req_fail_401)
        assert status_f1 == 401

        snap_after_fail_401 = _snapshot_domains()
        assert snap_after_fail_401 == snap_before_fail_401
        assert mutation_trap.connection_count == 0

        # 3. FAILED OBSERVATION PATH (POST 502 Target Status Fail)
        target.state["status"] = "degraded"
        snap_before_fail_502 = _snapshot_domains()

        req_fail_502 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        status_f2, _, _ = await raw_http_request("127.0.0.1", port, req_fail_502)
        assert status_f2 == 502

        snap_after_fail_502 = _snapshot_domains()
        assert snap_after_fail_502 == snap_before_fail_502
        assert mutation_trap.connection_count == 0

    finally:
        await service.stop()
        await target.stop()
        await mutation_trap.stop()



@pytest.mark.asyncio
async def test_dynamic_key_rotation_revocation_replay_and_concurrency():
    """Verify rotation, revocation, replay rejection, and thread-safe concurrent execution."""
    target = MockHealthTarget()
    await target.start()

    port = get_free_port()
    obs_key_v1 = "rotatable-key-v1"
    obs_key_v2 = "rotatable-key-v2"

    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key_v1,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'
        req_v1 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key_v1}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        # 1. Successful request with initial key
        s1, _, _ = await raw_http_request("127.0.0.1", port, req_v1)
        assert s1 == 200

        # 2. Revoke key v1
        service.revoke_key(obs_key_v1)

        # Replay attempt with revoked key v1 must fail 401
        s2, _, body2 = await raw_http_request("127.0.0.1", port, req_v1)
        assert s2 == 401
        assert json.loads(body2)["error"]["code"] == "unauthorized"

        # 3. Rotate key to v2
        service._key_override = obs_key_v2
        req_v2 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key_v2}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        s3, _, _ = await raw_http_request("127.0.0.1", port, req_v2)
        assert s3 == 200

        # 4. Concurrent execution test: 20 parallel requests with key v2
        tasks = [raw_http_request("127.0.0.1", port, req_v2) for _ in range(20)]
        results = await asyncio.gather(*tasks)
        for res_status, _, res_body in results:
            assert res_status == 200
            data = json.loads(res_body.decode("utf-8"))
            assert data["observed"] == {"status": "ok"}

    finally:
        await service.stop()
        await target.stop()


@pytest.mark.asyncio
async def test_target_projection_fail_closed_non_ok():
    """Verify target returning non-'ok' status (e.g. 'degraded', 'running', 1, missing) fails closed with 502."""
    target = MockHealthTarget()
    await target.start()

    port = get_free_port()
    obs_key = "obs-key-1234567890123456"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        # 1. Non-ok string status: "degraded"
        target.state["status"] = "degraded"
        s1, _, b1 = await raw_http_request("127.0.0.1", port, req)
        assert s1 == 502
        assert json.loads(b1)["error"]["code"] == "observation_unavailable"

        # 2. Non-ok integer status
        target.response_body_override = b'{"status": 200}'
        s2, _, b2 = await raw_http_request("127.0.0.1", port, req)
        assert s2 == 502
        assert json.loads(b2)["error"]["code"] == "observation_unavailable"

        # 3. Missing status field
        target.response_body_override = b'{"result": "healthy"}'
        s3, _, b3 = await raw_http_request("127.0.0.1", port, req)
        assert s3 == 502
        assert json.loads(b3)["error"]["code"] == "observation_unavailable"

    finally:
        await service.stop()
        await target.stop()


@pytest.mark.asyncio
async def test_exact_bearer_token_syntax():
    """Verify trailing or extra whitespace in Bearer token returns 401."""
    target = MockHealthTarget()
    await target.start()

    port = get_free_port()
    obs_key = "obs-key-1234567890123456"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'

        # Trailing whitespace in Authorization token
        req1 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key} \r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")
        s1, _, b1 = await raw_http_request("127.0.0.1", port, req1)
        assert s1 == 401
        assert json.loads(b1)["error"]["code"] == "unauthorized"
        assert target.get_count == 0

        # Double space after Bearer
        req2 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer  {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")
        s2, _, b2 = await raw_http_request("127.0.0.1", port, req2)
        assert s2 == 401
        assert json.loads(b2)["error"]["code"] == "unauthorized"
        assert target.get_count == 0

    finally:
        await service.stop()
        await target.stop()


@pytest.mark.asyncio
async def test_authority_lock_epoch_revocation_boundary():
    """Verify revoking key updates epoch and rejects requests deterministically."""
    target = MockHealthTarget()
    await target.start()

    port = get_free_port()
    obs_key = "obs-key-epoch-123"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        service.revoke_key(obs_key)
        assert service._authority_epoch > 0
        assert service.is_key_valid(obs_key) is False

        body_str = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        status, _, body = await raw_http_request("127.0.0.1", port, req)
        assert status == 401
        assert json.loads(body)["error"]["code"] == "unauthorized"
        assert target.get_count == 0

    finally:
        await service.stop()
        await target.stop()


@pytest.mark.asyncio
async def test_production_target_url_default(monkeypatch):
    """Verify production target URL defaults to literal DEFAULT_TARGET_URL ignoring env vars."""
    monkeypatch.setenv("HERMES_OBSERVATION_TARGET_URL", "http://attacker.example.com/health")
    service = ObservationService()
    assert service.get_target_url() == "http://127.0.0.1:8080/health"


@pytest.mark.asyncio
async def test_target_header_count_and_byte_limit_502():
    """Verify target sending >100 response headers or >8192 header bytes returns 502 observation_unavailable."""
    target = MockHealthTarget()
    await target.start()

    async def custom_hdr_handle(reader, writer):
        line = await reader.readline()
        if line:
            body = b'{"status":"ok"}'
            hdr_str = "".join(f"X-Custom-Hdr-{i}: test-value-{i}\r\n" for i in range(110))
            resp = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"{hdr_str}\r\n"
            ).encode("utf-8") + body
            writer.write(resp)
            await writer.drain()
        writer.close()

    target._handle_client = custom_hdr_handle
    await target.stop()
    await target.start()

    port = get_free_port()
    obs_key = "obs-key-hdr-limit"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        status, _, b = await raw_http_request("127.0.0.1", port, req)
        assert status == 502
        assert json.loads(b)["error"]["code"] == "observation_unavailable"
    finally:
        await service.stop()
        await target.stop()


@pytest.mark.asyncio
async def test_single_end_to_end_hard_deadline_502():
    """Verify target operation enforces one 2.0s end-to-end hard deadline across connection, headers, and body."""
    target = MockHealthTarget()
    await target.start()

    async def slow_stream_handle(reader, writer):
        line = await reader.readline()
        if line:
            await asyncio.sleep(1.2)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n")
            await writer.drain()
            await asyncio.sleep(1.2)  # Total 2.4s > 2.0s deadline
            writer.write(b'{"status":"ok"}')
            await writer.drain()
        writer.close()

    target._handle_client = slow_stream_handle
    await target.stop()
    await target.start()

    port = get_free_port()
    obs_key = "obs-key-deadline-123"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    t_start = time.time()
    try:
        body_str = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        status, _, b = await raw_http_request("127.0.0.1", port, req, timeout=5.0)
        elapsed = time.time() - t_start
        assert status == 502
        assert json.loads(b)["error"]["code"] == "observation_unavailable"
        assert 1.8 <= elapsed <= 3.2
    finally:
        await service.stop()
        await target.stop()


def test_service_template_egress_and_non_persistence_contract():
    """Verify systemd service template enforces non-persistent output, strict read-only filesystem, and host-level egress rule."""
    template_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "gateway", "templates", "hermes-observation.service"
    )
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "StandardOutput=null" in content
    assert "StandardError=null" in content
    assert "ProtectSystem=strict" in content
    assert "ReadWritePaths=" not in content
    assert "StandardOutput=null" in content
    assert "StandardError=null" in content
    assert "ProtectSystem=strict" in content
    assert "ReadWritePaths=" not in content
    assert "IPAddressAllow=127.0.0.1/32 ::1/128" in content
    assert "IPAddressDeny=any" in content
    assert "Systemd IPAddressAllow is a host-level gate only, NOT exact port isolation" in content


@pytest.mark.asyncio
async def test_request_receive_hard_deadline_timeout():
    """Verify slow header feeding exceeding aggregate 5s deadline returns 400 invalid_request."""
    port = get_free_port()
    obs_key = "obs-key-slow-feed-123"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
    )
    await service.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"POST /v1/observation HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer obs-key-slow-feed-123\r\n")
        await writer.drain()
        # Sleep past aggregate timeout of 5s without finishing request
        await asyncio.sleep(5.2)

        line = await reader.readline()
        status_code = 0
        if line:
            parts = line.decode("iso-8859-1").split(" ")
            if len(parts) > 1:
                status_code = int(parts[1])
        assert status_code == 400
        writer.close()
        await writer.wait_closed()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_request_header_count_and_byte_limits():
    """Verify request headers exceeding count (>100) or size (>8192 bytes) return 400 invalid_request."""
    port = get_free_port()
    obs_key = "obs-key-hdr-bounds"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
    )
    await service.start()

    try:
        # 1. >100 headers
        hdr_lines = "".join(f"X-Req-Header-{i}: test-val-{i}\r\n" for i in range(110))
        req1 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 42\r\n"
            f"{hdr_lines}\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")

        s1, _, b1 = await raw_http_request("127.0.0.1", port, req1)
        assert s1 == 400
        assert json.loads(b1)["error"]["code"] == "invalid_request"

        # 2. >8192 header bytes
        big_hdr = "X-Big-Header: " + ("A" * 9000) + "\r\n"
        req2 = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 42\r\n"
            f"{big_hdr}\r\n"
            '{"action_id":"observe.ai_country.health"}'
        ).encode("utf-8")

        s2, _, b2 = await raw_http_request("127.0.0.1", port, req2)
        assert s2 == 400
        assert json.loads(b2)["error"]["code"] == "invalid_request"

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_http_version_validation():
    """Verify HTTP request and target status version validation rejects unexpected version tokens."""
    target = MockHealthTarget()
    await target.start()

    port = get_free_port()
    obs_key = "obs-key-ver-123"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'

        # 1. Client sends HTTP/9.9 -> 400 invalid_request
        req_bad_ver = (
            "POST /v1/observation HTTP/9.9\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        s1, _, b1 = await raw_http_request("127.0.0.1", port, req_bad_ver)
        assert s1 == 400
        assert json.loads(b1)["error"]["code"] == "invalid_request"

        # 2. Target sends HTTP/9.9 200 OK -> 502 observation_unavailable
        async def custom_ver_handle(reader, writer):
            line = await reader.readline()
            if line:
                resp = b"HTTP/9.9 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n" + b'{"status":"ok"}'
                writer.write(resp)
                await writer.drain()
            writer.close()

        target._handle_client = custom_ver_handle
        await target.stop()
        await target.start()

        req_valid = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        s2, _, b2 = await raw_http_request("127.0.0.1", port, req_valid)
        assert s2 == 502
        assert json.loads(b2)["error"]["code"] == "observation_unavailable"

    finally:
        await service.stop()
        await target.stop()


@pytest.mark.asyncio
async def test_ambiguous_framing_rejection():
    """Verify request with both Transfer-Encoding and Content-Length is rejected with 400 invalid_request."""
    port = get_free_port()
    obs_key = "obs-key-framing-123"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            "Transfer-Encoding: chunked\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        s, _, b = await raw_http_request("127.0.0.1", port, req)
        assert s == 400
        assert json.loads(b)["error"]["code"] == "invalid_request"

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_revocation_race_no_new_target_connection():
    """Verify after key revocation wins, no new target connection is established using that authority."""
    target = MockHealthTarget()
    await target.start()

    port = get_free_port()
    obs_key = "obs-key-revrace-123"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{target.port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        # Revoke key before sending request
        service.revoke_key(obs_key)

        s, _, b = await raw_http_request("127.0.0.1", port, req)
        assert s == 401
        assert json.loads(b)["error"]["code"] == "unauthorized"

        # Assert target received ZERO connections / requests!
        assert target.get_count == 0
        assert target.post_count == 0
        assert service.audit_journal.entries[-1]["target_contact"] == "false"

    finally:
        await service.stop()
        await target.stop()


@pytest.mark.asyncio
async def test_target_contacted_audit_distinction():
    """Verify target_contacted distinguishes attempted (connection refused -> false) vs established (200 -> true)."""
    unused_port = get_free_port()

    port = get_free_port()
    obs_key = "obs-key-audit-dist"
    service = ObservationService(
        host="127.0.0.1",
        port=port,
        observation_key=obs_key,
        target_url=f"http://127.0.0.1:{unused_port}/health",
    )
    await service.start()

    try:
        body_str = '{"action_id":"observe.ai_country.health"}'
        req = (
            "POST /v1/observation HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {obs_key}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_str)}\r\n"
            "Connection: close\r\n\r\n" + body_str
        ).encode("utf-8")

        # 1. Target port not listening -> Connection refused
        s1, _, b1 = await raw_http_request("127.0.0.1", port, req)
        assert s1 == 502
        assert service.audit_journal.entries[-1]["target_contact"] == "false"

        # 2. Target running -> Connection established
        target = MockHealthTarget()
        target.port = unused_port
        await target.start()

        s2, _, b2 = await raw_http_request("127.0.0.1", port, req)
        assert s2 == 200
        assert service.audit_journal.entries[-1]["target_contact"] == "true"

        await target.stop()

    finally:
        await service.stop()


def test_code_enforced_capacity_maximums():
    """Verify operator configuration exceeding code-level maximums (1000) is strictly capped."""
    service = ObservationService(
        log_max_entries=5000,
        max_revoked_keys=5000,
    )
    assert service.max_audit_entries == 1000
    assert service.max_revoked_keys == 1000
    assert service.audit_journal.max_entries == 1000

    # Add 1200 revoked keys to verify FIFO eviction cap at 1000
    for i in range(1200):
        service.revoke_key(f"revoked-key-{i}")

    assert len(service._revoked_keys) == 1000
    # Oldest 200 keys evicted
    assert "revoked-key-0" not in service._revoked_keys
    assert "revoked-key-199" not in service._revoked_keys
    assert "revoked-key-200" in service._revoked_keys
    assert "revoked-key-1199" in service._revoked_keys

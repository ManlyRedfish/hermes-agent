"""
Hermes Observation Service V1 — Standalone Observation Listener.

Exposes:
- GET  /health          — unauthenticated liveness: 200 {"status":"ok"}
- POST /v1/observation — authenticated fixed action: observe.ai_country.health

Strictly enforces:
- Exact path & method matching (raw target validation, no path traversal/overrides/queries/fragments/encoded slashes)
- Header multiplicity & override rejection (X-HTTP-Method-Override, Forwarded, X-Forwarded-*, etc.)
- Header authentication before body reading/parsing or target contact
- Key separation (HERMES_OBSERVATION_KEY != API_SERVER_KEY)
- Body size cap (256 bytes) and Content-Type: application/json (no parameters)
- Strict UTF-8 JSON object parsing with duplicate JSON key rejection
- Bounded loopback GET to server-pinned target (no DNS, no redirects, 2.0s timeout, 64 KiB cap)
- Bounded sanitized audit log
- Zero MCP servers and zero agent runtime
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("hermes.observation_service")

# Allowed action ID in V1
ACTION_ID_HEALTH = "observe.ai_country.health"
OBSERVATION_IDENTITY = "hermes-observer"

# Fixed production target constants (literal, deployment-owned)
DEFAULT_TARGET_HOST = "127.0.0.1"
DEFAULT_TARGET_PORT = 8080
DEFAULT_TARGET_PATH = "/health"
DEFAULT_TARGET_URL = f"http://{DEFAULT_TARGET_HOST}:{DEFAULT_TARGET_PORT}{DEFAULT_TARGET_PATH}"

# Bounded limits
MAX_BODY_BYTES = 256
MAX_TARGET_RESPONSE_BYTES = 64 * 1024  # 64 KiB
MAX_TARGET_HEADER_COUNT = 100
MAX_TARGET_HEADER_BYTES = 8192  # 8 KiB
TARGET_TIMEOUT_SECONDS = 2.0

# Forbidden override & proxy headers (case-insensitive)
FORBIDDEN_HEADERS: Set[str] = {
    "x-http-method-override",
    "x-http-method",
    "x-method-override",
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-forwarded-path",
    "x-forwarded-url",
    "x-forwarded-server",
    "x-forwarded-scheme",
    "x-original-url",
    "x-rewrite-url",
}


def _json_error(code: str, message: str, status_code: int) -> Tuple[int, bytes]:
    body = json.dumps(
        {"error": {"code": code, "message": message}},
        separators=(",", ":"),
    ).encode("utf-8")
    return status_code, body


class BoundedAuditJournal:
    """Bounded, sanitized audit log journal with deterministic max capacity and rotation."""

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.entries: List[Dict[str, Any]] = []

    def record(self, action: str, outcome_code: int, target_contact: bool) -> Dict[str, Any]:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry = {
            "timestamp": ts,
            "identity": OBSERVATION_IDENTITY,
            "action": action,
            "outcome": outcome_code,
            "target_contact": "true" if target_contact else "false",
        }
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        logger.info(
            "AUDIT timestamp=%s identity=%s action=%s outcome=%d target_contact=%s",
            ts,
            OBSERVATION_IDENTITY,
            action,
            outcome_code,
            "true" if target_contact else "false",
        )
        return entry


def _parse_json_no_duplicates(text: str) -> Dict[str, Any]:
    """Parse JSON text into a dict, rejecting any duplicate keys in object token stream."""
    def _pairs_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        seen = set()
        for k, v in pairs:
            if k in seen:
                raise ValueError(f"Duplicate key in JSON object: {k}")
            seen.add(k)
        return dict(pairs)

    data = json.loads(text, object_pairs_hook=_pairs_hook)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def get_effective_mcp_tools() -> List[Any]:
    """Return effective MCP tool registry for the observation service. Must be empty in V1."""
    return []


class ObservationService:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        observation_key: Optional[str] = None,
        target_url: Optional[str] = None,
        api_server_key: Optional[str] = None,
        generic_key_fingerprint: Optional[str] = None,
        log_max_entries: Optional[int] = None,
    ):
        self.host = host or os.environ.get("HERMES_OBSERVATION_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("HERMES_OBSERVATION_PORT", "8643"))
        self._key_override = observation_key
        self._target_url_override = target_url
        if generic_key_fingerprint is not None:
            self._generic_key_fingerprint = generic_key_fingerprint.strip()
        elif api_server_key is not None:
            self._generic_key_fingerprint = hashlib.sha256(api_server_key.encode("utf-8")).hexdigest()
        else:
            env_fp = (
                os.environ.get("HERMES_GENERIC_KEY_FINGERPRINT", "")
                or os.environ.get("HERMES_API_SERVER_KEY_FINGERPRINT", "")
            ).strip()
            self._generic_key_fingerprint = env_fp if env_fp else None

        self.server: Optional[asyncio.AbstractServer] = None
        self._revoked_keys: Set[str] = set()
        self._authority_lock = asyncio.Lock()
        self._authority_epoch = 0
        max_logs = (
            log_max_entries
            if log_max_entries is not None
            else int(os.environ.get("HERMES_OBSERVATION_LOG_MAX_ENTRIES", "1000"))
        )
        self.audit_journal = BoundedAuditJournal(max_entries=max_logs)

    def get_observation_key(self) -> str:
        if self._key_override is not None:
            return self._key_override
        return os.environ.get("HERMES_OBSERVATION_KEY", "")

    def get_target_url(self) -> str:
        if self._target_url_override is not None:
            return self._target_url_override
        return DEFAULT_TARGET_URL

    def revoke_key(self, key: str) -> None:
        if key:
            self._revoked_keys.add(key)
            self._authority_epoch += 1

    def is_key_valid(self, token: str) -> bool:
        obs_key = self.get_observation_key()
        if not obs_key:
            return False
        if token in self._revoked_keys or obs_key in self._revoked_keys:
            return False
        return hmac.compare_digest(token.encode("utf-8"), obs_key.encode("utf-8"))

    def check_key_configuration(
        self, api_server_key: Optional[str] = None, generic_key_fingerprint: Optional[str] = None
    ) -> Tuple[bool, str]:
        obs_key = self.get_observation_key()
        if not obs_key:
            return False, "HERMES_OBSERVATION_KEY is not configured"

        obs_fp = hashlib.sha256(obs_key.encode("utf-8")).hexdigest()
        gen_fp = generic_key_fingerprint or self._generic_key_fingerprint
        if not gen_fp and api_server_key:
            gen_fp = hashlib.sha256(api_server_key.encode("utf-8")).hexdigest()

        if not gen_fp:
            return False, "Missing key separation deployment metadata"

        if hmac.compare_digest(obs_fp.encode("utf-8"), gen_fp.encode("utf-8")):
            return False, "HERMES_OBSERVATION_KEY equals generic key separation metadata"

        return True, "ok"

    def validate_target_url(self) -> Tuple[bool, str, int, str]:
        target = self.get_target_url()
        try:
            parsed = urlparse(target)
            if parsed.scheme != "http":
                return False, "", 0, ""
            host = parsed.hostname
            if not host or host not in ("127.0.0.1", "::1"):
                return False, "", 0, ""
            if parsed.username or parsed.password or parsed.params or parsed.query or parsed.fragment:
                return False, "", 0, ""
            port = parsed.port
            path = parsed.path or "/"
            if port is None or path != "/health":
                return False, "", 0, ""
            return True, host, port, path
        except Exception:
            return False, "", 0, ""

    def get_route_table(self) -> List[Tuple[str, str]]:
        """Return the exact registered route table."""
        return [
            ("GET", "/health"),
            ("POST", "/v1/observation"),
        ]

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        target_contacted = False
        action_name = "unknown"
        status_code = 500
        response_body = b""

        try:
            # Read request line (timeout 5s)
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            except Exception:
                status_code, response_body = _json_error("invalid_request", "Invalid request line", 400)
                return

            if not line:
                return

            try:
                request_line = line.decode("iso-8859-1").rstrip("\r\n")
                parts = request_line.split(" ")
                if len(parts) != 3:
                    status_code, response_body = _json_error("invalid_request", "Invalid HTTP request line", 400)
                    return
                method, raw_target, _http_version = parts
            except Exception:
                status_code, response_body = _json_error("invalid_request", "Invalid request line format", 400)
                return

            # Read headers line by line
            headers: List[Tuple[str, str]] = []
            raw_headers: List[Tuple[str, str, str]] = []  # (key, raw_val, stripped_val)
            while True:
                try:
                    hline = await asyncio.wait_for(reader.readline(), timeout=5.0)
                except Exception:
                    status_code, response_body = _json_error("invalid_request", "Error reading headers", 400)
                    return
                if not hline or hline in (b"\r\n", b"\n"):
                    break
                hstr = hline.decode("iso-8859-1").rstrip("\r\n")
                if ":" not in hstr:
                    status_code, response_body = _json_error("invalid_request", "Malformed header", 400)
                    return
                k, v = hstr.split(":", 1)
                headers.append((k.strip(), v.strip()))
                raw_headers.append((k.strip(), v, v.strip()))

            # Check for forbidden override/proxy headers
            for k, _ in headers:
                kl = k.lower()
                if kl in FORBIDDEN_HEADERS:
                    if kl in ("x-http-method-override", "x-http-method", "x-method-override"):
                        status_code, response_body = _json_error("method_not_allowed", "Method override forbidden", 405)
                    else:
                        status_code, response_body = _json_error("invalid_request", "Forbidden header", 400)
                    return

            # Check raw request target (must match origin-form octets exactly)
            if raw_target not in ("/health", "/v1/observation"):
                status_code, response_body = _json_error("not_found", "Not found", 404)
                return

            # Health liveness route: GET /health
            if raw_target == "/health":
                action_name = "observe.health"
                if method != "GET":
                    status_code, response_body = _json_error("method_not_allowed", "Method not allowed", 405)
                    return
                status_code = 200
                response_body = b'{"status":"ok"}'
                return

            # Observation action route: POST /v1/observation
            action_name = ACTION_ID_HEALTH
            if method != "POST":
                status_code, response_body = _json_error("method_not_allowed", "Method not allowed", 405)
                return

            # 1. AUTHENTICATION CHECK (performed before body parsing/target contact)
            raw_auth_entries = [r for r in raw_headers if r[0].lower() == "authorization"]
            if len(raw_auth_entries) != 1:
                status_code, response_body = _json_error("unauthorized", "Unauthorized", 401)
                return

            raw_val = raw_auth_entries[0][1]
            if raw_val.endswith(" ") or raw_val.endswith("\t"):
                status_code, response_body = _json_error("unauthorized", "Unauthorized", 401)
                return

            val = raw_val[1:] if raw_val.startswith(" ") else raw_val
            if not val.startswith("Bearer "):
                status_code, response_body = _json_error("unauthorized", "Unauthorized", 401)
                return

            token = val[7:]
            if not token or token != token.strip() or token.startswith(" ") or token.endswith(" ") or "\t" in token:
                status_code, response_body = _json_error("unauthorized", "Unauthorized", 401)
                return

            # Verify observation key configuration & separation
            config_ok, msg = self.check_key_configuration()
            if not config_ok:
                status_code, response_body = _json_error("observation_disabled", msg, 503)
                return

            # Verify token equality and revocation status
            async with self._authority_lock:
                if not self.is_key_valid(token):
                    status_code, response_body = _json_error("unauthorized", "Unauthorized", 401)
                    return

            # 2. Content-Type check (application/json with no parameters or surrounding whitespace)
            ct_entries = [r for r in raw_headers if r[0].lower() == "content-type"]
            if len(ct_entries) != 1:
                status_code, response_body = _json_error("invalid_request", "Invalid Content-Type header", 400)
                return

            _k, raw_v, stripped_v = ct_entries[0]
            media_type_raw = raw_v[1:] if raw_v.startswith(" ") else raw_v
            if media_type_raw != media_type_raw.strip() or stripped_v.lower() != "application/json":
                status_code, response_body = _json_error(
                    "invalid_request", "Content-Type must be application/json with no parameters or surrounding whitespace", 400
                )
                return

            # 3. Content-Length & Body size cap (enforced before buffering entire body)
            cl_headers = [v for k, v in headers if k.lower() == "content-length"]
            if len(cl_headers) != 1:
                status_code, response_body = _json_error("invalid_request", "Content-Length required", 400)
                return
            try:
                cl = int(cl_headers[0])
                if cl < 0:
                    raise ValueError
                if cl > MAX_BODY_BYTES:
                    status_code, response_body = _json_error("request_too_large", "Payload too large", 413)
                    return
            except ValueError:
                status_code, response_body = _json_error("invalid_request", "Invalid Content-Length", 400)
                return

            try:
                body_bytes = await reader.readexactly(cl)
            except asyncio.IncompleteReadError:
                status_code, response_body = _json_error("invalid_request", "Incomplete request body", 400)
                return

            if not body_bytes:
                status_code, response_body = _json_error("invalid_request", "Empty request body", 400)
                return

            # Parse JSON body with strict duplicate key rejection
            try:
                body_str = body_bytes.decode("utf-8")
                data = _parse_json_no_duplicates(body_str)
            except Exception:
                status_code, response_body = _json_error("invalid_request", "Invalid JSON payload or duplicate keys", 400)
                return

            # Exact schema rules
            if len(data) != 1 or "action_id" not in data or not isinstance(data["action_id"], str):
                status_code, response_body = _json_error("invalid_request", "Body must contain exactly action_id field", 400)
                return

            if data["action_id"] != ACTION_ID_HEALTH:
                status_code, response_body = _json_error("invalid_request", "Unsupported action", 400)
                return

            # 4. Target transport (bounded GET to literal loopback)
            valid_target, thost, tport, tpath = self.validate_target_url()
            if not valid_target:
                status_code, response_body = _json_error("observation_disabled", "Observation target is not configured", 503)
                return

            # Protect final revocation check and target connection start with authority lock/epoch
            async with self._authority_lock:
                epoch_at_start = self._authority_epoch
                if not self.is_key_valid(token):
                    status_code, response_body = _json_error("unauthorized", "Unauthorized", 401)
                    return

                target_contacted = True

            # Target operation wrapped in ONE hard end-to-end deadline of 2.0s
            async def _do_target_exchange() -> bytes:
                treader, twriter = await asyncio.open_connection(thost, tport)
                try:
                    # Validate connected peer address via getpeername before sending any request data
                    sock = twriter.get_extra_info("socket")
                    peer = sock.getpeername() if sock is not None else twriter.get_extra_info("peername")
                    if not peer or not isinstance(peer, (tuple, list)) or len(peer) < 2:
                        raise ValueError("Invalid peer address")
                    peer_ip, peer_port = str(peer[0]), int(peer[1])
                    if peer_ip not in ("127.0.0.1", "::1") or peer_ip != thost or peer_port != tport:
                        raise ValueError(f"Peer address mismatch: {peer_ip}:{peer_port}")

                    # Re-verify authority epoch and token validity after socket connection before sending data
                    async with self._authority_lock:
                        if self._authority_epoch != epoch_at_start or not self.is_key_valid(token):
                            raise PermissionError("Key revoked during target connection")

                    # Send GET request to target with safe headers only (NO Authorization header!)
                    req_lines = [
                        f"GET {tpath} HTTP/1.1",
                        f"Host: {thost}:{tport}",
                        "User-Agent: hermes-observer/1.0",
                        "Accept: application/json",
                        "Connection: close",
                        "",
                        "",
                    ]
                    twriter.write("\r\n".join(req_lines).encode("utf-8"))
                    await twriter.drain()

                    # Read status line
                    t_status_line = await treader.readline()
                    if not t_status_line:
                        raise ValueError("Empty target response")

                    t_parts = t_status_line.decode("iso-8859-1").split(" ")
                    if len(t_parts) < 2:
                        raise ValueError("Invalid target status line")

                    try:
                        t_status = int(t_parts[1])
                    except ValueError:
                        raise ValueError("Invalid target status code")

                    # Read target response headers with header count & total header byte bounds
                    header_count = 0
                    header_bytes = 0
                    t_headers: List[Tuple[str, str]] = []
                    while True:
                        thline = await treader.readline()
                        if not thline:
                            break
                        header_bytes += len(thline)
                        if header_bytes > MAX_TARGET_HEADER_BYTES:
                            raise ValueError("Target response headers exceeded byte limit")

                        if thline in (b"\r\n", b"\n"):
                            break

                        header_count += 1
                        if header_count > MAX_TARGET_HEADER_COUNT:
                            raise ValueError("Target response header count exceeded limit")

                        thstr = thline.decode("iso-8859-1").rstrip("\r\n")
                        if ":" in thstr:
                            tk, tv = thstr.split(":", 1)
                            t_headers.append((tk.strip(), tv.strip()))

                    # Reject redirects: 300..399 -> 502 and NO follow-up
                    if 300 <= t_status < 400:
                        raise ValueError(f"Target returned redirect status {t_status}")

                    if t_status != 200:
                        raise ValueError(f"Target returned HTTP status {t_status}")

                    # Read target body (max MAX_TARGET_RESPONSE_BYTES)
                    t_body = await treader.read(MAX_TARGET_RESPONSE_BYTES + 1)
                    if len(t_body) > MAX_TARGET_RESPONSE_BYTES:
                        raise ValueError("Target response payload exceeded byte limit")

                    return t_body

                finally:
                    twriter.close()
                    try:
                        await twriter.wait_closed()
                    except Exception:
                        pass

            try:
                t_body = await asyncio.wait_for(_do_target_exchange(), timeout=TARGET_TIMEOUT_SECONDS)
            except PermissionError:
                status_code, response_body = _json_error("unauthorized", "Unauthorized", 401)
                return
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning("Target operation timed out after %.1fs", TARGET_TIMEOUT_SECONDS)
                status_code, response_body = _json_error("observation_unavailable", "Target operation timed out", 502)
                return
            except Exception as te:
                logger.warning("Target operation failed: %s", te)
                status_code, response_body = _json_error("observation_unavailable", "Target operation failed", 502)
                return

            try:
                t_json = json.loads(t_body.decode("utf-8"))
                if not isinstance(t_json, dict):
                    raise ValueError("Target JSON must be object")
            except Exception:
                status_code, response_body = _json_error("observation_unavailable", "Target JSON invalid", 502)
                return

            # Target projection must fail closed: require exact allowlisted status string "ok"
            if t_json.get("status") != "ok":
                status_code, response_body = _json_error("observation_unavailable", "Target status is not ok", 502)
                return

            success_obj = {
                "action_id": ACTION_ID_HEALTH,
                "observation_identity": OBSERVATION_IDENTITY,
                "mutation_capability": "none",
                "provenance": "real_observation",
                "observed": {"status": "ok"},
                "limitations": [],
            }
            status_code = 200
            response_body = json.dumps(success_obj, separators=(",", ":")).encode("utf-8")


        except Exception as ex:
            logger.error("Unhandled observation service error: %s", type(ex).__name__)
            status_code, response_body = _json_error("observation_unavailable", "Internal error", 500)

        finally:
            self.audit_journal.record(action_name, status_code, target_contacted)
            status_text = "OK" if status_code == 200 else "Error"
            resp_lines = [
                f"HTTP/1.1 {status_code} {status_text}",
                "Content-Type: application/json",
                f"Content-Length: {len(response_body)}",
                "Connection: close",
                "",
                "",
            ]
            try:
                writer.write("\r\n".join(resp_lines).encode("utf-8") + response_body)
                await writer.drain()
            except Exception:
                pass
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        logger.info("Observation service listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    service = ObservationService()
    ok, msg = service.check_key_configuration()
    if not ok:
        logger.error("Observation service startup check failed: %s", msg)
        sys.exit(1)
    await service.start()
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())

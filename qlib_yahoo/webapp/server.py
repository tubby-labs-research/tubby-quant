"""Loopback-only HTTP server for the desktop UI.

Security model: bind 127.0.0.1 on a random port; a random token is exchanged
once for an HttpOnly, SameSite=Strict cookie; the Host header must name this
server (DNS-rebinding guard); state-changing POSTs also require the token
header and a same-origin Origin. Static files come only from the packaged
``static`` folder and report files only from ``<data root>/runs/*/reports`` plus
the four saved-data files those reports link to.
"""

from __future__ import annotations

import hmac
import json
import mimetypes
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from qlib_yahoo.webapp.service import ConsoleService, ServiceError, reject_recovery_options
from qlib_yahoo.dataset import DatasetError

STATIC = Path(__file__).resolve().parent / "static"
COOKIE = "tqc"
MAX_BODY = 1_000_000


def _same(supplied: str, token: str) -> bool:
    """Constant-time comparison that also refuses (rather than crashes on) non-ASCII input."""
    return hmac.compare_digest(supplied.encode("utf-8", "surrogateescape"), token.encode("utf-8"))


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, service: ConsoleService, port: int = 0):
        super().__init__(("127.0.0.1", port), Handler)
        self.service = service
        self.token = secrets.token_urlsafe(32)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    @property
    def start_url(self) -> str:
        return f"{self.origin}/?token={self.token}"


class Handler(BaseHTTPRequestHandler):
    server: AppServer
    server_version = "TubbyQlib/1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # quiet by default
        return

    # -------------------------------------------------------------- helpers

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _authorized(self) -> bool:
        cookies = {}
        for part in self.headers.get("Cookie", "").split(";"):
            if "=" in part:
                key, value = part.strip().split("=", 1)
                cookies[key] = value
        return _same(cookies.get(COOKIE, ""), self.server.token)

    def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, value: Any, status: int = 200) -> None:
        self._send(status, json.dumps(value, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _file(self, path: Path, extra: dict[str, str] | None = None) -> None:
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind in ("application/javascript", "application/json"):
            kind += "; charset=utf-8"
        self._send(200, path.read_bytes(), kind, extra)

    # -------------------------------------------------------------- routing

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._error(HTTPStatus.MISDIRECTED_REQUEST, "Unexpected host")
        url = urlparse(self.path)
        query = parse_qs(url.query, keep_blank_values=url.path in ('/api/sec-fields', '/api/sec-dictionary'))
        if url.path in ('/api/sec-fields', '/api/sec-dictionary') and any(len(values) != 1 for values in query.values()):
            return self._error(HTTPStatus.BAD_REQUEST, "SEC field parameters must occur once")
        if url.path == "/" and "token" in query:
            if not _same(query["token"][0], self.server.token):
                return self._error(HTTPStatus.FORBIDDEN, "Invalid token")
            return self._send(HTTPStatus.SEE_OTHER, b"", "text/plain", {
                "Location": "/", "Set-Cookie": f"{COOKIE}={self.server.token}; Path=/; HttpOnly; SameSite=Strict"})
        if not self._authorized():
            return self._error(HTTPStatus.FORBIDDEN, "Open the app from its launcher")
        if url.path in ("/", "/index.html"):
            return self._file(STATIC / "index.html", {"Content-Security-Policy": (
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                "script-src 'self'; frame-src 'self'; connect-src 'self'")})
        if url.path.startswith("/static/"):
            name = unquote(url.path[len("/static/"):])
            path = (STATIC / name).resolve()
            if STATIC.resolve() not in path.parents or not path.is_file():
                return self._error(HTTPStatus.NOT_FOUND, "Not found")
            return self._file(path)
        if url.path.startswith("/files/"):
            try:
                return self._file(self.server.service.report_file(unquote(url.path[len("/files/"):])))
            except ServiceError as exc:
                return self._error(HTTPStatus.NOT_FOUND, str(exc))
        if url.path == "/research-agent-guide":
            path = Path(__file__).resolve().parents[2] / "docs/RESEARCH_AGENT_API.md"
            if not path.is_file():
                return self._error(HTTPStatus.NOT_FOUND, "Agent API guide is unavailable")
            return self._send(HTTPStatus.OK, path.read_bytes(), "text/plain; charset=utf-8")
        if url.path.startswith("/api/"):
            return self._api("GET", url.path[5:], {key: values[0] for key, values in query.items()})
        return self._error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._error(HTTPStatus.MISDIRECTED_REQUEST, "Unexpected host")
        if not self._authorized():
            return self._error(HTTPStatus.FORBIDDEN, "Not authorized")
        origin = self.headers.get("Origin")
        if origin not in (None, self.server.origin, f"http://localhost:{self.server.server_address[1]}"):
            return self._error(HTTPStatus.FORBIDDEN, "Cross-origin request refused")
        if not _same(self.headers.get("X-Token", ""), self.server.token):
            return self._error(HTTPStatus.FORBIDDEN, "Missing request token")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request too large")
        try:
            raw = self.rfile.read(length) or b"{}"
            if urlparse(self.path).path.startswith("/api/workflow-chronology-"):
                from qlib_yahoo.target_bundle import _json as strict_json
                body = strict_json(raw)
            else:
                body = json.loads(raw)
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
        if not isinstance(body, dict):
            return self._error(HTTPStatus.BAD_REQUEST, "JSON object expected")
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            return self._error(HTTPStatus.NOT_FOUND, "Not found")
        return self._api("POST", url.path[5:], body)

    def _api(self, method: str, name: str, data: dict[str, Any]) -> None:
        service = self.server.service

        def sec_fields(dictionary=False):
            offset = data.get('offset', '0')
            custom = data.get('include_custom', 'false')
            if (not isinstance(offset, str) or not offset.isascii() or not offset.isdecimal()
                    or len(offset) > 8 or custom not in ('true', 'false')):
                raise ServiceError('Invalid SEC field page or include_custom boolean.')
            search = service.sec_dictionary if dictionary else service.sec_fields
            options = {"view": data.get("view", "recipes")} if dictionary else {}
            return search(data.get('catalog', ''), data.get('query', ''),
                          offset=int(offset), include_custom=custom == 'true', **options)

        routes = {
            ("GET", "session"): lambda: {"token": self.server.token},
            ("GET", "overview"): service.overview,
            ("GET", "datasets"): service.datasets,
            ("GET", "sec-catalogs"): service.sec_catalogs,
            ("GET", "sec-search"): lambda: service.sec_search(data.get("catalog", ""), str(data.get("query", "")), int(data.get("offset") or 0)),
            ("GET", "sec-fields"): sec_fields,
            ("GET", "sec-dictionary"): lambda: sec_fields(dictionary=True),
            ("POST", "sec-dictionary-build"): lambda: service.start_sec_dictionary_build(data),
            ("POST", "sec-name"): lambda: service.sec_name(data),
            ("GET", "sec-variants"): lambda: service.sec_variants(data.get("catalog", ""), str(data.get("tag", "")), str(data.get("taxonomy", ""))),
            ("GET", "sec-rows"): lambda: service.sec_rows(data.get("catalog", ""), str(data.get("table", "")), data.get("filters"), int(data.get("offset") or 0)),
            ("POST", "sec-bindings"): lambda: service._sec_config(data.get("sec")),
            ("GET", "dataset"): lambda: service.dataset(data.get("path")),
            ("GET", "runs"): service.runs,
            ("GET", "run"): lambda: service.run_detail(data.get("path", "")),
            ("GET", "job"): lambda: service.job(data.get("id", ""), int(data.get("since") or 0)),
            ("GET", "settings"): service.settings,
            ("POST", "check-alpha"): lambda: service.check_alpha(str(data.get("alpha", "")), **({"sec": data["sec"]} if "sec" in data else {})),
            ("POST", "backtest"): lambda: service.start_backtest(data),
            ("POST", "import"): lambda: service.start_import(data),
            ("POST", "report"): lambda: service.start_report(str(data.get("run", "")), str(data.get("profile", ""))),
            ("POST", "cancel"): lambda: {"cancelled": service.cancel()},
            ("POST", "default-dataset"): lambda: service.make_default(str(data.get("path", ""))),
            ("POST", "reveal"): lambda: service.reveal(str(data.get("path", ""))),
            ("POST", "open"): lambda: service.open_external(str(data.get("path", ""))),
            ("POST", "settings"): lambda: service.save_settings(data),
            ("GET", "alpha-library"): lambda: service.alpha_library(str(data.get("sort") or "created_at")),
            ("GET", "alpha-result"): lambda: service.alpha_result(str(data.get("id", ""))),
            ("GET", "alpha-batch"): lambda: service.alpha_batch(str(data.get("path", ""))),
            ("GET", "alpha-groups"): lambda: service.alpha_groups(data.get("dataset"), data.get("group_overlay")),
            ("POST", "alpha-groups"): lambda: service.alpha_groups(data.get("dataset"), data.get("group_overlay")),
            ("POST", "alpha-check"): lambda: service.alpha_check(str(data.get("text", "")), data.get("dataset"), **({key: data[key] for key in ("sec", "group_overlay") if key in data})),
            ("POST", "alpha-run"): lambda: service.start_alpha(data),
            ("GET", "workflow-attempts"): lambda: service.workflow_attempts(int(data.get("limit") or 100)),
            ("GET", "workflow-detail"): lambda: service.workflow_detail(str(data.get("id", ""))),
            ("POST", "workflow-preview"): lambda: service.workflow_preview(data),
            ("POST", "workflow-validate"): lambda: service.start_candidate_validation(data),
            ("POST", "workflow-recover"): service.workflow_recover,
            ("POST", "workflow-portfolio-preview"): lambda: service.workflow_portfolio_preview(data),
            ("POST", "workflow-portfolio-run"): lambda: service.start_portfolio_validation(data),
            ("POST", "workflow-chronology-preview"): lambda: service.workflow_chronology_preview(data),
            ("POST", "workflow-chronology-declare"): lambda: service.workflow_chronology_declare(data),
            ("POST", "workflow-chronology-trial"): lambda: service.start_chronology_trial(data),
            ("POST", "workflow-chronology-freeze"): lambda: service.workflow_chronology_freeze(data),
            ("POST", "workflow-chronology-final"): lambda: service.start_chronology_final(data),
            ("POST", "workflow-chronology-view"): lambda: service.workflow_chronology_view(data),
            ("GET", "sources"): service.latest_sources,
            ("POST", "annotate"): lambda: service.annotate(data),
            ("POST", "trash-runs"): lambda: service.trash_runs(data.get("paths")),
            ("GET", "trash"): service.trash,
            ("POST", "restore-runs"): lambda: service.restore_runs(data.get("names")),
            ("POST", "empty-trash"): lambda: service.empty_trash(data.get("names")),
            ("POST", "alpha-delete"): lambda: service.alpha_delete(data.get("ids"), bool(data.get("deleted", True))),
            ("GET", "run-signal"): lambda: service.run_signal(data.get("path", "")),
            ("GET", "backtest-batch"): lambda: service.backtest_batch(str(data.get("path", ""))),
            ("POST", "refresh-data"): lambda: service.start_refresh(data),
            ("GET", "research-overview"): service.research.overview,
            ("GET", "research-detail"): lambda: service.research.detail(str(data.get("id", ""))),
            ("GET", "research-series"): lambda: service.research.series(str(data.get("campaign_id", "")), str(data.get("experiment_id", ""))),
            ("GET", "research-config"): lambda: service.research.config(str(data.get("id", ""))),
            ("POST", "research-save"): lambda: service.research.save(data),
            ("POST", "research-agent-check"): lambda: service.research.check_agent(data),
            ("POST", "research-start"): lambda: service.start_research(data),
            ("POST", "research-demo"): lambda: service.start_research({}, demo=True),
            ("POST", "research-stop"): service.research.stop,
            ("POST", "research-finalize"): lambda: service.finalize_research(data),
            ("POST", "research-export"): lambda: service.research.export_batch(data),
        }
        handler = routes.get((method, name))
        if handler is None:
            return self._error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        try:
            # Submitted Run requests reach the service's durable invocation
            # boundary BEFORE its identical guard. Typing/discovery is not a trial.
            if name == "alpha-groups" or method == "POST" and name in ("check-alpha", "alpha-check"):
                reject_recovery_options(data)
            return self._json(handler())
        except (ServiceError, DatasetError) as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # keep the UI informed instead of dropping the connection
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")


def start_server(service: ConsoleService, port: int = 0) -> tuple[AppServer, threading.Thread]:
    server = AppServer(service, port)
    thread = threading.Thread(target=server.serve_forever, name="tqc-http", daemon=True)
    thread.start()
    return server, thread

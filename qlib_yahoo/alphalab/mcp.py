"""Model Context Protocol server for Alpha Lab and Research (stdio JSON-RPC, no dependencies).

An agent (Claude Code, Codex, any MCP client) can list operators, check and
simulate FASTEXPR alphas on prepared local data, read the saved alpha library
and read research-campaign reports, all inside the data root chosen at launch.
The server never places trades, never calls a model, never scores the reserved
research holdout, and refuses paths outside the data root.

    python -m qlib_yahoo.alphalab.mcp --data-root ~/tubby-data
    claude mcp add tubby-alpha-lab -- tubby-alpha-mcp --data-root ~/tubby-data

Transport: newline-delimited JSON-RPC 2.0 on stdin/stdout (the MCP ``stdio``
transport). Engine output is redirected to stderr so it cannot corrupt the
protocol stream; simulations run in this process (``workers`` controls the
batch pool, which never writes to stdout).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from qlib_yahoo import DEFAULT_DATA_ROOT, __version__
from qlib_yahoo.alphalab.settings import NEUTRALIZATIONS, Settings
from qlib_yahoo.dataset import DatasetError

SERVER_NAME = "tubby-alpha-lab"
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_MESSAGE_BYTES = 4_000_000
MAX_EXPRESSION = 4096
MAX_BATCH = 200
MAX_LIBRARY_ROWS = 500
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SETTING_KEYS = tuple(f.name for f in fields(Settings) if f.name != "evaluation_sessions")

INSTRUCTIONS = (
    "Alpha Lab simulates WorldQuant-BRAIN-style FASTEXPR alphas on prepared daily equity data. "
    "Call list_operators first, check_expression before simulating, and list_datasets to choose a dataset "
    "and universe. Metrics are fixed-book, close-to-close, after the configured turnover cost; they are "
    "research evidence, not a trading recommendation. Research holdouts are never scored here."
)

SETTINGS_SCHEMA = {
    "dataset": {"type": "string", "description": "Dataset folder name from list_datasets; default: the data root's default dataset."},
    "universe": {"type": "string", "description": "Named universe such as TOP3000 or TOP500; default: the dataset's base universe."},
    "delay": {"type": "integer", "enum": [0, 1], "description": "1 (default) trades the next session on today's signal; 0 is idealized same-close."},
    "decay": {"type": "integer", "minimum": 0, "maximum": 512, "description": "Linear decay window in sessions (default 0)."},
    "neutralization": {"type": "string", "enum": list(NEUTRALIZATIONS), "description": "MARKET (default), NONE, SECTOR, INDUSTRY or COUNTRY."},
    "truncation": {"type": "number", "minimum": 0, "maximum": 1, "description": "Largest absolute weight per stock (default 0.08; 0 disables)."},
    "pasteurization": {"type": "boolean", "description": "Hold positions only inside the universe (default true)."},
    "booksize": {"type": "number", "exclusiveMinimum": 0, "description": "Dollar book (default 20,000,000)."},
    "cost_bps": {"type": "number", "minimum": 0, "maximum": 1000, "description": "Cost per dollar traded in basis points (default 0; BRAIN-like)."},
    "start": {"type": "string", "description": "First test session YYYY-MM-DD (default: first full-universe session after warm-up)."},
    "end": {"type": "string", "description": "Last test session YYYY-MM-DD (default: dataset end)."},
    "warmup": {"type": "integer", "minimum": 0, "maximum": 2000, "description": "History sessions before start for time-series operators (default 252)."},
    "oos_start": {"type": "string", "description": "Also report in-sample / out-of-sample metrics split at this date."},
}

TOOLS: list[dict[str, Any]] = [
    {"name": "list_operators",
     "description": "FASTEXPR operators with their argument kinds and options, data fields, groups and aliases.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "check_expression",
     "description": "Check FASTEXPR syntax, operator signatures and field names without touching data.",
     "inputSchema": {"type": "object", "properties": {"expression": {"type": "string", "maxLength": MAX_EXPRESSION}},
                     "required": ["expression"], "additionalProperties": False}},
    {"name": "list_datasets",
     "description": "Prepared datasets under the data root with status, date range, fields and named universes.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "simulate_alpha",
     "description": "Simulate one alpha with BRAIN-style settings; returns summary metrics, yearly table, checks and settings.",
     "inputSchema": {"type": "object",
                     "properties": {"expression": {"type": "string", "maxLength": MAX_EXPRESSION},
                                    "include_daily": {"type": "boolean", "description": "Also return daily PnL and turnover series (default false)."},
                                    **SETTINGS_SCHEMA},
                     "required": ["expression"], "additionalProperties": False}},
    {"name": "simulate_batch",
     "description": "Simulate up to 200 alphas with the same settings; returns a leaderboard and the most correlated pairs.",
     "inputSchema": {"type": "object",
                     "properties": {"expressions": {"type": "array", "items": {"type": "string", "maxLength": MAX_EXPRESSION},
                                                    "minItems": 1, "maxItems": MAX_BATCH},
                                    "force": {"type": "boolean", "description": "Simulate again even when the library already has the result."},
                                    **SETTINGS_SCHEMA},
                     "required": ["expressions"], "additionalProperties": False}},
    {"name": "library_list",
     "description": "Saved Alpha Lab library entries with their headline metrics.",
     "inputSchema": {"type": "object",
                     "properties": {"sort": {"type": "string", "enum": ["fitness", "sharpe", "returns", "turnover", "created_at", "self_correlation"]},
                                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIBRARY_ROWS}},
                     "additionalProperties": False}},
    {"name": "research_report",
     "description": "Read-only report of research campaigns under the data root: one row per hypothesis with its latest decision, validation metrics and any final holdout verdict.",
     "inputSchema": {"type": "object",
                     "properties": {"campaign_id": {"type": "string", "maxLength": 128},
                                    "markdown": {"type": "boolean", "description": "Return a Markdown summary instead of JSON."}},
                     "additionalProperties": False}},
]


class ToolError(ValueError):
    """A problem with the request that the caller can fix."""


class AlphaLabServer:
    def __init__(self, data_root: str | Path, *, dataset: str | Path | None = None, workers: int = 2):
        self.data_root = Path(data_root).expanduser().resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.default_dataset = Path(dataset).expanduser() if dataset else None
        self.workers = max(1, int(workers))
        self.protocol_version = PROTOCOL_VERSIONS[0]
        self.initialized = False

    # ------------------------------------------------------------------ JSON-RPC
    def handle(self, message: Any) -> dict | None:
        """One request or notification; returns the response, or None for notifications."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return _error(message.get("id") if isinstance(message, dict) else None, -32600, "Invalid Request")
        request_id = message.get("id")
        method, params = message["method"], message.get("params") or {}
        if not isinstance(params, dict):
            return _error(request_id, -32602, "params must be an object")
        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "notifications/initialized":
                self.initialized = True
                return None
            elif method.startswith("notifications/"):
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self.call_tool(params.get("name"), params.get("arguments") or {})
            elif method in ("resources/list", "resources/templates/list"):
                result = {"resources": []} if method == "resources/list" else {"resourceTemplates": []}
            elif method == "prompts/list":
                result = {"prompts": []}
            else:
                return _error(request_id, -32601, f"Method not found: {method}")
        except ToolError as error:
            return _error(request_id, -32602, str(error))
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        self.protocol_version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return {"protocolVersion": self.protocol_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": __version__},
                "instructions": INSTRUCTIONS}

    # --------------------------------------------------------------------- tools
    def call_tool(self, name: Any, arguments: Any) -> dict:
        if not isinstance(name, str) or name not in {tool["name"] for tool in TOOLS}:
            raise ToolError(f"Unknown tool: {name!r}")
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be an object")
        handler = getattr(self, "tool_" + name)
        try:
            with contextlib.redirect_stdout(sys.stderr):  # the protocol owns stdout
                payload = handler(**arguments)
        except TypeError as error:
            return _tool_error(f"{name}: unexpected or missing argument ({error})")
        except (ToolError, DatasetError, ValueError) as error:
            return _tool_error(f"{name}: {self._redact(str(error))}")
        except Exception as error:  # the engine failed; never leak local paths
            return _tool_error(f"{name}: {type(error).__name__}: {self._redact(str(error))}")
        text = json.dumps(payload, allow_nan=False, ensure_ascii=False) if not isinstance(payload, str) else payload
        result = {"content": [{"type": "text", "text": text}], "isError": False}
        if isinstance(payload, dict):
            result["structuredContent"] = payload
        return result

    def _redact(self, text: str) -> str:
        return text.replace(str(self.data_root), "<data-root>")

    def tool_list_operators(self) -> dict:
        from qlib_yahoo.alphalab.engine import ALIASES, BASE_FIELDS, GROUPS, OPTIONAL_FIELDS, REGISTRY, operator_names

        kind_names = {"x": "series", "n": "number", "s": "text", "g": "group", "*": "series..."}
        signatures = {}
        for name, (_function, kinds, defaults) in sorted(REGISTRY.items()):
            positional = [kind_names.get(kind, kind) for kind in kinds]
            options = {key: (None if value is None or (isinstance(value, float) and not math.isfinite(value)) else value)
                       for key, value in defaults.items()}  # unbounded defaults (clamp) are reported as null
            signatures[name] = {"positional": positional, "options": options}
        return {"operators": operator_names(), "signatures": signatures, "aliases": dict(ALIASES),
                "elementwise": ["add", "subtract", "multiply", "divide", "and", "or", "not"],
                "fields": [*BASE_FIELDS, "returns", "vwap", "dollar_volume", "adv<d>", *OPTIONAL_FIELDS],
                "optional_fields_note": "close_raw, dividend and cap exist only when the snapshot stores them; "
                                        "adv<d> is the mean dollar volume over d sessions (adv20).",
                "groups": list(GROUPS),
                "grammar": "bare field names; operator(args, option=value); + - * / ^; comparisons; && || !; "
                           "condition ? a : b; assignments separated by ';' with the last statement as the alpha"}

    def tool_check_expression(self, expression: Any) -> dict:
        from qlib_yahoo.alphalab.engine import check_expression

        problems = check_expression(self._expression(expression))
        return {"ok": not problems, "problems": problems}

    def tool_list_datasets(self) -> dict:
        from qlib_yahoo.workspace import list_datasets

        rows = []
        for row in list_datasets(self.data_root):
            universes = row.get("universes") or {}
            rows.append({"name": row.get("name"), "status": row.get("status"), "provider": row.get("provider"),
                         "is_default": bool(row.get("is_default")), "symbols": row.get("symbols"),
                         "sessions": row.get("sessions"), "start": _text(row.get("start")), "end": _text(row.get("end")),
                         "fields": list(row.get("fields") or []),
                         "universes": {name: (record.get("size") if isinstance(record, dict) else None)
                                       for name, record in universes.items()},
                         "price_eligible": bool(row.get("price_eligible"))})
        return {"data_root": "<data-root>", "datasets": rows}

    def tool_simulate_alpha(self, expression: Any, include_daily: bool = False, **settings: Any) -> dict:
        from qlib_yahoo.alphalab.batch import run_batch
        from qlib_yahoo.alphalab.simulate import json_safe

        dataset = self._dataset(settings.pop("dataset", None))
        resolved = self._settings(dataset, settings)
        result = run_batch(dataset, [self._expression(expression)], resolved, data_root=self.data_root,
                           workers=1, worker_ceiling=1)[0]
        if "error" in result:
            raise ToolError(result["error"])
        payload = {key: result.get(key) for key in ("id", "expression", "engine", "settings", "summary", "yearly",
                                                    "periods", "checks", "cached", "self_correlation")}
        if include_daily:
            payload["daily"] = result.get("daily")
        payload["dataset"] = dataset.name
        return json_safe(payload)

    def tool_simulate_batch(self, expressions: Any, force: bool = False, **settings: Any) -> dict:
        from qlib_yahoo.alphalab.batch import batch_correlations, run_batch
        from qlib_yahoo.alphalab.simulate import json_safe

        if not isinstance(expressions, list) or not 1 <= len(expressions) <= MAX_BATCH:
            raise ToolError(f"expressions must be a list of 1 to {MAX_BATCH} FASTEXPR strings")
        texts = [self._expression(item) for item in expressions]
        dataset = self._dataset(settings.pop("dataset", None))
        resolved = self._settings(dataset, settings)
        results = run_batch(dataset, texts, resolved, data_root=self.data_root, workers=min(self.workers, len(texts)),
                            worker_ceiling=max(1, self.workers), force=bool(force))
        rows = []
        for result in results:
            summary = result.get("summary") or {}
            rows.append({"id": result.get("id"), "expression": result["expression"], "error": result.get("error"),
                         "cached": bool(result.get("cached")), "self_correlation": result.get("self_correlation"),
                         **{key: summary.get(key) for key in ("sharpe", "fitness", "returns", "turnover", "drawdown",
                                                             "margin_bps", "long_count", "short_count", "coverage")}})
        ranked = sorted((row for row in rows if not row["error"]),
                        key=lambda row: -(row["fitness"] if isinstance(row["fitness"], (int, float)) else -1e9))
        ids, matrix = batch_correlations(results)
        pairs = sorted(((matrix[i][j], ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))
                        if matrix[i][j] is not None), reverse=True)[:10]
        return json_safe({"dataset": dataset.name, "settings": resolved.__dict__, "leaderboard": ranked,
                          "errors": [row for row in rows if row["error"]],
                          "most_correlated": [{"correlation": c, "a": a, "b": b} for c, a, b in pairs]})

    def tool_library_list(self, sort: str = "fitness", limit: int = 30) -> dict:
        from qlib_yahoo.alphalab.batch import Library

        if type(limit) is not int or not 1 <= limit <= MAX_LIBRARY_ROWS:
            raise ToolError(f"limit must be between 1 and {MAX_LIBRARY_ROWS}")
        rows = Library(self.data_root / "alphalab" / "library.sqlite").rows(sort=sort, limit=limit)
        keep = ("id", "created_at", "expression", "engine", "sharpe", "fitness", "returns", "turnover", "drawdown",
                "status", "self_correlation")
        return {"alphas": [{key: row.get(key) for key in keep if key in row} for row in rows]}

    def tool_research_report(self, campaign_id: str | None = None, markdown: bool = False) -> Any:
        from autoresearch.records import ResearchStore

        if campaign_id is not None and (not isinstance(campaign_id, str) or not _NAME.fullmatch(campaign_id)):
            raise ToolError("campaign_id must be a record identifier")
        root = self.data_root / "autoresearch" / "records"
        empty = {"campaign_id": campaign_id, "counts": {}, "decisions": {}, "hypotheses": [], "finals": []}
        if not root.is_dir():
            return "No research records under the data root." if markdown else empty
        store = ResearchStore(root)
        try:  # the hand-off module (finals, markdown) may be newer than this checkout
            from autoresearch.handoff import campaign_report, render_markdown
        except ImportError:
            report = _basic_campaign_report(store, campaign_id)
            return _basic_markdown(report) if markdown else report
        report = campaign_report(store, campaign_id)
        return render_markdown(report) if markdown else report

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _expression(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ToolError("expression must be nonempty text")
        if len(value) > MAX_EXPRESSION:
            raise ToolError(f"expression exceeds {MAX_EXPRESSION} characters")
        return value.strip()

    def _dataset(self, name: Any) -> Path:
        from qlib_yahoo.dataset import resolve_dataset

        if name is None:
            return resolve_dataset(self.data_root, self.default_dataset)
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in {".", ".."}:
            raise ToolError("dataset must be a folder name from list_datasets")
        candidate = self.data_root / "datasets" / name
        if not candidate.resolve().is_relative_to(self.data_root):
            raise ToolError("dataset must stay inside the data root")
        return resolve_dataset(self.data_root, candidate)

    def _settings(self, dataset: Path, values: dict) -> Settings:
        unknown = sorted(set(values) - set(_SETTING_KEYS))
        if unknown:
            raise ToolError("unknown settings: " + ", ".join(unknown))
        if "universe" not in values:
            manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
            from qlib_yahoo.universe import dataset_universes
            universes = dataset_universes(manifest)
            if not universes:
                raise ToolError("this dataset has no named universe; pass universe explicitly")
            values = {**values, "universe": sorted(universes)[0].upper()}
        for key in ("delay", "decay", "warmup"):
            if key in values and isinstance(values[key], float) and values[key].is_integer():
                values[key] = int(values[key])
        try:
            return Settings(**values).validate()
        except TypeError as error:
            raise ToolError(f"invalid settings: {error}") from None


def _text(value: Any) -> Any:
    return None if value is None else str(value)


def _basic_campaign_report(store: Any, campaign_id: str | None) -> dict:
    """Latest decision and validation metrics per hypothesis, from the immutable records alone."""
    by_time = lambda records: sorted(records, key=lambda r: (str(r.get("created_at", "")), r["id"]))
    experiments = [e for e in by_time(store.list("experiments")) if campaign_id is None or e.get("campaign_id") == campaign_id]
    latest_experiment = {e["hypothesis_id"]: e for e in experiments}
    ids = {e["id"] for e in experiments}
    latest_review = {r["hypothesis_id"]: r for r in by_time(store.list("reviews")) if r.get("experiment_id") in ids}
    rows = []
    for hypothesis in by_time(store.list("hypotheses")):
        experiment = latest_experiment.get(hypothesis["id"])
        if experiment is None:
            continue
        review = latest_review.get(hypothesis["id"], {})
        validation = (experiment.get("result") or {}).get("metrics", {}).get("validation", {})
        rows.append({"hypothesis_id": hypothesis["id"], "title": hypothesis.get("title", ""),
                     "family": hypothesis.get("family", ""), "expression": hypothesis.get("expression", ""),
                     "outcome": (experiment.get("result") or {}).get("status"), "decision": review.get("decision"),
                     "reason": review.get("reason", ""),
                     "validation": {key: validation.get(key) for key in ("days", "sharpe", "returns", "turnover", "drawdown")},
                     "final": None})
    return {"campaign_id": campaign_id,
            "counts": {name: len(store.list(name)) for name in ("sources", "hypotheses", "experiments", "reviews")},
            "decisions": {d: sum(1 for r in rows if r["decision"] == d) for d in sorted({r["decision"] for r in rows if r["decision"]})},
            "hypotheses": rows, "finals": []}


def _basic_markdown(report: dict) -> str:
    lines = ["# Research campaign report", "", "| Decision | Title | Validation Sharpe | Expression |", "| --- | --- | --- | --- |"]
    for row in report["hypotheses"]:
        sharpe = row["validation"]["sharpe"]
        lines.append(f"| {row['decision'] or row['outcome']} | {row['title']} | "
                     f"{sharpe if isinstance(sharpe, (int, float)) else '-'} | `{row['expression']}` |")
    return "\n".join(lines) + "\n"


def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_error(message: str) -> dict:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def serve(server: AlphaLabServer, stdin=None, stdout=None) -> int:
    """Read newline-delimited JSON-RPC messages until EOF; write responses as single lines."""
    stdin = stdin if stdin is not None else sys.stdin.buffer
    stdout = stdout if stdout is not None else sys.stdout.buffer

    def emit(response: dict | None) -> None:
        if response is not None:
            stdout.write(json.dumps(response, allow_nan=False, ensure_ascii=False).encode("utf-8") + b"\n")
            stdout.flush()

    for raw in iter(stdin.readline, b""):
        line = raw.strip()
        if not line:
            continue
        if len(line) > MAX_MESSAGE_BYTES:
            emit(_error(None, -32600, "message exceeds the size limit"))
            continue
        try:
            message = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            emit(_error(None, -32700, "Parse error"))
            continue
        for item in (message if isinstance(message, list) else [message]):
            emit(server.handle(item))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alpha Lab MCP server (stdio).")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT),
                        help="data root holding datasets/, alphalab/ and autoresearch/ (default: TUBBY_QLIB_DATA_ROOT or the platform default)")
    parser.add_argument("--dataset", type=Path, help="default dataset folder (default: the data root's latest.json)")
    parser.add_argument("--workers", type=int, default=2, help="batch worker processes (default 2)")
    args = parser.parse_args(argv)
    server = AlphaLabServer(args.data_root, dataset=args.dataset, workers=args.workers)
    print(f"{SERVER_NAME} {__version__}: data root {server.data_root}", file=sys.stderr, flush=True)
    return serve(server)


if __name__ == "__main__":
    raise SystemExit(main())

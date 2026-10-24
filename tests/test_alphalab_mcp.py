"""Alpha Lab MCP server: protocol handling in-process and over stdio, tool contracts and data-root containment."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from qlib_yahoo.alphalab.mcp import TOOLS, AlphaLabServer, serve

HAVE_NUMERIC = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))
needs_engine = unittest.skipUnless(HAVE_NUMERIC, "operator and expression tools import the numerical engine")
ROOT = Path(__file__).resolve().parents[1]


def request(method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "method": method, "id": request_id}
    if params is not None:
        message["params"] = params
    return message


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.server = AlphaLabServer(Path(self.temp.name) / "data")

    def test_initialize_lists_tools_and_rejects_unknown_methods(self):
        response = self.server.handle(request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}}))
        self.assertEqual(response["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(response["result"]["serverInfo"]["name"], "tubby-alpha-lab")
        self.assertIn("tools", response["result"]["capabilities"])
        self.assertEqual(self.server.handle(request("initialize", {"protocolVersion": "1999-01-01"}))["result"]["protocolVersion"], "2025-06-18")
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertTrue(self.server.initialized)
        self.assertEqual(self.server.handle(request("ping"))["result"], {})
        names = [tool["name"] for tool in self.server.handle(request("tools/list"))["result"]["tools"]]
        self.assertEqual(names, [tool["name"] for tool in TOOLS])
        self.assertEqual({"list_operators", "check_expression", "list_datasets", "simulate_alpha", "simulate_batch",
                          "library_list", "research_report"}, set(names))
        for tool in TOOLS:
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertFalse(tool["inputSchema"].get("additionalProperties", True))
        self.assertEqual(self.server.handle(request("nope"))["error"]["code"], -32601)
        self.assertEqual(self.server.handle({"jsonrpc": "1.0", "method": "ping", "id": 3})["error"]["code"], -32600)
        self.assertEqual(self.server.handle(request("tools/call", {"name": "missing", "arguments": {}}))["error"]["code"], -32602)
        self.assertEqual(self.server.handle(request("resources/list"))["result"], {"resources": []})
        self.assertEqual(self.server.handle(request("prompts/list"))["result"], {"prompts": []})

    @needs_engine
    def test_operator_and_expression_tools_need_no_data(self):
        operators = self.server.handle(request("tools/call", {"name": "list_operators", "arguments": {}}))["result"]
        self.assertFalse(operators["isError"])
        payload = operators["structuredContent"]
        self.assertIn("ts_decay_linear", payload["operators"])
        self.assertEqual(payload["signatures"]["ts_corr"]["positional"], ["series", "series", "number"])
        self.assertEqual(payload["signatures"]["ts_rank"]["options"], {"constant": 0})
        self.assertEqual(payload["aliases"]["delay"], "ts_delay")
        self.assertEqual(json.loads(operators["content"][0]["text"])["groups"], ["market", "sector", "industry", "country"])
        ok = self.server.handle(request("tools/call", {"name": "check_expression", "arguments": {"expression": "rank(ts_delta(close, 5))"}}))
        self.assertEqual(ok["result"]["structuredContent"], {"ok": True, "problems": []})
        bad = self.server.handle(request("tools/call", {"name": "check_expression", "arguments": {"expression": "rank(nope) + ts_mean(close)"}}))
        problems = bad["result"]["structuredContent"]["problems"]
        self.assertFalse(bad["result"]["structuredContent"]["ok"])
        self.assertTrue(any("nope" in p for p in problems) and any("ts_mean" in p for p in problems))
        empty = self.server.handle(request("tools/call", {"name": "check_expression", "arguments": {"expression": "  "}}))
        self.assertTrue(empty["result"]["isError"])
        extra = self.server.handle(request("tools/call", {"name": "check_expression", "arguments": {"expression": "close", "bogus": 1}}))
        self.assertTrue(extra["result"]["isError"])
        self.assertIn("unexpected", extra["result"]["content"][0]["text"])

    def test_no_data_root_contents_yet(self):
        datasets = self.server.handle(request("tools/call", {"name": "list_datasets", "arguments": {}}))["result"]
        self.assertEqual(datasets["structuredContent"]["datasets"], [])
        report = self.server.handle(request("tools/call", {"name": "research_report", "arguments": {}}))["result"]
        self.assertEqual(report["structuredContent"]["hypotheses"], [])
        text = self.server.handle(request("tools/call", {"name": "research_report", "arguments": {"markdown": True}}))["result"]
        self.assertIn("No research records", text["content"][0]["text"])
        missing = self.server.handle(request("tools/call", {"name": "simulate_alpha", "arguments": {"expression": "rank(close)"}}))["result"]
        self.assertTrue(missing["isError"])
        self.assertNotIn(self.temp.name, missing["content"][0]["text"])

    @needs_engine
    def test_stdio_framing(self):
        lines = [json.dumps(request("initialize", {"protocolVersion": "2025-06-18"})),
                 json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                 "", "not json",
                 json.dumps([request("ping", request_id=2), request("tools/call", {"name": "check_expression", "arguments": {"expression": "close"}}, request_id=3)]),
                 json.dumps(request("tools/list", request_id=4))]
        stdin = io.BytesIO("\n".join(lines).encode() + b"\n")
        stdout = io.BytesIO()
        self.assertEqual(serve(self.server, stdin, stdout), 0)
        responses = [json.loads(line) for line in stdout.getvalue().decode().splitlines()]
        self.assertEqual([r.get("id") for r in responses], [1, None, 2, 3, 4])
        self.assertEqual(responses[1]["error"]["code"], -32700)
        self.assertEqual(responses[2]["result"], {})
        self.assertTrue(responses[3]["result"]["structuredContent"]["ok"])


@unittest.skipUnless(HAVE_NUMERIC, "requires local numpy and pandas")
class SimulationTests(unittest.TestCase):
    def setUp(self):
        from autoresearch.backend import create_demo_dataset

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_root = Path(os.path.realpath(self.temp.name)) / "data"
        dataset = create_demo_dataset(self.data_root / "datasets" / "demo")
        (self.data_root / "latest.json").write_text(json.dumps({"dataset": str(dataset)}))
        self.server = AlphaLabServer(self.data_root)
        self.window = {"start": "2024-01-02", "end": "2024-06-28", "warmup": 60, "truncation": 0}

    def call(self, name, **arguments):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            response = self.server.handle(request("tools/call", {"name": name, "arguments": arguments}))
        self.assertEqual(captured.getvalue(), "", "tool output must never reach the protocol stream")
        return response["result"]

    def test_datasets_simulation_batch_and_library(self):
        datasets = self.call("list_datasets")["structuredContent"]["datasets"]
        self.assertEqual([row["name"] for row in datasets], ["demo"])
        self.assertEqual(datasets[0]["universes"], {"top8": 8})
        self.assertTrue(datasets[0]["is_default"])
        single = self.call("simulate_alpha", expression="rank(ts_delta(close, 5))", **self.window)
        self.assertFalse(single["isError"], single["content"][0]["text"])
        payload = single["structuredContent"]
        self.assertEqual(payload["dataset"], "demo")
        self.assertEqual(payload["settings"]["universe"], "TOP8")  # defaulted to the dataset's base universe
        self.assertEqual(payload["settings"]["delay"], 1)
        self.assertIn("sharpe", payload["summary"])
        self.assertEqual(payload["summary"]["start"], "2024-01-02")
        self.assertNotIn("daily", payload)
        self.assertNotIn("weights", json.dumps(payload))
        with_daily = self.call("simulate_alpha", expression="rank(ts_delta(close, 5))", include_daily=True, **self.window)
        self.assertEqual(len(with_daily["structuredContent"]["daily"]["dates"]), payload["summary"]["days"])
        self.assertTrue(with_daily["structuredContent"]["cached"])
        named = self.call("simulate_alpha", expression="rank(close)", dataset="demo", universe="top8", delay=0, **self.window)
        self.assertEqual(named["structuredContent"]["settings"]["delay"], 0)
        batch = self.call("simulate_batch", expressions=["rank(ts_delta(close, 5))", "-rank(ts_delta(close, 5))", "rank(cap)"], **self.window)
        self.assertFalse(batch["isError"], batch["content"][0]["text"])
        board = batch["structuredContent"]
        self.assertEqual(len(board["leaderboard"]), 2)
        self.assertEqual(len(board["errors"]), 1)
        self.assertIn("cap", board["errors"][0]["error"])
        self.assertGreaterEqual(board["leaderboard"][0]["fitness"] or -1e9, board["leaderboard"][1]["fitness"] or -1e9)
        self.assertEqual(len(board["most_correlated"]), 1)
        self.assertAlmostEqual(board["most_correlated"][0]["correlation"], -1.0, places=3)
        library = self.call("library_list", sort="sharpe", limit=10)["structuredContent"]["alphas"]
        self.assertGreaterEqual(len(library), 2)
        self.assertTrue(all("expression" in row and "sharpe" in row for row in library))
        self.assertNotIn(str(self.data_root), json.dumps(library))

    def test_errors_are_contained_and_readable(self):
        bad_setting = self.call("simulate_alpha", expression="rank(close)", decay=-1, **self.window)
        self.assertTrue(bad_setting["isError"])
        self.assertIn("decay", bad_setting["content"][0]["text"])
        unknown = self.call("simulate_alpha", expression="rank(close)", speed=3)
        self.assertTrue(unknown["isError"])
        self.assertIn("unknown settings: speed", unknown["content"][0]["text"])
        bad_expression = self.call("simulate_alpha", expression="rank(close", **self.window)
        self.assertTrue(bad_expression["isError"])
        self.assertIn("expected ')'", bad_expression["content"][0]["text"])
        for dataset in ("../data", "/etc", "demo/../demo", "nope"):
            with self.subTest(dataset=dataset):
                escaped = self.call("simulate_alpha", expression="rank(close)", dataset=dataset, **self.window)
                self.assertTrue(escaped["isError"])
                self.assertNotIn(str(self.data_root), escaped["content"][0]["text"])
        too_many = self.call("simulate_batch", expressions=["rank(close)"] * 201, **self.window)
        self.assertTrue(too_many["isError"])
        limit = self.call("library_list", limit=0)
        self.assertTrue(limit["isError"])

    def test_subprocess_stdio_round_trip(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT), "TUBBY_QLIB_DATA_ROOT": str(self.data_root), "PYTHONUTF8": "1"}
        messages = [request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test"}}),
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    request("tools/call", {"name": "simulate_alpha", "arguments": {"expression": "rank(ts_delta(close, 5))", **self.window}}, request_id=2),
                    request("tools/call", {"name": "list_datasets", "arguments": {}}, request_id=3)]
        process = subprocess.run([sys.executable, "-m", "qlib_yahoo.alphalab.mcp", "--data-root", str(self.data_root)],
                                 input="\n".join(json.dumps(m) for m in messages).encode() + b"\n",
                                 capture_output=True, cwd=str(ROOT), env=env, timeout=180)
        self.assertEqual(process.returncode, 0, process.stderr.decode(errors="replace")[-2000:])
        responses = [json.loads(line) for line in process.stdout.decode().splitlines() if line.strip()]
        self.assertEqual([r["id"] for r in responses], [1, 2, 3])
        self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "tubby-alpha-lab")
        self.assertFalse(responses[1]["result"]["isError"], responses[1]["result"]["content"][0]["text"])
        self.assertIn("sharpe", responses[1]["result"]["structuredContent"]["summary"])
        self.assertEqual(responses[2]["result"]["structuredContent"]["datasets"][0]["name"], "demo")
        self.assertIn("data root", process.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()

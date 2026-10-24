"""OpenAI-compatible adapter against a local stub server: transport limits, auth, parsing and grounding."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
import threading
import unittest
from unittest.mock import patch

from autoresearch.models import ModelError, OpenAICompatibleModel, make_model

SOURCES = [{"id": "src1", "title": "Volume and reversal", "url": "https://example.invalid/1",
            "text": "High volume days are followed by short-term reversal in daily returns."}]
CANDIDATE = {"title": "Volume reversal", "mechanism": "Liquidity provision after volume spikes",
             "family": "reversal", "expression": "-returns * rank(volume)", "source_ids": ["src1"],
             "evidence": [{"source_id": "src1", "quote": "followed by short-term reversal"}],
             "expected_horizon": "one to five sessions", "required_fields": ["returns", "volume"],
             "rationale": "Bounded test of the stated mechanism."}


class Stub(BaseHTTPRequestHandler):
    reply = None  # (status, content_type, body_bytes) or a callable(handler)
    requests = []

    def log_message(self, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        Stub.requests.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(body)})
        status, content_type, payload = Stub.reply(self) if callable(Stub.reply) else Stub.reply
        self.send_response(status)
        if status in (301, 302):
            self.send_header("Location", "http://127.0.0.1:9/elsewhere")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def completion(text):
    return 200, "application/json; charset=utf-8", json.dumps(
        {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]}).encode()


HAVE_NUMERIC = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"))
needs_engine = unittest.skipUnless(HAVE_NUMERIC, "generation lists the engine's operators, which needs numpy and pandas")


class OpenAICompatibleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Stub.requests = []
        Stub.reply = completion(json.dumps([CANDIDATE]))

    def model(self, **overrides):
        return make_model({"provider": "openai", "model": "stub-model", "base_url": self.base_url, "api_key_env": "STUB_KEY",
                           "max_candidates": 2, **overrides})

    def test_configuration_is_explicit_and_credential_free(self):
        self.assertIsInstance(self.model(), OpenAICompatibleModel)
        for bad in ({"model": ""}, {"api_key": "sk-x"}, {"base_url": "http://example.com/v1"},
                    {"base_url": "https://u:p@example.com/v1"}, {"base_url": "https://example.com/v1?x=1"},
                    {"api_key_env": "not valid"}, {"max_tokens": 0}, {"timeout_seconds": 0}, {"temperature": 3}):
            with self.subTest(bad=bad), self.assertRaises(ModelError):
                self.model(**bad)
        identity = self.model().identity()
        self.assertEqual(identity["provider"], "openai")
        self.assertEqual(identity["endpoint_host"], "127.0.0.1")
        self.assertNotIn("STUB_KEY", json.dumps(identity))
        self.assertNotIn("api_key", json.dumps(identity))
        remote = OpenAICompatibleModel({"provider": "openai", "model": "gpt", "max_candidates": 1})
        self.assertEqual(remote.host, "api.openai.com")
        self.assertFalse(remote.loopback)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ModelError, "environment variable is unset"):
            remote.propose(SOURCES, [])

    @needs_engine
    def test_request_shape_auth_and_grounded_candidates(self):
        with patch.dict(os.environ, {"STUB_KEY": "secret-token"}):
            candidates = self.model().propose(SOURCES, [{"title": "earlier idea"}])
        self.assertEqual([c["expression"] for c in candidates], ["-returns * rank(volume)"])
        self.assertEqual(candidates[0]["provenance"]["provider"], "openai")
        request = Stub.requests[-1]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["headers"]["Authorization"], "Bearer secret-token")
        body = request["body"]
        self.assertEqual(body["model"], "stub-model")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        user = json.loads(body["messages"][1]["content"])
        self.assertEqual(user["task"], "propose")
        self.assertEqual(user["max_candidates"], 2)
        self.assertIn("rank", user["operator_names"])
        self.assertEqual(user["sources"][0]["id"], "src1")
        self.assertEqual(user["history"], [{"title": "earlier idea"}])

    @needs_engine
    def test_loopback_server_needs_no_credential(self):
        with patch.dict(os.environ, {}, clear=True):
            self.model(api_key_env=None).propose(SOURCES, [])
        self.assertNotIn("Authorization", Stub.requests[-1]["headers"])

    @needs_engine
    def test_fenced_and_wrapped_outputs_are_accepted_and_ungrounded_ones_rejected(self):
        with patch.dict(os.environ, {"STUB_KEY": "k"}):
            Stub.reply = completion("```json\n" + json.dumps([CANDIDATE]) + "\n```")
            self.assertEqual(len(self.model().propose(SOURCES, [])), 1)
            Stub.reply = completion(json.dumps({"candidates": [CANDIDATE]}))
            self.assertEqual(len(self.model().propose(SOURCES, [])), 1)
            Stub.reply = completion("not json")
            with self.assertRaisesRegex(ModelError, "JSON array"):
                self.model().propose(SOURCES, [])
            Stub.reply = completion(json.dumps({"a": 1, "b": 2}))
            with self.assertRaisesRegex(ModelError, "JSON array"):
                self.model().propose(SOURCES, [])
            invented = {**CANDIDATE, "evidence": [{"source_id": "src1", "quote": "text that is not in the source"}]}
            Stub.reply = completion(json.dumps([invented]))
            with self.assertRaises(ModelError):
                self.model().propose(SOURCES, [])
            Stub.reply = completion(json.dumps([CANDIDATE, CANDIDATE, CANDIDATE]))
            with self.assertRaises(ModelError):
                self.model().propose(SOURCES, [])

    @needs_engine
    def test_transport_failures_are_closed_and_undetailed(self):
        with patch.dict(os.environ, {"STUB_KEY": "k"}):
            Stub.reply = (500, "application/json", b"{}")
            with self.assertRaisesRegex(ModelError, "check the server connection") as caught:
                self.model().propose(SOURCES, [])
            self.assertNotIn("127.0.0.1", str(caught.exception))
            Stub.reply = (200, "text/html", b"<html>")
            with self.assertRaisesRegex(ModelError, "HTTP 200 application/json"):
                self.model().propose(SOURCES, [])
            Stub.reply = (302, "application/json", b"{}")
            with self.assertRaisesRegex(ModelError, "redirects are not allowed"):
                self.model().propose(SOURCES, [])
            self.assertTrue(all(r["path"] == "/v1/chat/completions" for r in Stub.requests))
            Stub.reply = (200, "application/json", json.dumps({"choices": []}).encode())
            with self.assertRaisesRegex(ModelError, "choices"):
                self.model().propose(SOURCES, [])

    @needs_engine
    def test_refinement_carries_lineage(self):
        child = {**CANDIDATE, "parent_experiment_id": "experiment_abc", "question": "Does a longer window help?",
                 "expected_change": "Lower turnover with similar Sharpe."}
        Stub.reply = completion(json.dumps([child]))
        with patch.dict(os.environ, {"STUB_KEY": "k"}):
            model = self.model()
            model.bind_sources(SOURCES)
            refined = model.refine(CANDIDATE, {"id": "experiment_abc", "result": {"status": "ok"}}, [])
        self.assertEqual(refined[0]["parent_experiment_id"], "experiment_abc")
        self.assertEqual(json.loads(Stub.requests[-1]["body"]["messages"][1]["content"])["task"], "refine")


if __name__ == "__main__":
    unittest.main()

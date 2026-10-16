"""Offline storage boundaries for clean installs and retained private research."""
import errno
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from autoresearch.__main__ import main
from autoresearch.config import default_research_root, default_runtime_root, load_config
from autoresearch.controller import Controller
from autoresearch.records import ResearchStore
from qlib_yahoo.webapp.research_service import ResearchService


class ResearchStorageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.source = self.root / "checkout"
        self.source.mkdir()
        self.data = self.root / "external-data"
        self.env = patch.dict(os.environ, {"TUBBY_QLIB_DATA_ROOT": str(self.data)})
        self.env.start()
        self.addCleanup(self.env.stop)
        for target in ("autoresearch.models.AnthropicModel._generate",
                       "autoresearch.models.APIAgentModel._request",
                       "autoresearch.discovery.urlopen"):
            guard = patch(target, side_effect=AssertionError("unexpected model/network call"))
            guard.start()
            self.addCleanup(guard.stop)
        self.cfg = {"name": "Storage fixture", "backend": {},
                    "model": {"provider": "fixture", "hypotheses": []},
                    "discovery": {"network_enabled": False}}
        self.backend = SimpleNamespace(identity=lambda: {"fixture": "storage"})
        self.service = ResearchService(self.data, self.source)
        self.addCleanup(self.service.close)

    def controller(self, cfg=None):
        return Controller(cfg or self.cfg, backend=self.backend, discoverer=lambda _: [])

    def snapshot(self):
        return {str(p.relative_to(self.source)): p.read_bytes()
                for p in self.source.rglob("*") if p.is_file()}

    def campaign(self, root, name, campaign_id="campaign_one"):
        store = ResearchStore(root)
        store.put("campaigns", {"id": campaign_id, "config": {"name": name}})
        return store

    def reviewed_config(self):
        dataset = self.data / "datasets/sample"
        dataset.mkdir(parents=True)
        (dataset / "manifest.json").write_text("{}", encoding="utf-8")
        return {"name": "Reviewed storage", "backend": {
            "dataset": "sample", "universe": "TOP8", "cost_bps": 5,
            "development_start": "2023-01-01", "development_end": "2023-12-31",
            "validation_start": "2024-01-01", "validation_end": "2024-12-31", "warmup": 100},
            "model": {"provider": "reviewed", "model": "session-reviewed", "hypotheses": [{
                "title": "Synthetic plumbing", "family": "fixture", "mechanism": "Test storage",
                "expression": "rank(close)", "source_ids": ["source_fixture"],
                "evidence": [{"source_id": "source_fixture", "quote": "synthetic evidence"}],
                "expected_horizon": "one session", "required_fields": ["close"],
                "rationale": "Offline storage check"}], "refinements": []},
            "discovery": {"network_enabled": False, "arxiv_queries": [], "fixture_sources": [{
                "id": "source_fixture", "title": "Fixture", "url": "https://example.invalid/fixture",
                "text": "synthetic evidence for storage"}]}, "policy": {}, "limits": {}}

    def test_defaults_write_outside_checkout_without_private_inputs(self):
        controller = self.controller()
        self.assertEqual(controller.store.root, self.data / "autoresearch/records")
        self.assertEqual(controller.runtime, self.data / "autoresearch/runtime")
        self.assertFalse(controller.store.root.is_relative_to(self.source))
        controller.tick()
        self.assertTrue(controller.state_path.is_file())
        self.assertEqual(len(controller.store.list("campaigns")), 1)
        self.assertEqual(self.snapshot(), {})
        self.assertFalse((self.source / "research").exists())

    def test_defaults_use_current_qlib_default_when_environment_unset(self):
        with patch.dict(os.environ, {}, clear=True), patch("qlib_yahoo.default_data_root", return_value=str(self.data)) as default:
            self.assertEqual(default_research_root(), self.data / "autoresearch/records")
            self.assertEqual(default_runtime_root(), self.data / "autoresearch/runtime")
            self.assertEqual(default.call_count, 2)

    def test_explicit_config_paths_and_relative_resolution_unchanged(self):
        path = self.source / "campaign.json"
        path.write_text(json.dumps({**self.cfg, "research_root": "chosen-records",
                                    "runtime_root": "chosen-runtime"}), encoding="utf-8")
        cfg = load_config(path)
        controller = self.controller(cfg)
        self.assertEqual(controller.store.root, self.source / "chosen-records")
        self.assertEqual(controller.runtime, self.source / "chosen-runtime")
        absolute = self.controller({**self.cfg, "research_root": str(self.root / "records"),
                                    "runtime_root": str(self.root / "runtime")})
        self.assertEqual(absolute.store.root, self.root / "records")
        self.assertEqual(absolute.runtime, self.root / "runtime")

    def test_cli_report_and_reindex_defaults_and_explicit_paths(self):
        for explicit in (False, True):
            root = self.root / "chosen" if explicit else default_research_root()
            self.campaign(root, "CLI fixture")
            args = ["--research-root", str(root)] if explicit else []
            with self.subTest(explicit=explicit), patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(main(["report", *args]), 0)
                self.assertEqual(json.loads(output.getvalue())["campaigns"], 1)
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(main(["reindex", *args]), 0)
                index = Path(json.loads(output.getvalue())["index"])
                self.assertEqual(index.parent, root)
                self.assertTrue(index.exists())
        self.assertEqual(self.snapshot(), {})

    def test_clean_service_startup_does_not_create_or_require_private_files(self):
        with patch("subprocess.Popen", side_effect=AssertionError("unexpected worker")):
            self.assertEqual(self.service.overview()["campaigns"], [])
        self.assertFalse(self.data.exists())
        self.assertEqual(self.snapshot(), {})

    def test_legacy_external_and_demo_visible_with_read_only_legacy_discovery(self):
        self.campaign(self.source / "research/alphas", "Legacy", "campaign_legacy")
        self.campaign(self.service.records_root, "External", "campaign_external")
        self.campaign(self.service.demo_root / "research", "Demo", "campaign_demo")
        before = self.snapshot()
        with patch("autoresearch.records.ResearchStore", side_effect=AssertionError("UI must read directly")):
            campaigns = self.service.overview()["campaigns"]
            self.assertEqual({r["id"] for r in campaigns}, {"campaign_legacy", "campaign_external", "campaign_demo"})
            self.assertTrue(next(r for r in campaigns if r["id"] == "campaign_demo")["demo"])
            self.assertEqual(self.service.detail("campaign_legacy")["campaign"]["name"], "Legacy")
        self.assertEqual(self.snapshot(), before)

    def test_duplicate_id_prefers_external_for_overview_detail_and_series(self):
        legacy = self.campaign(self.source / "research/alphas", "Legacy")
        legacy.put("experiments", {"id": "exp_legacy", "campaign_id": "campaign_one", "result": {"status": "error"}})
        external = self.campaign(self.service.records_root, "External")
        external.put("experiments", {"id": "exp_external", "campaign_id": "campaign_one", "result": {"status": "error"}})
        self.campaign(self.service.demo_root / "research", "Demo duplicate")
        before = self.snapshot()
        campaigns = self.service.overview()["campaigns"]
        self.assertEqual(len(campaigns), 1)
        self.assertEqual(campaigns[0]["name"], "External")
        detail = self.service.detail("campaign_one")
        self.assertEqual(detail["campaign"]["name"], "External")
        self.assertEqual([r["id"] for r in detail["experiments"]], ["exp_external"])
        self.assertEqual(self.service.series("campaign_one", "exp_external")["status"], "failed")
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self.service.series("campaign_one", "exp_legacy")
        self.assertEqual(self.snapshot(), before)

    def test_new_start_writes_only_managed_external_records_and_runtime(self):
        self.campaign(self.source / "research/alphas", "Legacy", "campaign_legacy")
        before = self.snapshot()
        saved = self.service.save(self.reviewed_config())
        def offline_worker(path, *args, **kwargs):
            cfg = load_config(path)
            self.assertEqual(Path(cfg["research_root"]), self.service.records_root)
            self.assertEqual(Path(cfg["runtime_root"]), self.service.runtime_root / saved["id"])
            self.assertEqual(cfg["model"]["provider"], "fixture")
            self.assertFalse(cfg["discovery"]["network_enabled"])
            self.controller(cfg).tick()
            return {"status": "stopped"}
        with patch.object(self.service, "_spawn", side_effect=offline_worker):
            self.service.start({"id": saved["id"]})
        self.assertEqual(len(ResearchStore(self.service.records_root).list("campaigns")), 1)
        self.assertEqual(len(list((self.service.runtime_root / saved["id"]).glob("campaign_*.json"))), 1)
        self.assertEqual(len(self.service.overview()["campaigns"]), 2)
        self.assertEqual(self.snapshot(), before)

    def test_source_root_remains_worker_cwd(self):
        with patch("subprocess.Popen") as spawn, patch("threading.Thread"):
            self.service._spawn(self.data / "campaign.json", "config_one", False, False, reviewed=True)
            self.assertEqual(spawn.call_args.kwargs["cwd"], self.source)
            self.assertNotIn("--allow-model", spawn.call_args.args[0])
        self.service._process = None

    def test_symlink_records_escape_rejected_before_start_or_read(self):
        saved = self.service.save(self.reviewed_config())
        outside = self.root / "outside"
        outside.mkdir()
        try:
            self.service.records_root.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            if exc.errno in (errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP) or getattr(exc, "winerror", None) == 1314:
                self.skipTest("symlink creation unavailable")
            raise
        with patch.object(self.service, "_spawn") as spawn:
            with self.assertRaises(ValueError):
                self.service.start({"id": saved["id"]})
            spawn.assert_not_called()
        with self.assertRaises(ValueError):
            self.service.overview()
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

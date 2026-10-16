"""Offline HTTP contract tests: mock agent only, no external model calls."""
import copy
import importlib.util
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from autoresearch.agents import load_agents
from autoresearch.controller import _public_config, Controller
from autoresearch.models import AGENT_CONTRACT_VERSION, APIAgentModel, ModelError
from qlib_yahoo.webapp.research_service import ResearchService
from test_autoresearch_models import IDEA, SOURCES
import test_autoresearch_controller as controller_fixtures


class AgentHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.server.requests.append(('GET', self.path, self.headers.get('Authorization'), None))
        value = dict(contract_version=AGENT_CONTRACT_VERSION, status='ready', model='mock-model', provider_name='Mock local')
        value.update(self.server.health_changes)
        self.respond(value)
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(('POST', self.path, self.headers.get('Authorization'), data))
        idea = copy.deepcopy(IDEA)
        if data['task'] == 'refine':
            idea.update(parent_experiment_id=data['experiment']['id'], question='Does a change help?', expected_change='Improved validation')
        value = dict(contract_version=AGENT_CONTRACT_VERSION, request_id=data['request_id'], candidates=[idea])
        value.update(self.server.changes)
        self.respond(value)
    def respond(self, value):
        if self.server.redirect:
            self.send_response(307); self.send_header('Location', self.server.redirect); self.end_headers(); return
        body = self.server.raw if self.server.raw is not None else json.dumps(value).encode()
        self.send_response(200); self.send_header('Content-Type', self.server.content_type)
        self.end_headers(); self.wfile.write(body)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), AgentHandler)
        self.server.requests = []; self.server.changes = {}; self.server.health_changes = {}
        self.server.redirect = None; self.server.raw = None; self.server.content_type = 'application/json'
        thread = threading.Thread(target=self.server.serve_forever, daemon=True); thread.start()
        self.addCleanup(self.server.server_close); self.addCleanup(self.server.shutdown)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = dict(provider='agent', agent_id='mock', provider_name='Mock local', model='mock-model',
                        endpoint='http://127.0.0.1:' + str(self.server.server_port), api_key_env='CUSTOM_AUTH')
        environment = {'CUSTOM_AUTH': 'private-mock-token'}
        # urllib constructs its default HTTPS handler even for loopback HTTP.
        # Windows certificate initialization needs SystemRoot in a fresh process;
        # keep that OS path while still clearing all real credentials/proxy settings.
        if os.name == 'nt' and os.environ.get('SystemRoot'):
            environment['SystemRoot'] = os.environ['SystemRoot']
        self.env = patch.dict(os.environ, environment, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)
        self.catalog = self.root / 'autoresearch/agents.local.json'; self.catalog.parent.mkdir()
        row = {k: v for k, v in self.cfg.items() if k not in {'provider', 'agent_id'}}
        row.update(id='mock', name='My local researcher')
        self.catalog.write_text(json.dumps({'agents': [row]}))
        self.service = ResearchService(self.root, self.root / 'source'); self.addCleanup(self.service.close)
        (self.root / 'datasets/sample').mkdir(parents=True); (self.root / 'datasets/sample/manifest.json').write_text('{}')
        self.campaign = dict(name='API test', backend=dict(dataset='sample', universe='TOP8', cost_bps=5,
                             development_start='2023-01-01', development_end='2023-12-31',
                             validation_start='2024-01-01', validation_end='2024-12-31'),
                             model=dict(provider='agent', agent_id='mock', model='mock-model'),
                             discovery=dict(arxiv_queries=['all:momentum']), limits={})

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"),
                         "requires local numpy and pandas")
    def test_lazy_config_health_propose_and_refine(self):
        model = APIAgentModel(self.cfg)
        self.assertEqual(self.server.requests, [])
        model.check_connection()
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.server.requests[0][:3], ('GET', '/health', 'Bearer private-mock-token'))
        proposed = model.propose(SOURCES, [{'decision': 'reject', 'reason': 'already tried'}])
        self.assertEqual(proposed[0]['provenance']['provider_name'], 'Mock local')
        self.assertNotIn('endpoint', proposed[0]['provenance'])
        data = self.server.requests[-1][3]
        self.assertEqual(data['constraints']['max_output_tokens'], 4096)
        self.assertIn('ts_corr', data['constraints']['operator_names'])
        self.assertEqual(data['history'][0]['decision'], 'reject')
        self.assertEqual(model.refine(IDEA, {'id': 'parent-one'}, [])[0]['parent_experiment_id'], 'parent-one')
        self.assertEqual(self.server.requests[-1][3]['task'], 'refine')
        self.assertEqual(len(self.server.requests), 3)

    def test_health_never_generates_and_catalog_reads_are_offline(self):
        saved = self.service.save(self.campaign)
        overview = self.service.overview()
        self.assertEqual(self.server.requests, [])
        connection = next(a for a in overview['connections']['agents'] if a.get('agent_id') == 'mock')
        self.assertEqual(connection['state'], 'configured')
        checked = self.service.check_agent({'agent_id': 'mock'})
        self.assertEqual(checked['state'], 'connected')
        self.assertTrue(all(r[0] == 'GET' for r in self.server.requests))
        public = json.dumps(self.service.overview())
        for secret in ('private-mock-token', 'CUSTOM_AUTH', self.cfg['endpoint']): self.assertNotIn(secret, public)
        self.assertNotIn('endpoint', self.service.config(saved['id'])['model'])
        self.assertIn('[redacted]', self.service._clean('Error private-mock-token'))
        self.server.health_changes['model'] = 'wrong-model'
        self.assertEqual(self.service.check_agent({'agent_id': 'mock'})['state'], 'unavailable')

    def test_missing_credentials_and_removed_or_changed_agent(self):
        saved = self.service.save(self.campaign)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self.service.overview()['configs'][0]['ready'])
            self.assertEqual(self.service.check_agent({'agent_id': 'mock'})['state'], 'missing_credential')
            with self.assertRaises(ModelError): APIAgentModel(self.cfg).check_connection()
        self.assertEqual(self.server.requests, [])
        catalog = json.loads(self.catalog.read_text()); catalog['agents'][0]['model'] = 'new-model'
        self.catalog.write_text(json.dumps(catalog))
        self.assertIn('model changed', ' '.join(self.service.overview()['configs'][0]['issues']))
        self.catalog.unlink()
        self.assertIn('no longer configured', ' '.join(self.service.overview()['configs'][0]['issues']))
        self.assertEqual(self.service.config(saved['id'])['model']['model'], 'mock-model')

    def test_browser_cannot_inject_endpoint_or_credentials_and_start_needs_consent(self):
        for field in ('endpoint', 'api_key_env', 'api_key', 'max_tokens', 'provider_name'):
            cfg = copy.deepcopy(self.campaign); cfg['model'][field] = 'injected'
            with self.subTest(field=field), self.assertRaises(ValueError): self.service.save(cfg)
        saved = self.service.save(self.campaign)
        with patch('subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'allow_model'): self.service.start({'id': saved['id']})
            spawn.assert_not_called()
        with patch.object(self.service, '_spawn', return_value={'status': 'mock'}):
            self.service.start({'id': saved['id'], 'allow_model': True})
        runtime = json.loads((self.service.runtime_root / saved['id'] / 'campaign.json').read_text())
        self.assertEqual(runtime['model']['endpoint'], self.cfg['endpoint'])
        self.assertNotIn('private-mock-token', json.dumps(runtime))
        public = _public_config(runtime)
        self.assertNotIn('endpoint', public['model']); self.assertNotIn('api_key_env', public['model'])
        self.assertIn('endpoint_sha256', public['model'])

    def test_catalog_validation_and_health_cache_revision(self):
        self.service.check_agent({'agent_id': 'mock'})
        catalog = json.loads(self.catalog.read_text()); catalog['agents'][0]['revision'] = '2'
        self.catalog.write_text(json.dumps(catalog))
        self.assertEqual(self.service.agent_catalog()['agents'][-1]['state'], 'configured')
        with patch('qlib_yahoo.webapp.research_service.time.monotonic', return_value=1e20):
            self.assertEqual(self.service.agent_catalog()['agents'][-1]['state'], 'configured')
        for contents in ({'agents': [{}]}, {'agents': [catalog['agents'][0], catalog['agents'][0]]}, {'agents': 'wrong'}):
            self.catalog.write_text(json.dumps(contents))
            with self.assertRaises(ModelError): load_agents(self.root)
            self.assertTrue(self.service.agent_catalog()['issues'])

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"),
                         "requires local numpy and pandas")
    def test_bad_response_citations_bounds_and_redirect_do_not_retry(self):
        model = APIAgentModel(self.cfg)
        for changes in ({'request_id': 'wrong'}, {'contract_version': 'wrong'}, {'extra': 'wrong'},
                        {'candidates': [IDEA] * 4}, {'candidates': [{**IDEA, 'source_ids': ['invented']}]},
                        {'candidates': [{**IDEA, 'evidence': [{'source_id': 'source-a', 'quote': 'invented'}]}]}):
            self.server.changes = changes
            before = len(self.server.requests)
            with self.subTest(changes=changes), self.assertRaises(ModelError): model.propose(SOURCES, [])
            self.assertEqual(len(self.server.requests), before + 1)
        self.server.changes = {}; self.server.raw = b'x' * 2_000_001
        with self.assertRaisesRegex(ModelError, 'byte limit'): model.propose(SOURCES, [])
        self.server.raw = b'[]'; self.server.content_type = 'text/plain'
        with self.assertRaises(ModelError): model.propose(SOURCES, [])
        self.server.raw = None; self.server.content_type = 'application/json'; self.server.redirect = self.cfg['endpoint'] + '/other'
        before = len(self.server.requests)
        with self.assertRaises(ModelError): model.propose(SOURCES, [])
        self.assertEqual(len(self.server.requests), before + 1)

    def test_unsafe_config_and_secret_rotation_identity(self):
        model = APIAgentModel(self.cfg)
        with patch.dict(os.environ, {'CUSTOM_AUTH': 'rotated-private-token'}):
            self.assertEqual(model.fingerprint(), APIAgentModel(self.cfg).fingerprint())
        self.assertNotEqual(model.fingerprint(), APIAgentModel({**self.cfg, 'revision': '2'}).fingerprint())
        for endpoint in ('file:///etc/passwd', 'http://example.com', 'https://name:secret@example.com', 'https://example.com?q=key', 'https://example.com/#x'):
            with self.subTest(endpoint=endpoint), self.assertRaises(ModelError): APIAgentModel({**self.cfg, 'endpoint': endpoint})
        for changes in ({'timeout_seconds': float('nan')}, {'timeout_seconds': True}, {'max_tokens': 0}, {'max_candidates': 21}):
            with self.subTest(changes=changes), self.assertRaises(ModelError): APIAgentModel({**self.cfg, **changes})

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"),
                         "requires local numpy and pandas")
    def test_controller_retains_api_failure_budget_and_duplicate_result(self):
        cfg = copy.deepcopy(self.campaign)
        cfg.update(research_root=str(self.root / 'evidence'), runtime_root=str(self.root / 'runtime'))
        cfg['model'] = self.cfg
        cfg['discovery'] = {'fixture_sources': SOURCES, 'network_enabled': False}
        backend = controller_fixtures.FakeBackend()
        first = Controller(cfg, backend=backend)
        controller_fixtures.ControllerTests.drain(self, first)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(first.store.list('reviews')[0]['decision'], 'reject')
        second = Controller({**cfg, 'name': 'another campaign'}, backend=backend)
        controller_fixtures.ControllerTests.drain(self, second)
        self.assertEqual(len(backend.calls), 1)
        self.assertTrue(any(e['kind'] == 'duplicate_experiment' for e in second.store.list('events')))
        self.server.changes = {'request_id': 'invalid'}
        failed = Controller({**cfg, 'name': 'failed', 'runtime_root': str(self.root / 'failed')}, backend=backend)
        failed.tick()
        self.assertEqual(failed.state['model_calls'], 1)
        self.assertTrue(any(e['kind'] == 'discovery_error' for e in failed.store.list('events')))

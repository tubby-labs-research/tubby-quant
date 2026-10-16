import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from autoresearch.models import ModelError, make_model, validate_candidates

SOURCES = [{'id': 'source-a', 'title': 'Evidence', 'url': 'fixture://evidence',
            'text': 'High volume may predict short term reversal.', 'evidence_level': 'fixture'}]
IDEA = {'title': 'Volume reversal', 'mechanism': 'Temporary pressure reverses', 'family': 'reversal',
        'expression': '-returns * rank(volume)', 'source_ids': ['source-a'],
        'evidence': [{'source_id': 'source-a', 'quote': 'High volume may predict short term reversal.'}],
        'expected_horizon': 'one to five days', 'required_fields': ['returns', 'volume'],
        'rationale': 'Test whether pressure fades next day'}


class ModelTests(unittest.TestCase):
    def test_research_focus_reaches_proposal_and_refinement_and_changes_identity(self):
        model = make_model({'provider': 'fixture', 'research_focus': 'Does volume pressure reverse?'})
        with patch.object(model, '_generate', return_value=[]) as generate:
            model.propose(SOURCES, [])
            self.assertEqual(generate.call_args.args[1]['research_focus'], model.research_focus)
            model.refine(IDEA, {'id': 'experiment-one'}, [])
            self.assertEqual(generate.call_args.args[1]['research_focus'], model.research_focus)
        self.assertNotEqual(model.fingerprint(), make_model({'provider': 'fixture'}).fingerprint())
        for value in (None, {}, 'x' * 2001):
            with self.subTest(value=type(value)), self.assertRaises(ModelError):
                make_model({'provider': 'fixture', 'research_focus': value})

    def test_fixture_proposal_is_grounded_and_deterministic(self):
        model = make_model({'provider': 'fixture', 'hypotheses': [IDEA]})
        first = model.propose(SOURCES, [])
        self.assertEqual(first, model.propose(SOURCES, []))
        self.assertTrue(first[0]['provenance']['fixture'])
        self.assertIn('prompt_schema_version', first[0]['provenance'])
        self.assertEqual(model.fingerprint(), make_model(model.config).fingerprint())
        self.assertNotIn('provenance', IDEA)

    def test_invalid_citations_and_quotes(self):
        for changes in ({'source_ids': ['unknown']}, {'evidence': [{'source_id': 'source-a', 'quote': 'invented'}]},
                        {'evidence': [{'source_id': 'source-a', 'quote': ''}]}):
            with self.subTest(changes=changes), self.assertRaises(ModelError):
                validate_candidates([{**IDEA, **changes}], SOURCES)

    def test_price_volume_contract_including_expression(self):
        for fields, expression in [(['cap'], 'cap'), (['returns', 'volume'], 'cap * returns'),
                                   (['returns'], '-returns * rank(volume)')]:
            with self.subTest(expression=expression), self.assertRaises(ModelError):
                validate_candidates([{**IDEA, 'required_fields': fields, 'expression': expression}], SOURCES)
        local = {**IDEA, 'expression': 'x = rank(volume); -returns * x'}
        self.assertEqual(len(validate_candidates([local], SOURCES)), 1)

    def test_proposal_count_is_a_hard_bound(self):
        model = make_model({'provider': 'fixture', 'max_candidates': 1, 'hypotheses': [IDEA, IDEA]})
        with self.assertRaisesRegex(ModelError, 'max_candidates'):
            model.propose(SOURCES, [])

    def test_builtin_market_group_requires_no_classification_field(self):
        candidate = {**IDEA, 'expression': 'group_neutralize(returns, market)', 'required_fields': ['returns']}
        self.assertEqual(len(validate_candidates([candidate], SOURCES)), 1)
        for group in ('sector', 'industry', 'country', 'invented'):
            with self.subTest(group=group), self.assertRaises(ModelError):
                validate_candidates([{**candidate, 'expression': f'group_neutralize(returns, {group})'}], SOURCES)

    def test_refinement_requires_lineage_and_testable_question(self):
        refined = {**IDEA, 'parent_experiment_id': 'experiment-one', 'question': 'Does slower reversal improve IC?',
                   'expected_change': 'Higher validation IC'}
        model = make_model({'provider': 'fixture', 'refinements': [refined]})
        model.bind_sources(SOURCES)
        self.assertEqual(model.refine(IDEA, {'id': 'experiment-one'}, [])[0]['parent_experiment_id'], 'experiment-one')
        with self.assertRaises(ModelError):
            model.refine(IDEA, {'id': 'experiment-two'}, [])
        del refined['question']
        model = make_model({'provider': 'fixture', 'refinements': [refined]})
        model.bind_sources(SOURCES)
        with self.assertRaises(ModelError):
            model.refine(IDEA, {'id': 'experiment-one'}, [])

    def test_fixture_refinement_parent_template(self):
        template = {**IDEA, 'parent_experiment_id': '$parent', 'question': 'Does slower reversal improve IC?',
                    'expected_change': 'Higher validation IC'}
        model = make_model({'provider': 'fixture', 'refinements': [template]})
        model.bind_sources(SOURCES)
        for parent in ('dynamic-one', 'dynamic-two'):
            self.assertEqual(model.refine(IDEA, {'id': parent}, [])[0]['parent_experiment_id'], parent)
        self.assertEqual(template['parent_experiment_id'], '$parent')
        self.assertEqual(model.config['refinements'][0]['parent_experiment_id'], '$parent')

    def test_anthropic_is_lazy_requires_explicit_model_and_env(self):
        with self.assertRaises(ModelError):
            make_model({'provider': 'anthropic'})
        with self.assertRaises(ModelError):
            make_model({'provider': 'anthropic', 'model': 'test', 'api_key': 'example_do_not_write'})
        with patch.dict(os.environ, {}, clear=True), patch('builtins.__import__', wraps=__import__) as imports:
            model = make_model({'provider': 'anthropic', 'model': 'test', 'api_key_env': 'TEST_SECRET'})
            self.assertFalse(any(call.args[0] == 'anthropic' for call in imports.call_args_list))
            with self.assertRaisesRegex(ModelError, 'environment variable is unset'):
                model.propose(SOURCES, [])

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"),
                         "requires local numpy and pandas")
    def test_sdk_request_and_output_validation(self):
        sdk = MagicMock()
        sdk.Anthropic.return_value.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type='text', text=json.dumps([IDEA]))])
        with patch.dict('sys.modules', {'anthropic': sdk}), patch.dict(os.environ, {'TEST_SECRET': 'example_private_value'}):
            model = make_model({'provider': 'anthropic', 'model': 'explicit-test-model', 'api_key_env': 'TEST_SECRET'})
            result = model.propose(SOURCES, [{'decision': 'reject', 'reason': 'Failed validation'}])
            self.assertEqual(result[0]['provenance']['model'], 'explicit-test-model')
            request = sdk.Anthropic.return_value.messages.create.call_args.kwargs
            self.assertIn('untrusted DATA', request['system'])
            self.assertIn('Failed validation', request['messages'][0]['content'])
            prompt_data = json.loads(request['messages'][0]['content'])
            self.assertIn('rank', prompt_data['operator_names'])
            self.assertIn('ts_corr', prompt_data['operator_names'])
            self.assertIn('FASTEXPR grammar', request['system'])
            self.assertNotIn('example_private_value', model.fingerprint())
            sdk.Anthropic.assert_called_once_with(api_key='example_private_value', timeout=60, max_retries=0)
            sdk.Anthropic.return_value.messages.create.return_value.content[0].text = '```json\n[]\n```'
            with self.assertRaisesRegex(ModelError, 'required JSON'):
                model.propose(SOURCES, [])

    def test_file_fixture_no_credentials_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            path.write_text(json.dumps({'hypotheses': [IDEA]}))
            before = path.read_bytes()
            model = make_model({'provider': 'fixture', 'fixture_path': str(path)})
            self.assertEqual(len(model.propose(SOURCES, [])), 1)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()

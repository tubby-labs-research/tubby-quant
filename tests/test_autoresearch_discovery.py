import unittest
import os
import ssl
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from autoresearch.discovery import DiscoveryError, discover, parse_arxiv

ATOM = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2401.00001v1</id><title>Volume and prices</title>
<summary>High volume may predict short term reversal.</summary>
</entry><entry><id>http://arxiv.org/abs/2401.00001v2</id>
<title>Volume and prices</title><summary>High volume may predict short term reversal.</summary>
</entry></feed>'''


class DiscoveryTests(unittest.TestCase):
    def test_atom_version_identity_and_abstract(self):
        first = parse_arxiv(ATOM, '2026-01-01T00:00:00+00:00')
        second = parse_arxiv(ATOM, '2026-01-02T00:00:00+00:00')
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]['id'], second[0]['id'])
        self.assertNotEqual(first[0]['retrieved_at'], second[0]['retrieved_at'])
        self.assertEqual(first[0]['url'], 'https://arxiv.org/abs/2401.00001')
        self.assertEqual(first[0]['evidence_level'], 'abstract')
        changed = parse_arxiv(ATOM.replace(b'High volume', b'Low volume'))
        self.assertNotEqual(first[0]['id'], changed[0]['id'])

    def test_network_is_explicit(self):
        with patch('autoresearch.discovery.urlopen') as transport:
            with self.assertRaises(DiscoveryError):
                discover({'arxiv_queries': ['all:volume']})
            transport.assert_not_called()

    def test_real_atom_with_mocked_http(self):
        response = MagicMock()
        response.geturl.return_value = 'https://export.arxiv.org/api/query'
        response.read.return_value = ATOM
        response.__enter__.return_value = response
        with patch('autoresearch.discovery.urlopen', return_value=response) as transport:
            result = discover({'network_enabled': True, 'arxiv_queries': ['all:volume'],
                               'timeout_seconds': 3, 'max_response_bytes': 1000})
        self.assertEqual(len(result), 1)
        self.assertEqual(transport.call_args.kwargs['timeout'], 3)
        context = transport.call_args.kwargs['context']
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertIn('search_query=all%3Avolume', transport.call_args.args[0].full_url)
        response.read.assert_called_once_with(1001)

    def test_failed_retrieval_is_error_not_generated_evidence(self):
        with patch('autoresearch.discovery.urlopen', side_effect=OSError('offline')):
            with self.assertRaisesRegex(DiscoveryError, 'retrieval failed'):
                discover({'network_enabled': True, 'arxiv_queries': ['all:volume']})

    def test_response_limit_and_empty_feed(self):
        with patch('autoresearch.discovery._fetch', return_value=b'<feed xmlns="http://www.w3.org/2005/Atom"/>'):
            self.assertEqual(discover({'network_enabled': True, 'arxiv_queries': ['all:volume']}), [])
        response = MagicMock()
        response.geturl.return_value = 'https://export.arxiv.org/api/query'
        response.read.return_value = b'x' * 11
        response.__enter__.return_value = response
        with patch('autoresearch.discovery.urlopen', return_value=response):
            with self.assertRaisesRegex(DiscoveryError, 'byte limit'):
                discover({'network_enabled': True, 'arxiv_queries': ['all:volume'], 'max_response_bytes': 10})

    def test_fixture_status_and_safe_id(self):
        fixture = {'id': 'fixture-one', 'title': 'Demo', 'url': 'fixture://demo', 'text': 'Demo evidence'}
        with patch('autoresearch.discovery.urlopen') as transport:
            source = discover({'fixture_sources': [fixture]})[0]
        self.assertEqual(source['id'], fixture['id'])
        self.assertEqual(source['evidence_level'], 'fixture')
        transport.assert_not_called()
        with self.assertRaises(DiscoveryError):
            discover({'fixture_sources': [{**fixture, 'id': '../escape'}]})

    def test_consecutive_queries_are_politely_spaced(self):
        with patch('autoresearch.discovery._fetch', return_value=ATOM) as transport, \
                patch('autoresearch.discovery.time.sleep') as sleeper:
            discover({'network_enabled': True, 'arxiv_queries': ['all:volume', 'all:price', 'all:reversal']})
            self.assertEqual(transport.call_count, 3)
            self.assertEqual([call.args[0] for call in sleeper.call_args_list], [3, 3])
        with self.assertRaisesRegex(DiscoveryError, 'query_interval_seconds'):
            discover({'query_interval_seconds': 2})

    def test_https_ca_selection_preserves_verification(self):
        response = MagicMock()
        response.geturl.return_value = 'https://export.arxiv.org/api/query'
        response.read.return_value = ATOM
        response.__enter__.return_value = response
        config = {'network_enabled': True, 'arxiv_queries': ['all:volume']}
        verified_context = ssl.create_default_context()
        with patch.dict(os.environ, {'SSL_CERT_FILE': '/configured/ca.pem'}), \
                patch('autoresearch.discovery.ssl.create_default_context', return_value=verified_context) as factory, \
                patch('autoresearch.discovery.urlopen', return_value=response) as transport:
            discover(config)
            factory.assert_called_once_with(cafile='/configured/ca.pem')
            self.assertIs(transport.call_args.kwargs['context'], verified_context)
        with patch.dict(os.environ, {}, clear=True), \
                patch.dict('sys.modules', {'certifi': SimpleNamespace(where=lambda: '/certifi/ca.pem')}), \
                patch('autoresearch.discovery.ssl.create_default_context', return_value=verified_context) as factory, \
                patch('autoresearch.discovery.urlopen', return_value=response):
            discover(config)
            factory.assert_called_once_with(cafile='/certifi/ca.pem')
        with patch.dict(os.environ, {}, clear=True), patch.dict('sys.modules', {'certifi': None}), \
                patch('autoresearch.discovery.ssl.create_default_context', return_value=verified_context) as factory, \
                patch('autoresearch.discovery.urlopen', return_value=response):
            discover(config)
            factory.assert_called_once_with()
        self.assertEqual(verified_context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(verified_context.check_hostname)

    def test_malformed_xml(self):
        with self.assertRaises(DiscoveryError):
            parse_arxiv(b'<html>Unexpected</html>')


if __name__ == '__main__':
    unittest.main()

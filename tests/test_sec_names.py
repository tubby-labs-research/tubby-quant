"""Offline naming contract; no catalogs, databases or dataset materialization."""
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo import sec_names as names
from qlib_yahoo import sec_registry as registry


def recipe(tag='Assets', **changes):
    return dict(dict(tag=tag, taxonomy='us-gaap/', unit='USD', qtrs=0,
                     segments='', coreg=''), **changes)


class SecNameTests(unittest.TestCase):
    def assert_roundtrip(self, spec, alias):
        self.assertEqual(names.preferred_name(spec), alias)
        self.assertEqual(alias, alias.lower())
        candidate = names.spec_for_name(alias)
        self.assertIsNotNone(candidate)
        # Catalog lookup restores the real capitalization; the parser's generic
        # PascalCase candidate is not an exact-tag identity guarantee.
        self.assertEqual(candidate['tag'].lower(), spec['tag'].lower())
        self.assertEqual(dict(candidate, tag=spec['tag']), spec)
        self.assertTrue(names.accepts_name(alias, spec))
        self.assertTrue(names.recognized_alias(alias))

    def test_five_common_aliases_and_storage_definitions_unchanged(self):
        expected = {
            'sec_assets': recipe(),
            'sec_inventory': recipe('InventoryNet'),
            'sec_liabilities': recipe('Liabilities'),
            'sec_net_income_q': recipe('NetIncomeLoss', qtrs=1),
            'sec_net_income_4q': recipe('NetIncomeLoss', qtrs=4),
        }
        self.assertEqual(registry.COMMON_FIELDS, expected)
        self.assertEqual(registry.REGISTRY_VERSION, 'sec-dictionary-2')
        for alias, spec in expected.items():
            with self.subTest(alias=alias):
                self.assert_roundtrip(spec, alias)
                self.assertEqual(names.spec_for_name(alias), spec)
                self.assertEqual(names.curated_defs()[alias], spec)
                self.assertEqual(registry._name_for_recipe(spec), alias)
        eur = recipe(unit='EUR')
        old_name = registry._name_for_recipe(eur)
        self.assertRegex(old_name, r'^sec_assets_us_gaap_eur_family_q0_[0-9a-f]{20}$')
        self.assertEqual(names.preferred_name(eur), 'sec_assets__eur')
        self.assertEqual(registry._name_for_recipe(eur), old_name)

    def test_curated_map_exact_and_bounded(self):
        expected = {
            'Assets': 'assets', 'AssetsCurrent': 'assets_curr',
            'InventoryNet': 'inventory', 'Liabilities': 'liabilities',
            'LiabilitiesCurrent': 'liabilities_curr',
            'CashAndCashEquivalentsAtCarryingValue': 'cash_equivalents',
            'StockholdersEquity': 'equity', 'NetIncomeLoss': 'net_income',
            'NetCashProvidedByUsedInOperatingActivities': 'cashflow_op',
            'NetCashProvidedByUsedInInvestingActivities': 'cashflow_investing',
            'NetCashProvidedByUsedInFinancingActivities': 'cashflow_financing',
            'OperatingIncomeLoss': 'operating_income', 'GrossProfit': 'gross_profit',
            'Revenues': 'revenues', 'ResearchAndDevelopmentExpense': 'rd_expense',
            'SellingGeneralAndAdministrativeExpense': 'sga_expense',
            'IncomeTaxExpenseBenefit': 'income_tax',
            'EarningsPerShareBasic': 'eps_basic', 'EarningsPerShareDiluted': 'eps_diluted',
            'RetainedEarningsAccumulatedDeficit': 'retained_earnings',
            'AccountsPayableCurrent': 'accounts_payable_curr',
            'LongTermDebtCurrent': 'debt_lt_current',
            'LongTermDebtNoncurrent': 'debt_lt_noncurrent',
        }
        self.assertEqual(dict(names.CURATED_NAMES), expected)
        self.assertLessEqual(len(expected), 25)
        self.assertEqual(len(set(expected.values())), len(expected))
        with self.assertRaises(TypeError):
            names.CURATED_NAMES['Assets'] = 'other'

    def test_defaults_explicit_period_and_unit(self):
        instant = {'Assets', 'AssetsCurrent', 'InventoryNet', 'Liabilities',
                   'LiabilitiesCurrent', 'CashAndCashEquivalentsAtCarryingValue',
                   'StockholdersEquity', 'RetainedEarningsAccumulatedDeficit',
                   'AccountsPayableCurrent', 'LongTermDebtCurrent', 'LongTermDebtNoncurrent'}
        definitions = names.curated_defs()
        self.assertEqual(len(definitions), 24)
        for alias, spec in definitions.items():
            with self.subTest(alias=alias):
                qtrs = 4 if alias == 'sec_net_income_4q' else (0 if spec['tag'] in instant else 1)
                self.assertEqual(spec['qtrs'], qtrs)
                self.assertEqual(spec['unit'], 'USD/shares' if spec['tag'].startswith('EarningsPerShare') else 'USD')
                self.assert_roundtrip(spec, alias)
        self.assertEqual(definitions['sec_eps_basic_q__usd_per_share']['tag'], 'EarningsPerShareBasic')
        self.assert_roundtrip(recipe('EarningsPerShareBasic', unit='shares', qtrs=2),
                              'sec_eps_basic_2q__shares')
        self.assert_roundtrip(recipe('EarningsPerShareBasic', unit='USD', qtrs=0), 'sec_eps_basic')

    def test_curated_definitions_are_defensive_copies(self):
        first = names.curated_defs()
        first['sec_assets']['tag'] = 'Other'
        first.pop('sec_inventory')
        self.assertEqual(names.curated_defs()['sec_assets'], recipe())
        self.assertIn('sec_inventory', names.curated_defs())

    def test_generic_alternates_for_every_curated_tag(self):
        for tag in names.CURATED_NAMES:
            for qtrs, suffix in ((0, ''), (1, '_q'), (2, '_2q'), (4, '_4q')):
                with self.subTest(tag=tag, qtrs=qtrs):
                    spec = recipe(tag, qtrs=qtrs)
                    alternate = 'sec_' + names.display_label(tag).lower().replace(' ', '_') + suffix
                    self.assertEqual(names.generic_alias_for_spec(spec), alternate)
                    self.assertEqual(names.spec_for_name(alternate), spec)
                    self.assertTrue(names.accepts_name(alternate, spec))
        self.assertEqual(names.spec_for_name('sec_assets_current'), recipe('AssetsCurrent'))
        self.assertEqual(names.spec_for_name('sec_assets_curr'), recipe('AssetsCurrent'))

    def test_long_tag_words_are_preserved_without_truncation_or_hash(self):
        for tag, base in (
                ('AccountsReceivableNetCurrent', 'accounts_receivable_net_current'),
                ('PaymentsToAcquirePropertyPlantAndEquipment', 'payments_to_acquire_property_plant_and_equipment'),
                ('OriginalSEC' + 'VeryLongEnglishTag' * 30,
                 'original_sec' + '_very_long_english_tag' * 30)):
            with self.subTest(tag=tag):
                self.assert_roundtrip(recipe(tag), 'sec_' + base)
                self.assertEqual(base.replace('_', ''), tag.lower())

    def test_all_durations_and_units_roundtrip_without_collisions(self):
        seen = {}
        for tag in ('Assets', 'AssetsCurrent', 'AssetsQuarter', 'Assets4Quarter',
                    'AccountsReceivableNetCurrent', 'Revenues'):
            for qtrs in range(101):
                suffix = '' if qtrs == 0 else ('_q' if qtrs == 1 else f'_{qtrs}q')
                for unit, unit_suffix in (('USD', ''), ('shares', '__shares'),
                                          ('USD/shares', '__usd_per_share'), ('EUR', '__eur')):
                    spec = recipe(tag, qtrs=qtrs, unit=unit)
                    base = names.CURATED_NAMES.get(tag, names.display_label(tag).lower().replace(' ', '_'))
                    alias = 'sec_' + base + suffix + unit_suffix
                    self.assert_roundtrip(spec, alias)
                    self.assertNotIn(alias, seen)
                    seen[alias] = spec
        self.assertEqual(names.preferred_name(recipe(qtrs=4)), 'sec_assets_4q')
        self.assertNotIn('ttm', names.preferred_name(recipe(qtrs=4)))

    def test_supported_currency_suffixes_are_explicit_reversible(self):
        for unit in ('EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'HKD',
                     'NZD', 'SEK', 'NOK', 'DKK', 'SGD', 'INR', 'KRW', 'BRL',
                     'MXN', 'ZAR', 'TWD'):
            self.assert_roundtrip(recipe(unit=unit), 'sec_assets__' + unit.lower())

    def test_taxonomy_identity_and_fixed_suffix_order(self):
        cases = [('us-gaap/2025', 'us_gaap_2025'), ('ifrs-full/', 'ifrs_full'),
                 ('ifrs-full/2025', 'ifrs_full_2025'), ('ifrs/', 'ifrs'),
                 ('ifrs/2025', 'ifrs_2025')]
        for taxonomy, suffix in cases:
            for tag in ('AssetsCurrent', 'Revenue', 'EarningsPerShareBasic'):
                for qtrs in (0, 1, 2, 3, 4, 12):
                    for unit, unit_suffix in (('USD', ''), ('EUR', '__eur'),
                                              ('shares', '__shares'), ('USD/shares', '__usd_per_share')):
                        with self.subTest(taxonomy=taxonomy, tag=tag, qtrs=qtrs, unit=unit):
                            duration = '' if qtrs == 0 else ('_q' if qtrs == 1 else f'_{qtrs}q')
                            base = names.display_label(tag).lower().replace(' ', '_')
                            alias = 'sec_' + base + duration + unit_suffix + '__' + suffix
                            self.assert_roundtrip(recipe(tag, taxonomy=taxonomy, qtrs=qtrs, unit=unit), alias)
        # Outside the us-gaap family, a reviewed abbreviation has no special
        # meaning. It can only be a generic candidate for a literally named tag.
        self.assertEqual(names.spec_for_name('sec_assets_curr__us_gaap_2025'),
                         recipe('AssetsCurr', taxonomy='us-gaap/2025'))
        self.assertFalse(names.accepts_name('sec_assets_curr__us_gaap_2025',
                                           recipe('AssetsCurrent', taxonomy='us-gaap/2025')))
        self.assertEqual(names.spec_for_name('sec_assets__ifrs'), recipe(taxonomy='ifrs/'))
        self.assertNotEqual(names.spec_for_name('sec_revenue__ifrs'),
                            names.spec_for_name('sec_revenue__ifrs_full'))

    def test_display_label_full_words_acronyms_and_numbers(self):
        cases = {
            'AccountsReceivableNetCurrent': 'Accounts Receivable Net Current',
            'EarningsPerShareBasic': 'Earnings Per Share Basic',
            'EarningsPerShareDiluted': 'Earnings Per Share Diluted',
            'USGAAPRevenue': 'USGAAP Revenue', 'EBITDA': 'EBITDA',
            'IncomeTaxExpenseBenefit': 'Income Tax Expense Benefit',
            'XMLHTTPRequest': 'XMLHTTP Request', 'USDPerShare': 'USD Per Share',
            'Level3FairValue': 'Level 3 Fair Value',
            'NetCashProvidedByUsedInOperatingActivities': 'Net Cash Provided By Used In Operating Activities',
        }
        for tag, label in cases.items():
            self.assertEqual(names.display_label(tag), label)
        for tag in (None, 4, '', 'lowercase', 'Assets_Current', 'Assets-Current', 'Ässets'):
            self.assertEqual(names.display_label(tag), '')

    def test_distinct_tags_do_not_pool(self):
        for tag in ('CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
                    'CashAndShortTermInvestments', 'InventoryGross',
                    'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                    'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax',
                    'Revenue', 'AccountsReceivableNetCurrent'):
            alias = 'sec_' + names.display_label(tag).lower().replace(' ', '_')
            self.assert_roundtrip(recipe(tag), alias)
        self.assertEqual(names.spec_for_name('sec_sales'), recipe('Sales'))
        self.assertFalse(names.accepts_name('sec_sales', recipe('Revenues')))

    def test_no_contexts_or_metadata_silently_dropped(self):
        for key in ('segments', 'coreg'):
            for value in (None, ' ', 'Axis=One', 'Issuer', 0, False, []):
                self.assertIsNone(names.preferred_name(recipe(**{key: value})))
                self.assertIsNone(names.generic_alias_for_spec(recipe(**{key: value})))
                self.assertFalse(names.accepts_name('sec_assets', recipe(**{key: value})))
        for key in recipe():
            spec = recipe()
            del spec[key]
            self.assertIsNone(names.preferred_name(spec))
        for key in ('kind', 'custom', 'version', 'dimh', 'dimn', 'iprx'):
            spec = recipe()
            spec[key] = ''
            self.assertIsNone(names.preferred_name(spec))

    def test_unsupported_taxonomies_and_customs(self):
        for taxonomy in ('dei/', 'custom/', '0000123456-25-000001', 'ifrs-full',
                         'ifrs', 'us-gaap', 'us-gaap/2025/', 'us-gaap/25',
                         'us-gaap/02025', 'us-gaap/0000', 'us-gaap/２０２５',
                         'ifrs/2025extra', 'US-GAAP/', ''):
            self.assertIsNone(names.preferred_name(recipe(taxonomy=taxonomy)))

    def test_invalid_specs(self):
        for value in (None, [], 'Assets', 42):
            self.assertIsNone(names.preferred_name(value))
        for tag in ('', 'assets', 'Assets_Current', 'Assets-Current', 'Assets.Current',
                    ' Assets', 'Assets\n', 'Ässets', '1Assets', 'Assets/Current'):
            self.assertIsNone(names.preferred_name(recipe(tag)))
        for qtrs in (-1, True, False, 1.0, '1', None):
            self.assertIsNone(names.preferred_name(recipe(qtrs=qtrs)))
        for unit in ('usd', 'USD ', 'EUR/shares', 'USD / shares', 'USD/shares/shares',
                     'SHARES', 'USDshares', 'ZZZ', '', None):
            self.assertIsNone(names.preferred_name(recipe(unit=unit)))

    def test_malformed_names_suffixes_and_ambiguities(self):
        for alias in (None, [], 42, '', 'assets', 'SEC_Assets', 'sec_',
                      'sec_AssetsCurrent', 'sec_Assets_Current', 'sec_Assets-Current',
                      'sec_Assets.Current', 'sec_Assets\n', 'sec_Ässets',
                      'sec_assets-current', 'sec_assets.current', 'sec_assets\n',
                      'sec_1assets', 'sec_ässsets', 'sec_assets_',
                      'sec_assets_0q', 'sec_assets_1q', 'sec_assets_01q',
                      'sec_assets_02q', 'sec_assets_q1', 'sec_assets_2Q',
                      'sec_assets_q_q', 'sec_assets_4q_q', 'sec_assets__usd', 'sec_assets__EUR',
                      'sec_assets__eur_per_share', 'sec_assets__zzz',
                      'sec_assets__shares__eur', 'sec_assets__eur__shares',
                      'sec_assets__', 'sec_assets___eur', 'sec_assets__eur_q',
                      'sec_assets__us_gaap', 'sec_assets__us_gaap_025',
                      'sec_assets__us_gaap_0000', 'sec_assets__us_gaap_2025__eur',
                      'sec_assets__ifrs__ifrs_full', 'sec_assets__ifrs_２０２５',
                      'sec_assets__ifrs_full_2025__shares', 'sec_assets__ifrs_2025__',
                      'sec_assets__eur__ifrs__extra', 'sec_assets__us_gaap_2025_4q'):
            with self.subTest(alias=alias):
                self.assertIsNone(names.spec_for_name(alias))
                self.assertFalse(names.recognized_alias(alias))
                self.assertFalse(names.accepts_name(alias, recipe()))

    def test_generic_syntax_recognition_requires_caller_existence_check(self):
        spec = recipe('ClearlyNotAnObservedSECConcept')
        self.assert_roundtrip(spec, 'sec_clearly_not_an_observed_sec_concept')
        self.assertEqual(names.spec_for_name('sec_nonsense'), recipe('Nonsense'))
        # These words are generic tag candidates, never inferred calculations.
        self.assertEqual(names.spec_for_name('sec_assets_ttm'), recipe('AssetsTtm'))
        self.assertFalse(names.accepts_name('sec_assets_ttm', recipe(qtrs=4)))

    def test_acronym_guesses_require_canonical_lookup(self):
        cases = [('USGAAPRevenue', 'usgaap_revenue', 'UsgaapRevenue'),
                 ('EBITDA', 'ebitda', 'Ebitda'),
                 ('XMLHTTPRequest', 'xmlhttp_request', 'XmlhttpRequest'),
                 ('Level3FairValue', 'level_3_fair_value', 'Level3FairValue')]
        for tag, base, guess in cases:
            spec = recipe(tag, unit='EUR', qtrs=3, taxonomy='ifrs/2025')
            alias = 'sec_' + base + '_3q__eur__ifrs_2025'
            self.assert_roundtrip(spec, alias)
            self.assertEqual(names.spec_for_name(alias), dict(spec, tag=guess))

    def test_exact_normalized_alias_filters_compact_lookup_candidates(self):
        # Same case-insensitive compact key, but different word boundaries.
        specs = [recipe('USGAAPRevenue'), recipe('UsGaapRevenue')]
        alias = 'sec_usgaap_revenue'
        guess = names.spec_for_name(alias)['tag']
        indexed_candidates = [spec for spec in specs if spec['tag'].lower() == guess.lower()]
        self.assertEqual(len(indexed_candidates), 2)
        self.assertEqual([spec for spec in indexed_candidates if names.accepts_name(alias, spec)],
                         [recipe('USGAAPRevenue')])

    def test_generic_acronym_collisions_remain_for_caller_to_reject(self):
        specs = [recipe('EBITDA'), recipe('Ebitda')]
        self.assertEqual([names.preferred_name(spec) for spec in specs], ['sec_ebitda'] * 2)
        matches = [spec for spec in specs if names.accepts_name('sec_ebitda', spec)]
        # A caller must reject this nonunique result when resolving or displaying.
        self.assertEqual(len(matches), 2)

    def test_generic_collisions_with_reviewed_bases_fail_closed(self):
        for tag, alias, reviewed in (
                ('Inventory', 'sec_inventory', 'InventoryNet'),
                ('AssetsCurr', 'sec_assets_curr', 'AssetsCurrent'),
                ('CashEquivalents', 'sec_cash_equivalents', 'CashAndCashEquivalentsAtCarryingValue'),
                ('NetIncome', 'sec_net_income', 'NetIncomeLoss'),
                ('ASSETS', 'sec_assets', 'Assets')):
            spec = recipe(tag)
            self.assertIsNone(names.preferred_name(spec))
            self.assertIsNone(names.generic_alias_for_spec(spec))
            self.assertFalse(names.accepts_name(alias, spec))
            self.assertEqual(names.spec_for_name(alias)['tag'], reviewed)
        # Reviewed bases are not reserved outside their explicit family scope.
        self.assert_roundtrip(recipe('Inventory', taxonomy='ifrs/'), 'sec_inventory__ifrs')

    def test_duration_shaped_tag_endings_do_not_steal_selectors(self):
        for tag in ('AssetsQ', 'Assets4q', 'Assets2Q'):
            self.assertIsNone(names.preferred_name(recipe(tag)))
            self.assertIsNone(names.generic_alias_for_spec(recipe(tag)))
        self.assertEqual(names.spec_for_name('sec_assets_q'), recipe(qtrs=1))
        self.assertFalse(names.accepts_name('sec_assets_q', recipe('AssetsQ')))

    def test_accepts_name_enforces_all_selectors_and_exact_word_boundaries(self):
        spec = recipe('AccountsReceivableNetCurrent', taxonomy='ifrs/2025', unit='EUR', qtrs=3)
        alias = 'sec_accounts_receivable_net_current_3q__eur__ifrs_2025'
        self.assertTrue(names.accepts_name(alias, spec))
        for changes in ({'tag': 'AccountsReceivablenetCurrent'}, {'unit': 'USD'},
                        {'qtrs': 2}, {'taxonomy': 'ifrs-full/2025'},
                        {'taxonomy': 'ifrs/'}, {'segments': 'Axis=One'}, {'coreg': 'Issuer'}):
            self.assertFalse(names.accepts_name(alias, dict(spec, **changes)))
        for invalid_spec in (None, [], {}, recipe(tag='Bad_Tag')):
            self.assertFalse(names.accepts_name(alias, invalid_spec))
        self.assertTrue(names.accepts_name('sec_assets_curr', recipe('AssetsCurrent')))
        self.assertTrue(names.accepts_name('sec_assets_current', recipe('AssetsCurrent')))

    def test_curated_collision_and_suffix_guards(self):
        for pairs in ([('Assets', 'assets'), ('Other', 'assets')],
                      [('Assets', 'assets'), ('Assets', 'other')],
                      [('Assets', 'assets_q')], [('Assets', 'assets__eur')],
                      [('Assets', 'Assets')], [('Assets_Current', 'assets')]):
            with self.assertRaises(ValueError):
                names._curated_maps(pairs)

    def test_reviewed_identity_guard_fails_closed(self):
        with patch.object(names, '_CURATED_TAGS', {'assets': 'Liabilities'}):
            self.assertIsNone(names.preferred_name(recipe()))

    def test_pure_module_import_and_functions_without_storage_or_numeric_imports(self):
        script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'pandas', 'pyarrow', 'sqlite3'} or name.endswith('sec_registry'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from qlib_yahoo import sec_names
def no_storage(*args, **kwargs):
    raise AssertionError('storage opened')
builtins.open = no_storage
assert sec_names.preferred_name(sec_names.spec_for_name('sec_assets_current')) == 'sec_assets_curr'
assert sec_names.curated_defs()['sec_inventory']['tag'] == 'InventoryNet'
'''
        result = subprocess.run([sys.executable, '-B', '-S', '-c', script],
                                cwd=Path(__file__).resolve().parent.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()

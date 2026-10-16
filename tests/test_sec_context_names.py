"""Bounded offline alias fixtures; no database, catalog, downloads or data writes."""
import copy
import unittest

from qlib_yahoo import sec_context_names as names
from qlib_yahoo import sec_names as ordinary


ENTITY = 'MorganStanleySmithBarneySpectrumTechnicalLP'
ACCESSION = '0000895421-24-000123'


def recipe(tag='Cash', **changes):
    return dict(dict(tag=tag, taxonomy='us-gaap/', unit='USD', qtrs=0,
                     segments='', coreg=''), **changes)


def filing(**changes):
    return dict(dict(adsh=ACCESSION, name='MORGAN STANLEY', form='10-Q', filed='20240501'), **changes)


def confirmed_fixture(name, candidates, *, complete=True):
    """Model main's required confirmation over a bounded complete candidate set.

    candidates are (exact_spec, kind, source_SUB_metadata) triples already
    nominated by indexed lookup. This is deliberately not production DB code.
    """
    if not complete:
        return None
    exact_matches = {}
    for spec, kind, metadata in candidates:
        if names.accepts_name(name, spec, kind=kind, filing=metadata):
            exact_matches[tuple(sorted(spec.items()))] = spec
    return next(iter(exact_matches.values())) if len(exact_matches) == 1 else None


class StandardContextNamesTests(unittest.TestCase):
    def test_requested_cash_entity_and_successor_names(self):
        entity = recipe(segments='LegalEntity=' + ENTITY, coreg=ENTITY)
        self.assertEqual(names.preferred_name(entity),
                         'sec_cash__entity_morgan_stanley_smith_barney_spectrum_technical_lp')
        successor = recipe(segments='Scenario=Successor')
        self.assertEqual(names.preferred_name(successor), 'sec_cash__scenario_successor')
        self.assertTrue(names.accepts_name('sec_cash__scenario_successor', successor))
        self.assertIsNone(ordinary.preferred_name(entity))
        self.assertIsNone(ordinary.preferred_name(successor))

    def test_parser_supplies_only_indexed_lookup_hints(self):
        parsed = names.candidate_for_name('sec_cash__scenario_successor')
        self.assertEqual(parsed, dict(tag='Cash', taxonomy='us-gaap/', unit='USD',
                                      qtrs=0, context_labels=('scenario_successor',)))
        self.assertNotIn('segments', parsed)
        self.assertNotIn('coreg', parsed)
        self.assertIsNotNone(names.candidate_for_name('sec_nonexistent_economic_tag__scenario_successor'))

    def test_all_unit_duration_taxonomy_selectors_retained(self):
        for taxonomy, suffix in (('us-gaap/', ''), ('us-gaap/2024', '__us_gaap_2024'),
                                 ('ifrs/', '__ifrs'), ('ifrs/2024', '__ifrs_2024'),
                                 ('ifrs-full/', '__ifrs_full'), ('ifrs-full/2024', '__ifrs_full_2024')):
            for unit, unit_suffix in (('USD', ''), ('EUR', '__eur'),
                                      ('USD/shares', '__usd_per_share'), ('shares', '__shares')):
                for qtrs, duration in ((0, ''), (1, '_q'), (4, '_4q'), (12, '_12q')):
                    with self.subTest(taxonomy=taxonomy, unit=unit, qtrs=qtrs):
                        spec = recipe(unit=unit, qtrs=qtrs, taxonomy=taxonomy, segments='Scenario=Successor')
                        name = 'sec_cash' + duration + unit_suffix + suffix + '__scenario_successor'
                        self.assertEqual(names.preferred_name(spec), name)
                        hints = names.candidate_for_name(name)
                        self.assertEqual({key: hints[key] for key in ('tag', 'taxonomy', 'unit', 'qtrs')},
                                         {key: spec[key] for key in ('tag', 'taxonomy', 'unit', 'qtrs')})
                        self.assertTrue(names.accepts_name(name, spec))

    def test_ordinary_names_unchanged_and_no_context_silently_removed(self):
        for tag in ('Assets', 'InventoryNet', 'Cash', 'USGAAPRevenue', 'EarningsPerShareBasic'):
            for unit in ('USD', 'EUR', 'USD/shares'):
                spec = recipe(tag, unit=unit, qtrs=4)
                self.assertEqual(names.preferred_name(spec), ordinary.preferred_name(spec))
                self.assertEqual(names.generic_alias_for_spec(spec), ordinary.generic_alias_for_spec(spec))
        advanced = recipe('Assets', segments='Scenario=Successor')
        self.assertFalse(names.accepts_name('sec_assets', advanced))
        self.assertFalse(names.accepts_name('sec_assets__scenario_successor', recipe('Assets')))

    def test_reviewed_base_and_full_original_tag_alternate(self):
        spec = recipe('InventoryNet', segments='Scenario=Successor')
        self.assertEqual(names.preferred_name(spec), 'sec_inventory__scenario_successor')
        self.assertEqual(names.generic_alias_for_spec(spec), 'sec_inventory_net__scenario_successor')
        self.assertTrue(names.accepts_name('sec_inventory_net__scenario_successor', spec))
        self.assertIsNone(names.preferred_name(recipe('Inventory', segments='Scenario=Successor')))
        spec = recipe('RevenueFromContractWithCustomerExcludingAssessedTax', segments='Scenario=Successor')
        self.assertEqual(names.preferred_name(spec),
                         'sec_revenue_from_contract_with_customer_excluding_assessed_tax__scenario_successor')

    def test_acronym_case_is_restored_only_from_actual_candidate(self):
        spec = recipe('EBITDA', segments='Scenario=Successor')
        alias = names.preferred_name(spec)
        self.assertEqual(alias, 'sec_ebitda__scenario_successor')
        self.assertEqual(names.candidate_for_name(alias)['tag'], 'Ebitda')
        self.assertEqual(confirmed_fixture(alias, [(spec, 'standard', None)]), spec)
        collision = recipe('Ebitda', segments='Scenario=Successor')
        self.assertIsNone(confirmed_fixture(alias, [(spec, 'standard', None), (collision, 'standard', None)]))

    def test_full_dimension_member_and_namespace_words(self):
        spec = recipe(segments='us-gaap:BusinessSegmentsAxis=issuer:AsiaPacificSegmentMember')
        self.assertEqual(names.preferred_name(spec),
                         'sec_cash__context_us_gaap_business_segments_axis_issuer_asia_pacific_segment_member')
        self.assertEqual(names.preferred_name(recipe(segments='LegalEntityAxis=SuccessorMember')),
                         'sec_cash__context_legal_entity_axis_successor_member')

    def test_multiple_contexts_canonical_sorted_and_only_exact_coreg_collapses(self):
        pairs = ['LegalEntity=' + ENTITY, 'Scenario=Successor', 'BusinessSegment=AsiaPacificSegment']
        spec = recipe(segments=';'.join(pairs), coreg=ENTITY)
        alias = ('sec_cash__context_business_segment_asia_pacific_segment'
                 '__entity_morgan_stanley_smith_barney_spectrum_technical_lp__scenario_successor')
        self.assertEqual(names.preferred_name(spec), alias)
        self.assertEqual(names.preferred_name(dict(spec, segments=';'.join(reversed(pairs)) + ';')), alias)
        different = dict(spec, coreg='Morgan Stanley Smith Barney Spectrum Technical LP')
        self.assertIn('__coreg_morgan_stanley_smith_barney_spectrum_technical_lp', names.preferred_name(different))

    def test_normalization_cannot_certify_distinct_exact_contexts(self):
        cases = [
            (recipe(coreg='Acme & Sons'), recipe(coreg='Acme-Sons')),
            (recipe(segments='Scenario=Successor'), recipe(segments='Scenario=SUCCESSOR')),
            (recipe(segments='Axis=One;Other=Two'), recipe(segments='Other=Two;Axis=One')),
            (recipe(segments='LegalEntity=' + ENTITY), recipe(segments='LegalEntity=' + ENTITY, coreg=ENTITY)),
            (recipe(segments='Axis=One'), recipe(segments='Axis=One;')),
            # Generic axis/member boundaries can normalize to identical words.
            (recipe(segments='AxisNorth=AmericanMember'), recipe(segments='Axis=NorthAmericanMember')),
        ]
        for one, two in cases:
            with self.subTest(one=one, two=two):
                alias = names.preferred_name(one)
                self.assertIsNotNone(alias)
                self.assertEqual(alias, names.preferred_name(two))
                self.assertIsNone(confirmed_fixture(alias, [(one, 'standard', None), (two, 'standard', None)]))

    def test_incomplete_lookup_does_not_certify_uniqueness_and_duplicates_are_same_identity(self):
        spec = recipe(segments='Scenario=Successor')
        alias = names.preferred_name(spec)
        candidates = [(spec, 'standard', None)] * 2
        self.assertEqual(confirmed_fixture(alias, candidates), spec)
        self.assertIsNone(confirmed_fixture(alias, candidates, complete=False))

    def test_reserved_words_stay_inside_context_namespace(self):
        spec = recipe(segments='EUR=USDPerShare;Taxonomy=USGAAP2024', coreg='Acme__eur__us_gaap_2024')
        alias = names.preferred_name(spec)
        self.assertEqual(alias, 'sec_cash__context_eur_usd_per_share__context_taxonomy_usgaap_2024'
                               '__coreg_acme_eur_us_gaap_2024')
        hints = names.candidate_for_name(alias)
        self.assertEqual(hints['unit'], 'USD')
        self.assertEqual(hints['taxonomy'], 'us-gaap/')

    def test_readable_punctuation_safe_but_no_control_xml_or_opaque_fallback(self):
        self.assertEqual(names.preferred_name(recipe(coreg="Acme's (Research), Inc.")),
                         'sec_cash__coreg_acme_s_research_inc')
        for value in ('', ' ', '12345', ACCESSION, 'a' * 40, 'abc\nCompany', 'Acme\tCompany',
                      '<script>Company</script>', 'Acme\\Company', 'München', 'Ｃｏｍｐａｎｙ',
                      '123e4567-e89b-12d3-a456-426614174000'):
            with self.subTest(value=value):
                if value:
                    self.assertIsNone(names.preferred_name(recipe(coreg=value)))
                self.assertIsNone(names.preferred_name(recipe(segments='LegalEntity=' + value)))

    def test_reject_malformed_raw_segments_and_duplicate_axes(self):
        for segment in (';', ';;', '=Member', 'Axis=', 'Axis=One=Two', 'Axis=One;;',
                        'Axis=One;axis=Two', 'Axis=One;Axis=One', 'PlainRawContext',
                        'Axis=One\nOther=Two', 'Axis=One;Other=', 'Axis=One|Other=Two'):
            self.assertIsNone(names.preferred_name(recipe(segments=segment)), segment)

    def test_null_missing_extra_invalid_selector_types_fail_closed(self):
        for key in recipe():
            for value in (None, [], False):
                self.assertIsNone(names.preferred_name(recipe(**{key: value})))
            spec = recipe()
            del spec[key]
            self.assertIsNone(names.preferred_name(spec))
        for qtrs in (-1, 1.0, '1'):
            self.assertIsNone(names.preferred_name(recipe(qtrs=qtrs)))
        self.assertIsNone(names.preferred_name(dict(recipe(), kind='standard')))
        self.assertIsNone(names.preferred_name(recipe(), kind='custom'))

    def test_long_original_tags_and_contexts_not_truncated(self):
        tag = 'Cash' + 'LongOriginalTagWords' * 100
        entity = 'MorganStanley' + 'FullSourceEntityWords' * 200
        spec = recipe(tag, segments='LegalEntity=' + entity)
        alias = names.preferred_name(spec)
        self.assertGreater(len(alias), 5000)
        self.assertEqual(alias, 'sec_' + ordinary.display_label(tag).lower().replace(' ', '_')
                         + '__entity_morgan_stanley' + '_full_source_entity_words' * 200)
        self.assertTrue(names.accepts_name(alias, spec))

    def test_parser_rejects_noncanonical_suffixes(self):
        for name in (None, 1, '', 'sec_Cash__scenario_successor', 'sec_cash__',
                     'sec_cash__scenario_', 'sec_cash__scenario_123',
                     'sec_cash__scenario_successor__eur', 'sec_cash__us_gaap_2024__eur',
                     'sec_cash__scenario_successor__coreg_acme',
                     'sec_cash__scenario_successor__scenario_successor',
                     'sec_cash__entity_one__entity_two', 'sec_cash__unknown_successor',
                     'sec_cash__scenario_a-b', 'sec_cash__scenario_successor\n',
                     'sec_cash_1q__scenario_successor', 'sec_cash_04q__scenario_successor'):
            self.assertIsNone(names.candidate_for_name(name), name)

    def test_pure_functions_do_not_mutate_input(self):
        spec = recipe(segments='LegalEntity=' + ENTITY, coreg=ENTITY)
        before = copy.deepcopy(spec)
        alias = names.preferred_name(spec)
        names.generic_alias_for_spec(spec)
        names.accepts_name(alias, spec)
        self.assertEqual(spec, before)

    def test_nonexample_accounting_tags_business_and_geography_contexts(self):
        # Hand-written expectations exercise categories beyond Cash/Inventory.
        cases = (
            ('AccountsReceivableNetCurrent', 0, 'CHF',
             'BusinessSegment=CommercialBankingMember',
             'sec_accounts_receivable_net_current__chf__context_business_segment_commercial_banking_member'),
            ('EarningsPerShareDiluted', 1, 'EUR/shares',
             'GeographicalArea=EuropeanOperationsMember',
             'sec_eps_diluted_q__unit_eur_per_shares__context_geographical_area_european_operations_member'),
            ('RevenueFromContractWithCustomerExcludingAssessedTax', 3, 'ARS',
             'BusinessSegmentsAxis=InvestmentManagementMember;GeographyAxis=LatinAmericaMember',
             'sec_revenue_from_contract_with_customer_excluding_assessed_tax_3q__unit_ars'
             '__context_business_segments_axis_investment_management_member'
             '__context_geography_axis_latin_america_member'),
            ('LongTermDebtCurrent', 0, 'JPY', 'GeographyAxis=JapanMember',
             'sec_debt_lt_current__jpy__context_geography_axis_japan_member'),
            ('DebtInstrumentInterestExpense', 2, 'ILS', 'BusinessSegment=CorporateOperations',
             'sec_debt_instrument_interest_expense_2q__unit_ils__context_business_segment_corporate_operations'),
            ('EarningsPerShareBasic', 1, 'USD/shares', 'Scenario=Predecessor',
             'sec_eps_basic_q__usd_per_share__scenario_predecessor'),
        )
        for tag, qtrs, unit, segments, alias in cases:
            with self.subTest(tag=tag):
                spec = recipe(tag, unit=unit, qtrs=qtrs, segments=segments)
                self.assertEqual(names.preferred_name(spec), alias)
                self.assertEqual(confirmed_fixture(alias, [(spec, 'standard', None)]), spec)
                # Full original accounting words remain available for EVERY tag.
                generic = names.generic_alias_for_spec(spec)
                self.assertTrue(generic.startswith('sec_' + ordinary.display_label(tag).lower().replace(' ', '_')))
                self.assertTrue(names.accepts_name(generic, spec))

    def test_every_existing_reviewed_base_extends_uniformly_without_new_tag_mappings(self):
        for tag, base in ordinary.CURATED_NAMES.items():
            with self.subTest(tag=tag):
                spec = recipe(tag, unit='ARS', qtrs=5,
                              segments='BusinessSegment=ServicesMember;GeographyAxis=EuropeMember')
                alias = ('sec_' + base + '_5q__unit_ars__context_business_segment_services_member'
                         '__context_geography_axis_europe_member')
                self.assertEqual(names.preferred_name(spec), alias)
                self.assertTrue(names.accepts_name(alias, spec))
                generic = names.generic_alias_for_spec(spec)
                self.assertTrue(generic.startswith('sec_' + ordinary.display_label(tag).lower().replace(' ', '_')))
                self.assertTrue(names.accepts_name(generic, spec))


class AdditionalSelectorNamesTests(unittest.TestCase):
    def test_ars_cash_with_blank_or_advanced_context(self):
        for segment, suffix in (('', ''), ('Scenario=Successor', '__scenario_successor')):
            spec = recipe(unit='ARS', segments=segment)
            alias = 'sec_cash__unit_ars' + suffix
            self.assertEqual(names.preferred_name(spec), alias)
            hints = names.candidate_for_name(alias)
            self.assertEqual(hints['unit_hint'], 'ars')
            self.assertNotIn('unit', hints)
            self.assertEqual(hints['taxonomy'], 'us-gaap/')
            self.assertTrue(names.accepts_name(alias, spec))

    def test_arbitrary_readable_units_retain_source_words_and_operators(self):
        for unit, suffix in (('EUR/shares', 'eur_per_shares'), ('USD/mmBtu', 'usd_per_mm_btu'),
                             ('kWh', 'k_wh'), ('USD*shares', 'usd_times_shares'),
                             ('Acme-ReportedUnits', 'acme_reported_units')):
            with self.subTest(unit=unit):
                spec = recipe(unit=unit, qtrs=4, taxonomy='us-gaap/2024')
                alias = 'sec_cash_4q__unit_' + suffix + '__us_gaap_2024'
                self.assertEqual(names.preferred_name(spec), alias)
                self.assertEqual(names.candidate_for_name(alias)['unit_hint'], suffix)
                self.assertTrue(names.accepts_name(alias, spec))
        for unit in ('', '123', ACCESSION, 'USD/', '/shares', 'USD\nshares', 'USD//shares'):
            self.assertIsNone(names.preferred_name(recipe(unit=unit)), unit)

    def test_meaningful_unknown_taxonomies_explicit_scope_and_original_tag(self):
        for taxonomy, suffix in (('dei/', 'dei_family'), ('dei/2024', 'dei_2024_edition'),
                                 ('srt/2024', 'srt_2024_edition'),
                                 ('Vendor-Defined/', 'vendor_defined_family')):
            spec = recipe('InventoryNet', taxonomy=taxonomy, unit='ARS')
            alias = 'sec_inventory_net__unit_ars__taxonomy_' + suffix
            self.assertEqual(names.preferred_name(spec), alias)
            hints = names.candidate_for_name(alias)
            self.assertEqual(hints['taxonomy_hint'], suffix)
            self.assertNotIn('taxonomy', hints)
            self.assertTrue(names.accepts_name(alias, spec))
        self.assertNotEqual(names.preferred_name(recipe(taxonomy='dei/')),
                            names.preferred_name(recipe(taxonomy='dei')))

    def test_normalized_units_and_taxonomies_still_need_exact_collision_checks(self):
        for one, two in ((recipe(unit='ARS'), recipe(unit='ars')),
                         (recipe(taxonomy='vendor-name/'), recipe(taxonomy='vendor_name/'))):
            alias = names.preferred_name(one)
            self.assertIsNotNone(alias)
            self.assertEqual(alias, names.preferred_name(two))
            self.assertIsNone(confirmed_fixture(alias, [(one, 'standard', None), (two, 'standard', None)]))

    def test_unknown_taxonomy_collision_must_include_standard_and_custom_candidates(self):
        standard = recipe('Assets', taxonomy='new-family/2024', segments='Scenario=Successor')
        custom = recipe('Assets', taxonomy='new_family/2024', segments='Scenario=Successor')
        alias = 'sec_assets__taxonomy_new_family_2024_edition__scenario_successor'
        self.assertNotEqual(standard, custom)
        self.assertEqual(names.preferred_name(standard, kind='standard'), alias)
        self.assertEqual(names.preferred_name(custom, kind='custom'), alias)
        self.assertTrue(names.accepts_name(alias, standard, kind='standard'))
        self.assertTrue(names.accepts_name(alias, custom, kind='custom'))
        hints = names.candidate_for_name(alias)
        self.assertEqual(hints['taxonomy_hint'], 'new_family_2024_edition')
        self.assertNotIn('taxonomy', hints)
        self.assertNotIn('kind', hints)
        # Pure confirmation demonstrates the required candidate superset; the
        # actual indexed cross-kind lookup fixture belongs to integration.
        self.assertIsNone(confirmed_fixture(alias, [(standard, 'standard', None), (custom, 'custom', None)]))

    def test_extra_selectors_cannot_duplicate_or_change_order(self):
        for alias in ('sec_cash__unit_ars__eur', 'sec_cash__eur__unit_ars',
                      'sec_cash__taxonomy_dei_family__unit_ars',
                      'sec_cash__taxonomy_dei', 'sec_cash__taxonomy_123_edition',
                      'sec_cash__unit_', 'sec_cash__unit_123',
                      'sec_cash__taxonomy_dei_family__ifrs'):
            self.assertIsNone(names.candidate_for_name(alias), alias)


class FilingMetadataNamesTests(unittest.TestCase):
    def test_custom_accession_has_full_tag_issuer_form_date_and_no_accession(self):
        spec = recipe('CompanyReportedCashAndRestrictedDeposits', taxonomy=ACCESSION,
                      unit='ARS', qtrs=1, segments='Scenario=Successor')
        alias = ('sec_company_reported_cash_and_restricted_deposits_q__unit_ars'
                 '__issuer_morgan_stanley__form_10_q__filed_2024_05_01__scenario_successor')
        self.assertEqual(names.preferred_name(spec, kind='custom', filing=filing()), alias)
        self.assertNotIn(ACCESSION, alias)
        self.assertEqual(names.candidate_for_name(alias), dict(tag='CompanyReportedCashAndRestrictedDeposits',
                         unit_hint='ars', qtrs=1, context_labels=('scenario_successor',),
                         filing_hint=dict(issuer_slug='morgan_stanley', form_slug='10_q', filed='2024-05-01')))
        self.assertTrue(names.accepts_name(alias, spec, kind='custom', filing=filing()))

    def test_custom_curated_word_is_not_assigned_standard_meaning(self):
        spec = recipe('Inventory', taxonomy=ACCESSION)
        alias = names.preferred_name(spec, kind='custom', filing=filing())
        self.assertEqual(alias, 'sec_inventory__issuer_morgan_stanley__form_10_q__filed_2024_05_01')
        self.assertEqual(names.candidate_for_name(alias)['tag'], 'Inventory')
        spec = recipe('InventoryNet', taxonomy=ACCESSION)
        self.assertTrue(names.preferred_name(spec, kind='custom', filing=filing()).startswith('sec_inventory_net__'))

    def test_custom_nonexample_tags_and_units_use_same_full_word_filing_hook(self):
        for tag in ('AccountsReceivableNetCurrent', 'EarningsPerShareDiluted',
                    'RevenueFromContractWithCustomerExcludingAssessedTax', 'DebtInstrumentInterestExpense'):
            for unit in ('ILS', 'EUR/shares', 'USD/mmBtu'):
                with self.subTest(tag=tag, unit=unit):
                    spec = recipe(tag, taxonomy=ACCESSION, unit=unit, qtrs=4,
                                  segments='BusinessSegment=ServicesMember;GeographyAxis=EuropeMember')
                    metadata = filing(name='Other Source Issuer', form='20-F')
                    alias = names.preferred_name(spec, kind='custom', filing=metadata)
                    self.assertIsNotNone(alias)
                    self.assertTrue(alias.startswith('sec_' + ordinary.display_label(tag).lower().replace(' ', '_') + '_4q'))
                    self.assertIn('__issuer_other_source_issuer__form_20_f__filed_2024_05_01', alias)
                    self.assertTrue(alias.endswith('__context_business_segment_services_member__context_geography_axis_europe_member'))
                    self.assertEqual(confirmed_fixture(alias, [(spec, 'custom', metadata)]), spec)

    def test_source_filing_date_not_period_or_acceptance_and_amendment_form_retained(self):
        spec = recipe(taxonomy=ACCESSION)
        metadata = filing(form='10-Q/A', filed='2024-02-29', accepted='2024-03-01', period='20231231')
        alias = names.preferred_name(spec, kind='custom', filing=metadata)
        self.assertTrue(alias.endswith('__form_10_q_a__filed_2024_02_29'))
        self.assertFalse(names.accepts_name(alias, spec, kind='custom', filing=filing(form='10-Q/A')))
        self.assertEqual(names.preferred_name(spec, filing=filing(filed='20240501')),
                         names.preferred_name(spec, filing=filing(filed='2024-05-01')))

    def test_missing_conflicting_or_wrong_accession_metadata_is_not_invented(self):
        spec = recipe(taxonomy=ACCESSION)
        for metadata in (None, {}, filing(adsh='0000895421-24-000999'), filing(name=None), filing(name=''),
                         filing(form=None), filing(form='10 q'), filing(filed=None), filing(filed='20240230'),
                         filing(filed='2024-05-01T00:00:00'), filing(name=ACCESSION)):
            self.assertIsNone(names.preferred_name(spec, kind='custom', filing=metadata), metadata)
        self.assertIsNone(names.preferred_name(recipe(taxonomy='123456789'), filing=filing()))
        self.assertIsNone(names.preferred_name(recipe(taxonomy='us-gaap/' + ACCESSION), filing=filing()))

    def test_same_issuer_form_date_can_collide_across_exact_accessions(self):
        one = recipe(taxonomy=ACCESSION)
        other_accession = '0000895421-24-000124'
        two = recipe(taxonomy=other_accession)
        alias = names.preferred_name(one, kind='custom', filing=filing())
        candidates = [(one, 'custom', filing()), (two, 'custom', filing(adsh=other_accession))]
        self.assertEqual(alias, names.preferred_name(two, kind='custom', filing=filing(adsh=other_accession)))
        self.assertIsNone(confirmed_fixture(alias, candidates))
        self.assertEqual(confirmed_fixture(alias, candidates[:1]), one)

    def test_issuer_case_or_punctuation_collisions_still_rejected(self):
        one = recipe(taxonomy=ACCESSION)
        other_accession = '0000895421-24-000124'
        two = recipe(taxonomy=other_accession)
        alias = names.preferred_name(one, kind='custom', filing=filing(name='Acme & Sons'))
        self.assertEqual(alias, names.preferred_name(two, kind='custom',
                         filing=filing(adsh=other_accession, name='ACME-SONS')))
        self.assertIsNone(confirmed_fixture(alias, [(one, 'custom', filing(name='Acme & Sons')),
                         (two, 'custom', filing(adsh=other_accession, name='ACME-SONS'))]))

    def test_issuer_full_words_not_truncated_and_metadata_not_mutated(self):
        spec = recipe(taxonomy=ACCESSION)
        metadata = filing(name='Morgan Stanley ' + 'Full Source Company Name ' * 200)
        before = copy.deepcopy(metadata)
        alias = names.preferred_name(spec, kind='custom', filing=metadata)
        self.assertIn('__issuer_morgan_stanley' + '_full_source_company_name' * 200 + '__form_', alias)
        self.assertEqual(metadata, before)

    def test_filing_parser_rejects_incomplete_out_of_order_or_mixed_taxonomies(self):
        suffix = '__issuer_acme__form_10_q__filed_2024_05_01'
        for alias in ('sec_cash__issuer_acme', 'sec_cash__issuer_acme__filed_2024_05_01__form_10_q',
                      'sec_cash__issuer_123__form_10_q__filed_2024_05_01',
                      'sec_cash__issuer_acme__form_10_q__filed_2024_02_30',
                      'sec_cash__ifrs' + suffix, 'sec_cash__taxonomy_dei_family' + suffix,
                      'sec_cash__scenario_successor' + suffix,
                      'sec_cash' + suffix + '__issuer_other',
                      'sec_cash' + suffix + '__eur'):
            self.assertIsNone(names.candidate_for_name(alias), alias)


if __name__ == '__main__':
    unittest.main()

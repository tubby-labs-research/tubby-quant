"""Coverage and daily exclusion policy; no scientific dependencies."""
import unittest
from qlib_yahoo.chronology_scoring import scoring_split, phase_scoring


class ScoringCoverageTests(unittest.TestCase):
    def plan(self):
        from datetime import date, timedelta
        days = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(10)]
        return {'schema_version': 1, 'sessions': days,
                'development': {'start': days[0], 'end': days[2]},
                'validation': {'start': days[3], 'end': days[5]},
                'final': {'start': days[6], 'end': days[8]},
                'input_lineage': {'price': 'fixture'}, 'selection_rule': {'rule': 'supplied'}, 'metadata': {}}

    def complete(self):
        plan = self.plan()
        plan['metadata']['samples'] = [{'id': day, 'session': day,
            'information_start': day, 'information_end': day} for day in plan['sessions'][:9]]
        return plan

    def test_no_samples_remains_calendar_only(self):
        result = scoring_split(self.plan())
        self.assertEqual(result['coverage']['status'], 'not_supplied')
        self.assertEqual(result['overlap_check'], 'not_available')
        self.assertIsNone(phase_scoring(self.plan(), 'development'))

    def test_empty_or_partial_coverage_blocks_every_phase(self):
        for size in (0, 2, 8):
            plan = self.complete()
            plan['metadata']['samples'] = plan['metadata']['samples'][:size]
            result = scoring_split(plan)
            self.assertEqual(result['coverage']['status'], 'partial')
            for phase in ('development', 'validation', 'final'):
                with self.assertRaisesRegex(ValueError, 'complete session coverage'):
                    phase_scoring(plan, phase)

    def test_interior_exclusion_and_duplicate_session_are_conservative(self):
        plan = self.complete()
        days = plan['sessions']
        plan['metadata']['samples'].append({'id': 'another-component', 'session': days[1],
            'information_start': days[1], 'information_end': days[3]})
        result = scoring_split(plan)
        self.assertEqual(result['coverage']['status'], 'complete')
        self.assertEqual(phase_scoring(plan, 'development'), [days[0], days[2]])
        self.assertEqual(result['coverage']['phases']['development']['checked_count'], 3)

    def test_too_few_remaining_observations_refused(self):
        plan = self.complete()
        for sample in plan['metadata']['samples'][:2]:
            sample['information_end'] = plan['sessions'][3]
        with self.assertRaisesRegex(ValueError, 'at least two'):
            phase_scoring(plan, 'development')


if __name__ == '__main__':
    unittest.main()

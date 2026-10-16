import unittest
from autoresearch.policy import assess


def result(sharpe=1.2, returns=.1, days=80, **metrics):
    return {'status': 'ok', 'metrics': {'development': {'sharpe': 99},
            'validation': dict(sharpe=sharpe, returns=returns, turnover=.2, drawdown=.1, **metrics)},
            'periods': {'validation': {'days': days}}, 'settings': {'cost_bps': 10}}


class PolicyTests(unittest.TestCase):
    def decision(self, value, config=None, parent=None):
        assessment = assess(value, config or {}, parent)
        self.assertEqual(set(assessment), {'decision', 'reason', 'question', 'expected_change'})
        return assessment['decision']

    def test_explicit_gates_validation_and_deeper_research(self):
        self.assertEqual(self.decision(result()), 'promote')
        self.assertEqual(self.decision(result(), {'min_sharpe': 2}), 'refine')
        self.assertEqual(self.decision(result(), {'policy': {'min_validation_days': 100}}), 'park')
        value = result(sharpe=-.2)
        value['metrics']['holdout'] = {'sharpe': 100}
        self.assertEqual(self.decision(value), 'reject')

    def test_errors_are_separate_from_negative_science(self):
        self.assertEqual(self.decision({'status': 'error', 'error': 'timeout'}), 'repair')
        self.assertEqual(self.decision(result(returns=-.01)), 'reject')
        self.assertEqual(self.decision(result(sharpe=0)), 'reject')
        self.assertEqual(self.decision(result(sharpe=.8)), 'refine')
        self.assertEqual(self.decision(result(days=20)), 'park')

    def test_required_costs_and_finite_metrics(self):
        for key in ('sharpe', 'returns', 'turnover', 'drawdown'):
            for invalid in (None, float('nan'), float('inf'), True):
                value = result()
                value['metrics']['validation'][key] = invalid
                self.assertEqual(self.decision(value), 'repair')
        value = result()
        del value['settings']['cost_bps']
        self.assertEqual(self.decision(value), 'repair')
        for invalid in (None, [], 'bad'):
            value = result()
            value['metrics'] = invalid
            self.assertEqual(self.decision(value), 'repair')
        self.assertEqual(self.decision(result(), {'min_sharpe': float('nan')}), 'repair')
        self.assertEqual(self.decision(result(days=80.5)), 'repair')

    def test_stalled_refinement_terminates(self):
        self.assertEqual(self.decision(result(sharpe=.81), parent=result(sharpe=.8)), 'reject')
        self.assertEqual(self.decision(result(sharpe=.9), parent=result(sharpe=.8)), 'refine')
        self.assertEqual(self.decision(result(sharpe=1.2), parent=result(sharpe=.8)), 'promote')
        assessment = assess(result(sharpe=.7), {})
        self.assertTrue(assessment['question'])
        self.assertTrue(assessment['expected_change'])


if __name__ == '__main__':
    unittest.main()

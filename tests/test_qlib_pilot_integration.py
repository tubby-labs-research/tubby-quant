"""Retained independent fixture oracle checked through Qlib's file provider.

Production execution tests cover portfolio simulation; this test only covers
the prototype's unique availability/amendment/membership feature round-trip.
"""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.qlib_pilot_fixture import load_expected_asof, load_fixture, materialize_qlib_snapshot


@unittest.skipUnless(importlib.util.find_spec("qlib") and sys.version_info < (3, 13),
                     "optional Qlib runtime requires Python below 3.13")
class QlibFixtureIntegrationTests(unittest.TestCase):
    def test_fixture_feature_round_trip_matches_independent_oracle(self):
        fixture = Path(__file__).parent / "fixtures" / "qlib_pilot"
        expected = load_expected_asof(fixture / "expected_asof.csv")
        # Isolate Qlib global provider/cache state from production tests.
        script = """import hashlib, json, math, sys
import qlib
from qlib.constant import REG_US
from qlib.data import D
qlib.init(provider_uri=sys.argv[1], region=REG_US, kernels=1,
          dataset_cache=None, expression_cache=None)
fields = ['$open', '$close', '$volume', '$factor', '$change', '$book_value', 'Ref($close, 1)']
frame = D.features(['QLBA', 'QLBB', 'QLBC'], fields, start_time='2026-01-05',
                   end_time='2026-01-16', freq='day', disk_cache=0).sort_index()
rows = [[stamp.date().isoformat(), str(instrument).upper(),
         *[float(value) if math.isfinite(value) else None for value in values]]
        for (instrument, stamp), values in zip(frame.index, frame.to_numpy())]
scores = frame['$close'] / frame['Ref($close, 1)'] - 1.0
print(json.dumps({'rows': rows, 'score_rows': int(scores.notna().sum()),
                  'features_sha256': hashlib.sha256(frame.to_csv(lineterminator='\\n').encode()).hexdigest()}))
"""
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary).resolve() / "qlib_data"
            materialize_qlib_snapshot(load_fixture(fixture), destination)
            result = subprocess.run([sys.executable, "-c", script, str(destination)],
                                    capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout.splitlines()[-1])
        rows = {(day, instrument): values for day, instrument, *values in data["rows"]}
        self.assertEqual(len(rows), 25)
        self.assertEqual(set(rows), {(day.isoformat(), instrument) for day, instrument in expected})
        for (day, instrument), (value, _revision) in expected.items():
            with self.subTest(day=day, instrument=instrument):
                actual = rows[(day.isoformat(), instrument)][5]
                if value is None:
                    self.assertIsNone(actual)
                else:
                    self.assertTrue(math.isclose(actual, value, rel_tol=0, abs_tol=1e-5))
        self.assertIsNone(rows[("2026-01-12", "QLBB")][1])
        self.assertIsNone(rows[("2026-01-13", "QLBB")][6])
        self.assertEqual(data["score_rows"], 20)
        self.assertEqual(data["features_sha256"], "fd395b66467302c1830758577c10a4d2e303cfef79695d404d7e7d82022bb301")


if __name__ == "__main__":
    unittest.main()

# Independent Qlib availability and membership fixture

This is deterministic synthetic test data, not market data.

- `bars.csv`: daily OHLCV and factor values for three tradable instruments and a
  synthetic benchmark. QLBB has no 2026-01-12 bar. The benchmark has one
  post-period 2026-01-19 row so Qlib can resolve the end of its final daily
  execution interval; the evaluated backtest still ends on 2026-01-16.
- `universe.csv`: explicit effective membership intervals; QLBB starts late and
  QLBC ends early.
- `fundamentals.csv`: book-value events with publication availability and
  revisions for one reporting period. QLBB's event arrives after the daily
  cutoff, while QLBA is amended. Project code resolves these events before Qlib;
  this fixture does not test Qlib's native PIT store.
- `expected_asof.csv`: independently stated expected book value and revision at
  the 21:00 UTC signal cutoff. A missing market bar has no eligible feature row.

The standalone pilot runner was retired on 2026-10-14. The test-only helper
[`tests/qlib_pilot_fixture.py`](../../qlib_pilot_fixture.py) preserves its independent
as-of oracle and binary materializer. The unit tests retain all 25 expected
availability/revision checks, membership boundaries, missing-bar behavior and
the original fixture/binary hashes. The optional integration test reads the
features through Qlib and compares them to `expected_asof.csv`; production
execution tests cover the portfolio runner.

Do not replace these files with downloaded data or treat the fixture as a
canonical schema. The oracle supports one reporting period per field and is
independent of the production SEC adapter; it does not establish that adapter's
point-in-time correctness.

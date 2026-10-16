"""Complete target-decision timing, synthetic diagnostic rather than backtest acceptance.

Fresh native account/quote fixtures are prepared outside timing; each measured
generate_trade_decision includes eligibility guards, row proofs, array packing,
optional first compilation, target diagnostics and native Order construction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_probe(output, compiler_path, *, stocks=3000, trials=20, progress=None):
    output = Path(output).resolve()
    compiler_path = Path(compiler_path).resolve()
    if not output.parent.is_dir() or not compiler_path.is_dir():
        raise ValueError('Use an existing receipt directory and existing isolated compiler packages')
    if stocks < 3 or trials < 2:
        raise ValueError('Use at least three stocks and two paired trials')
    sys.path.insert(0, str(compiler_path))
    from qlib_yahoo.runtime import ensure_signed_runtime
    identity = ensure_signed_runtime()
    from tests.test_qlib_fast_signed import SignedChild
    compiler_versions = {name: importlib.metadata.version(name) for name in ('numba', 'llvmlite')}
    if compiler_versions != {'numba': '0.65.1', 'llvmlite': '0.47.0'}:
        raise ValueError('Requires reviewed isolated compiler versions')
    SignedChild.setUpClass()
    fixture = SignedChild()
    names = [f'S{i:05d}' for i in range(stocks)]
    rng = random.Random(7345)
    prices = {s: rng.uniform(.1, 100.) for s in names}
    weights = fixture.pd.Series({s: rng.uniform(-1, 1) / (2 * stocks) for s in names})
    holdings = {s: rng.randrange(-10, 11) for s in names[::3]}
    frame = fixture.frame(prices)

    def fresh(compiled):
        exchange = fixture.exchange(True, frame, fixture.DailyQuote)
        strategy, _ = fixture.strategy(True, weights, cash=1e6, frame=frame,
                                       holdings=holdings, exchange=exchange)
        strategy.compiled_targets = compiled
        return strategy

    def measured(compiled):
        strategy = fresh(compiled)
        before = time.perf_counter()
        decision = strategy.generate_trade_decision()
        elapsed = time.perf_counter() - before
        # Verification/encoding is outside target-boundary timing in BOTH paths.
        record = strategy.decisions[0]
        encoded = dict(record)
        encoded['issued_orders'] = fixture.orders(record['issued_orders'])
        result = (fixture.orders(decision.get_decision()), encoded)
        return elapsed, result, dict(strategy.target_session_counts)

    first_native = measured(False)
    if progress is not None:
        progress.update(phase='first_compiled', first_native_seconds=first_native[0])
        output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
    first_compiled = measured(True)
    if progress is not None:
        progress.update(phase='first_parity', first_compiled_including_jit_seconds=first_compiled[0],
                        first_compiled_counts=first_compiled[2])
        output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
    if first_native[1] != first_compiled[1]:
        raise AssertionError('First-use native/compiled complete records differ')
    if first_compiled[2]['compiled_sessions'] != 1:
        raise AssertionError('First-use trial fell back; cannot claim compiled target timing')
    pairs = []
    for trial in range(trials):
        values = {}
        for compiled in ((False, True) if trial % 2 == 0 else (True, False)):
            values[compiled] = measured(compiled)
        if values[False][1] != values[True][1]:
            raise AssertionError(f'Complete native/compiled record mismatch at trial {trial}')
        if values[True][2]['compiled_sessions'] != 1:
            raise AssertionError(f'Compiled trial {trial} fell back')
        pairs.append({'trial': trial, 'native_seconds': values[False][0],
                      'compiled_seconds': values[True][0], 'compiled_counts': values[True][2]})
        if progress is not None:
            progress.update(phase='paired_trials', completed_pairs=len(pairs), paired_trials=pairs)
            output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
    receipt = {'schema': 1, 'kind': 'synthetic_complete_target_boundary_only',
               'status': 'completed', 'phase': 'completed',
               'stocks': stocks, 'trials': trials, 'engine': identity,
               'compiler': compiler_versions,
               'first_native_seconds': first_native[0], 'first_compiled_including_jit_seconds': first_compiled[0],
               'first_compiled_counts': first_compiled[2], 'paired_trials': pairs,
               'native_median_seconds': statistics.median(p['native_seconds'] for p in pairs),
               'compiled_median_seconds': statistics.median(p['compiled_seconds'] for p in pairs),
               'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                   ('qlib_yahoo/fast_signed.py', 'qlib_yahoo/target_session.py', 'qlib_yahoo/target_session_kernel.py')},
               'exact_complete_decisions': True,
               'limitations': 'Synthetic prepared account/quote fixtures; not full-worker/backtest speed or acceptance.'}
    output.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def probe(output, compiler_path, *, stocks=3000, trials=20):
    """Leave a durable progress/failure receipt, including completed exact pairs."""
    output = Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Use a new receipt in an existing directory')
    progress = {'schema': 1, 'kind': 'synthetic_complete_target_boundary_only',
                'status': 'running', 'phase': 'preparation', 'stocks': stocks,
                'trials': trials, 'completed_pairs': 0, 'paired_trials': []}
    output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
    try:
        return _run_probe(output, compiler_path, stocks=stocks, trials=trials, progress=progress)
    except BaseException as exc:
        progress.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        output.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compiler-path', required=True)
    parser.add_argument('--stocks', type=int, default=3000)
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()
    receipt = probe(args.output, args.compiler_path, stocks=args.stocks, trials=args.trials)
    print(json.dumps({k: receipt[k] for k in ('first_compiled_including_jit_seconds',
                      'native_median_seconds', 'compiled_median_seconds', 'exact_complete_decisions')}))

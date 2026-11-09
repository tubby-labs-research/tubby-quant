# Lab benchmark results

Both benchmarks run on artificial data with known answers. They measure the search,
its selection control and its holdout discipline; they are not evidence about any
market or real dataset. Rerun after any change to the harness, a domain or the
Alpha Lab engine, and replace this record.

Recorded 2026-10-24 on the development tree from which this preview was exported (the alpha run preceded the machine-learning run; no machine-learning code changed between them); macOS arm64, CPython 3.12, numpy 1.26.4, pandas 2.2.3.

## Autonomous alpha discovery

```sh
tubby-lab benchmark alpha --root <new folder>
```

A 300-name, 2,000-session panel with three planted signals (noise seed 11), its null twin with identical noise, and four further null panels with other noise seeds. The same campaign runs on each: 20 rounds of 40 proposals, default policy (deflated Sharpe ratio at least 0.95 against the global null, Research gates, robustness vetoes).

| Panel | Screened | Validation trials | Promoted | Outcome |
| --- | --- | --- | --- | --- |
| planted | 341 | 12 | 8 | 3 of 3 planted signals recovered; 8 of 8 finals confirmed on the reserved holdout |
| null twin | 453 | 6 | 0 | 0 false promotions at review; 0 would reach the holdout |
| null, noise seed 101 | 392 | 3 | 0 | 0 at review; 0 would reach the holdout |
| null, noise seed 202 | 452 | 0 | 0 | 0 at review; 0 would reach the holdout |
| null, noise seed 303 | 420 | 0 | 0 | 0 at review; 0 would reach the holdout |
| null, noise seed 404 | 428 | 0 | 0 | 0 at review; 0 would reach the holdout |
| **all 5 null panels** | 2,145 | 9 | 0 | 0 false promotions at review; **0 would reach the holdout** |

A promotion is judged with the validation-trial count reached at that moment; before any holdout access,
final selection re-applies the deflated Sharpe ratio with the campaign's final count. "Would reach the
holdout" counts null promotions that survive that re-check.

| Planted signal | Oracle | Best daily rank correlation with a promoted alpha | Matching alpha | Generator |
| --- | --- | --- | --- | --- |
| lowvol | `-rank(ts_std_dev(returns, 20))` | 0.7247 | `-rank(ts_max(returns, 20))` | template |
| reversal | `-rank(ts_delta(close, 5) / ts_delay(close, 5))` | 0.9079 | `-rank(ts_sum((returns * (volume / ts_mean(volume, 20))), 5))` | template |
| volume | `rank(log(volume) - ts_mean(log(volume), 20))` | 0.9939 | `rank((volume / ts_mean(volume, 20)))` | template |

All 6 trial ledgers verified intact (2679 entries). Runtime 221.1 s.

History: the first final run promoted `-ts_delay(ts_min(returns, 40), 10)` on the
null panel. Its market factor had a positive drift while betas differed by name, so a low-beta tilt earned a
real but unplanted premium; the deflated Sharpe ratio at the final trial count (0.73) would have kept it from
the holdout. The panels are now driftless, and the benchmark runs several null panels.

## Machine learning

```sh
tubby-lab benchmark ml --panel --root <new folder>
```

8 rounds of 10 proposals per task, default policy (Bonferroni-adjusted paired bootstrap for tabular tasks; the alpha review for panels).

| Task | Known answer | Screened | Validation trials | Promoted | Confirmed finals | Best confirmed holdout |
| --- | --- | --- | --- | --- | --- | --- |
| friedman1 | irreducible RMSE 1.0 | 80 | 6 | 6 | 6/6 | rmse 1.204 vs constant baseline 4.986 (floor 1.0) |
| interaction | Bayes-optimal log loss on the holdout | 80 | 15 | 15 | 15/15 | log_loss 0.516 vs constant baseline 0.691 (Bayes 0.486) |
| null_regression | no relation: any promotion is false | 78 | 16 | 0 | 0/0 | - |
| panel_null | nothing planted: any promotion is false | 78 | 13 | 0 | 0/0 | - |
| panel_planted | three planted signals | 79 | 14 | 9 | 9/9 | annual Sharpe 5.19 over 280 sessions (artificial panel) |

Runtime 922.0 s.

## Reading these numbers

- Recall and false promotions are the pipeline's two failure modes; a change that
  improves one must be checked against the other.
- Confirmation means the reserved holdout agreed with validation after costs. On a
  planted panel that is expected; on real data it is evidence for deeper research only.
- Panel Sharpe ratios above come from artificial signals chosen to be detectable and
  say nothing about achievable returns.

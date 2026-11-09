# Research domains: autonomous alpha discovery and machine learning

The lab harness (`autoresearch/lab`) runs search campaigns for any research
domain that implements its contract. Two domains ship with it:

| Domain | A candidate is | Generators | Scored by | Selection control |
| --- | --- | --- | --- | --- |
| **alpha** — autonomous alpha discovery | a FASTEXPR expression | templates, unit-checked random trees, mutation, crossover, development-only ensembles, promoted blends, seeds, agent submissions | Alpha Lab, on period-sealed data | deflated Sharpe ratio over every validation trial, cost/sub-period/parameter vetoes |
| **ml** — machine learning | a training recipe `{family, params}` | defaults, random sampling, mutation, agent submissions | tabular loss against a baseline, or Alpha Lab for panel predictions | Bonferroni-adjusted paired bootstrap (tabular) or the deflated Sharpe ratio (panel) |

The grounded Research loop ([Research campaigns](USER_GUIDE.md#research-campaigns)) stays as it is:
it is the source-and-quote path for hypotheses an LLM proposes. The lab
domains are the high-throughput path: thousands of screened candidates, a
few dozen validation trials, and the bookkeeping that keeps that honest.

## The cascade

```
propose ─► dedupe (canonical key) ─► screen: development data only
        ─► population of the fittest ─► shortlist: domain gates + diversity
        ─► validate: one counted trial each, reserved before it runs
        ─► review: gates, multiple testing, robustness ─► promote
        ─► finalize: reserved holdout, once per candidate, explicit command only
```

Every screen, validation, review, redundancy decision and holdout access is a
line in the campaign's **trial ledger** (`<records>/ledgers/<campaign>.jsonl`),
each carrying the SHA-256 of the previous line. Editing, deleting or
reordering an entry breaks the chain; a torn final line after a crash is
trimmed and the repair itself is recorded. `tubby-lab verify` checks the chain
and that the checkpoint's counts equal the ledger's.

Rules the harness enforces for every domain:

- **Sealing is structural.** The alpha screen runs on an in-memory workspace
  whose price arrays end at the development end; validation probes on one that
  ends at the validation end. The ML panel task computes its features on the
  validation-sealed workspace and purges training rows whose target is
  realized after their split. Tabular tasks never keep holdout rows in memory.
  Only `finalize` loads the reserved period.
- **Everything tried is counted.** Validation trials are reserved in the
  checkpoint before they run; an interrupted validation becomes an
  `interrupted` experiment and still counts. The review sees the campaign's
  validation-trial count, and final selection re-applies the test with the
  final count before anything reaches the holdout.
- **Gates are identity, budgets are not.** Policy thresholds, settings, data
  and code hashes form the campaign ID, so changing a gate starts a new
  campaign rather than rescuing a candidate. Budgets (`limits`) can be raised
  to continue a campaign.
- **Evidence belongs to the data, not to a campaign.** Each domain names its
  *lineage*: the data, universe and periods a validation trial looks at, and
  the data and reserved period a final reads. Campaigns in one record store
  with the same lineage share their validation-trial count (the multiple-testing
  count every review uses) and their holdout accesses, so a new campaign made by
  nudging a gate, a seed or the code cannot reset either.
- **One holdout access per candidate and lineage.** Access is reserved in the
  ledger, the record store and the checkpoint before the evaluation. A repeat
  is refused unless `--allow-reuse`, recorded with its access sequence and
  never confirmed. Hypotheses selected after any holdout of the lineage was
  read carry `post_holdout: true`, and their finals are never confirmed either.
- **Crash-consistent.** An interrupted round or validation is completed on the
  next start from the records and the ledger, so the checkpoint, the store and
  the ledger agree and no trial is counted twice or lost.
- **Agents propose, nothing more.** A submitted candidate is screened, counted
  and reviewed like a generated one. Only the expression or recipe and a
  rationale are taken from a submission; complexity, family and grouping are
  recomputed. The inbox is taken whole at the start of each round, so its size
  cap applies to what is waiting.

## Quick start

```sh
tubby-lab domains
tubby-lab demo alpha --root ~/tubby-lab/alpha-demo      # planted panel, 8 rounds, holdout, report
tubby-lab demo ml --root ~/tubby-lab/ml-demo            # Friedman #1 regression
tubby-lab demo ml --task panel --root ~/tubby-lab/ml-panel-demo
tubby-lab benchmark alpha --root ~/tubby-lab/bench-alpha
tubby-lab benchmark ml --panel --root ~/tubby-lab/bench-ml
```

Demos and benchmarks use artificial data only. For a real campaign, copy
`lab-alpha.example.json` or `lab-ml.example.json` at the repository root,
point `settings.dataset` at a
prepared snapshot (or `settings.table` at a CSV) and run:

```sh
tubby-lab run lab-alpha.local.json --rounds 10           # stops early when a budget is reached
tubby-lab report lab-alpha.local.json --markdown
tubby-lab submit lab-alpha.local.json --expression "-rank(ts_delta(close, 3))" --rationale "short reversal"
tubby-lab finalize lab-alpha.local.json                  # the only command that reads the holdout
tubby-lab export lab-alpha.local.json --output promoted.txt   # Alpha Lab batch file
tubby-lab verify lab-alpha.local.json
```

Records default to `<data root>/autoresearch/lab/<domain>/records`, the
checkpoint and inbox to `.../runtime`. `research_root` and `runtime_root`
override them. Ctrl-C stops between rounds.

## Campaign file

```json
{"name": "us-daily-discovery", "domain": "alpha", "seed": 7,
 "settings": {"dataset": "../datasets/sharadar-2026", "universe": "TOP1000", "cost_bps": 5, "warmup": 130,
              "development_start": "2014-01-02", "development_end": "2018-12-31",
              "validation_start": "2019-01-02", "validation_end": "2022-12-30", "holdout_start": "2023-01-03",
              "search": {"max_depth": 4, "seeds": ["-rank(ts_delta(close, 5))"]}},
 "policy": {"min_sharpe": 1.0, "min_dsr": 0.95},
 "selection": {"max_correlation": 0.7},
 "limits": {"max_rounds": 40, "proposals_per_round": 64, "validations_per_round": 3, "max_validation_trials": 80}}
```

| Limit | Default | Meaning |
| --- | --- | --- |
| `max_rounds` | 25 | rounds before the campaign completes |
| `proposals_per_round` | 48 | candidates generated per round (before deduplication) |
| `max_screen_trials` | 2000 | development screens in total |
| `validations_per_round` | 3 | shortlisted candidates validated per round |
| `max_validation_trials` | 60 | validation trials in total (the multiple-testing count) |
| `max_promotions` | 20 | promotions before the campaign completes |
| `population_size` | 40 | fittest screened candidates kept as parents and shortlist pool |

`selection.max_correlation` is the redundancy threshold on development
signatures (absent: 0.7 for alpha, 0.98 for ml). Blends are compared with
other blends, not with their own components.

## Alpha domain

**Grammar.** Search uses a unit-aware subset of FASTEXPR: price (P), volume (V),
dollar (D), log level (L), ratio (R) and score (S). Adding a price to a volume,
dividing by a quantity that can be zero, or a root that is not dimensionless is
rejected before any data is touched. Mutation edits windows, operators,
fields and subtrees, wraps or hoists nodes; crossover swaps unit-compatible
subtrees; lookback is capped below the warmup. The candidate key is the parsed
Alpha Lab AST with a root negation stripped, so `x` and `-x` are one candidate.

**Screen.** Alpha Lab on the development period, oriented: a signal with a
negative development Sharpe is negated. Fitness is the weakest of three
development sub-period Sharpe ratios minus a small complexity penalty, which
rewards consistency over a single lucky stretch. The shortlist also requires
the full development Sharpe at least `min_dev_sharpe` and every fold positive.

**Ensembles and blends.** `ensemble` is the equal-weight rank blend of mutually
uncorrelated development leaders, chosen on development data only, so its
validation is a fair test. `combination` blends promoted alphas; its parts were
selected on validation, so only its holdout is independent evidence, and its
review says so.

**Validation and review.** The canonical Research backend scores development
and validation separately (same records as Research). Veto-only probes on the
validation-sealed workspace measure the Sharpe ratio at `cost_stress` times
costs, on each half of validation, and with every window scaled by 0.75 and
1.33. Review applies the Research gates (days, Sharpe, returns, turnover,
drawdown), then the **deflated Sharpe ratio** of the candidate's daily
validation PnL against the expected best of the campaign's validation trials,
then the vetoes.

`dsr_variance` chooses the benchmark: `null` (default) uses the null sampling
variance 1/T, a familywise test comparable to Bonferroni; `empirical` uses the
observed cross-trial variance (Bailey and López de Prado), which is stricter
when the trials include genuinely different signals. Correlated trials make
the count conservative.

| Policy key | Default | |
| --- | --- | --- |
| `min_dev_sharpe` | 0.75 | shortlist floor on development |
| `min_validation_days`, `min_sharpe`, `min_returns`, `max_turnover`, `max_drawdown` | 60, 1.0, 0, 0.7, 0.3 | Research gates on validation |
| `min_dsr`, `dsr_variance` | 0.95, `null` | deflated Sharpe ratio |
| `cost_stress`, `min_half_sharpe`, `min_perturbation_retention` | 2.0, 0, 0.5 | robustness vetoes |
| `min_holdout_days`, `min_holdout_sharpe` | 20, 0 | final confirmation |

**Generators from outside.** `search.seeds` adds expressions;
`search.seed_research_root` seeds promoted and refined hypotheses from a
Research store. Agents submit through MCP (`lab_submit`) or `tubby-lab submit`.

## Machine-learning domain

**Model families** (`autoresearch/domains/ml/models.py`, numpy only):
`constant` (baseline), `ridge`, `logistic`, `gbdt` (histogram gradient boosting
with learned missing-value directions, row/column subsampling, early
stopping), `random_forest` (bootstrap, per-split feature sampling) and `mlp`
(AdamW, early stopping). Every model serializes to JSON and reloads with
identical predictions; promoted and final models are saved under
`<runtime>/domain/models/`.

**Tasks.** `tabular`: any numeric CSV (`target`, optional `order_by` for a
chronological split, `problem` regression or classification) or a built-in
task — `friedman1` (irreducible RMSE 1.0), `interaction` (Bayes-optimal log
loss reported on the holdout) and `null_regression` (no relation at all).
`panel`: cross-sectional prediction of the delay-1 next-session return on an
Alpha Lab dataset from ranked FASTEXPR features (a 22-feature library, plus
`extra_features`, for example promoted alphas). Panel predictions are injected
into the unmodified Alpha Lab simulator as a data field, so an ML signal and a
FASTEXPR alpha are measured identically.

**Search.** Round one validates each family's defaults; later rounds mix random
recipes with mutations of tournament winners (one or two parameters
perturbed, occasionally a family jump). The screen trains on a common
subsample of the development split's first 75% and scores its last 25%
(common random numbers: identical recipes look identical). Validation trains
on the whole development split.

**Review.** Tabular: the improvement over the constant baseline must have a
paired-bootstrap p-value at most `alpha_level / validation trials` and a
relative improvement of at least `min_relative_improvement`. Panel: the alpha
domain's gates, deflated Sharpe ratio and cost/sub-period vetoes. Finalize
refits on development plus validation and scores the holdout once.

## From the console

The **Discover** view runs both domains without a campaign file: a preset
(Quick, Standard, Thorough), a dataset and universe, costs and gates become the
campaign file shown in the form's Plan box, written under
`<data root>/autoresearch/lab/<domain>/campaigns/` and run by the same
`tubby-lab` program in a worker process. Each domain's campaigns share one
record store (`.../records`), so lineage counting and holdout accounting work
across console and command-line campaigns alike. The console reads progress
from the worker's output, results from the record store and the ledger, and
exposes submit, finalize (with confirmation, never `--allow-reuse`), verify,
export and a budget extension that raises only operational limits. See the
[console quick start](CONSOLE_QUICKSTART.md) and the [workbench guide](USER_GUIDE.md).

## Agents (MCP)

`tubby-alpha-mcp` adds two tools. `lab_report` (read-only) lists a domain's
campaigns or reports one, recounting the funnel from its ledger.
`lab_submit` queues one expression (alpha) or recipe (ml) for the next round of
an existing campaign under the data root's default lab location. The server
never runs a campaign or reads a holdout.

## Benchmarks

Synthetic tasks with known answers; they measure the pipeline, not a market.

- **alpha** — two 300-name, 2,000-session panels with identical noise: one with
  three planted signals (five-day reversal, abnormal volume, low volatility),
  one with none. Reported: recall (best mean daily rank correlation between a
  promoted alpha and each planted oracle, recovered at 0.5 or above), false
  promotions on the null panel, holdout confirmations, trial counts.
- **ml** — Friedman #1, interaction classification, a null regression and,
  with `--panel`, a planted and a null cross-sectional panel.

Latest results are recorded in [LAB_BENCHMARKS.md](LAB_BENCHMARKS.md).

## Extending: a new domain

Subclass `autoresearch.lab.domain.Domain` (identity, propose, key, screen,
validate, selection_score, review, and optionally final_selection,
evaluate_holdout, assess_final), register it in `REGISTRY`, and keep sealing
structural. The harness supplies budgets, the ledger, records, diversity,
holdout accounting, reports, the CLI and MCP reporting.

## Limits

- The alpha domain does not call a language model itself; LLM ideas arrive
  through the grounded Research loop (as seeds) or through an agent over MCP.
- Search runs in one process; Alpha Lab evaluations take milliseconds on the
  benchmark panels and grow with universe size and history.
- The deflated Sharpe ratio assumes approximately independent trials; families
  of near-duplicates make it conservative, not lenient.
- Promotion and confirmation mean evidence for deeper research and Qlib
  validation, not a trading recommendation.

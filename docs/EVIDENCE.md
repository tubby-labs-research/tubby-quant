# Evidence: saved candidates, validation and portfolios

The **Evidence** view of the console and the `tubby-qlib workflow` commands
record every submitted attempt in an append-only, sealed store under the data
root (`<data root>/workflow`). From a saved Alpha Lab result you can preview and
run a signed Qlib validation, combine one or two validated signals in one
shared account with explicit risk limits, and declare a chronological protocol
whose final window is scored once. These operations produce historical
simulations on data you prepared; they place no trades and download nothing.

## Read the attempt, not just the library row

Open **Evidence · validation and portfolios**, then **Refresh** and select an
attempt. History shows the latest 100 records; the structured listing accepts
1–500. Open parents to trace console submission, worker, Lab evaluation, linked
validation and portfolio. Different levels can have different attempt IDs.

- `candidate_id` identifies the candidate; `id` identifies one submitted attempt;
  `result_digest` binds its immutable result version. Keep all three when handing
  work off. A mutable Alpha Lab library/cache key cannot replace an attempt ID.
- Outcomes include `completed`, `reused`, `rejected`, `failed`, `cancelled` and
  `interrupted_unknown`; unfinished records remain `pending`. A cache hit is
  recorded reuse, not another independent trial. Submitted invalid Runs count;
  editor typing/checking does not.
- Older library entries remain legacy evidence without complete attempt history.
  Rerunning through Alpha Lab records a new attempt; it cannot recover unlogged
  past searches or retroactively qualify an old result.
- **Recover unfinished attempts · check ownership** checks process ownership.
  Only confirmed-dead owners are marked interrupted/unknown; live or unverifiable
  owners remain pending. Recovery writes outcomes; it does not rerun simulations.

Listing is metadata only. Opening an attempt verifies its record/result seals
and returns compact details; it does not verify all underlying source data.
Use **Open saved Qlib run** or **Open saved portfolio run** for detailed saved
account evidence. Inspect errors, qualification, modeled costs and omissions
alongside metrics. Lab annual return/drawdown use half-book capital; Qlib uses
net account returns. They are different measurements. Seals detect corruption;
they are not signatures against someone able to rewrite the whole store.

## Saved Lab result → preview → signed Qlib

1. Select a `lab-evaluation` attempt with `completed` or `reused` status and a
   sealed result. Set the new account cash, trading cost and annual borrow rate.
   Saved Lab settings stay read only.
2. Choose **Preview selected validation**. Review the exact source, universe,
   period, settings, qualification and execution differences. A preview checks
   recorded bindings and current implementation compatibility; source-byte
   verification and numerical reproduction happen in the run.
3. Choose **Run selected validation**. The isolated worker verifies sources,
   reproduces the saved Lab evidence, constructs targets and executes the
   signed Qlib account. Refresh history and inspect the completed
   `candidate-validation` attempt and linked native run. Stop is available while
   the worker is cancellable. Changed inputs require a new preview.

The adapter transfers **already transformed target weights**, not translated
FASTEXPR or raw scores. Lab delay, decay, pasteurization, neutralization and
truncation are applied once. Signed Qlib consumes these close-decision rows at
the following session open, without another normalization, delay or gross-up.
Subunit target gross stays subunit. Lab delay 0 remains idealized same-close
screening evidence; this handoff still executes the saved close decision at the
next open.

Lab uses fixed-book adjusted close-to-close PnL, its configured turnover costs,
and no stock borrow. Qlib uses whole shares, available cash, restricted short
proceeds, transaction fees and stock-borrow charges. Equal return series are
not expected. Missing/blocked fills can leave actual holdings away from targets.
The validation policy has no cash borrowing and a zero annual cash rate. It
omits real borrow availability/recalls, broker margin/settlement, market impact,
taxes and unmodeled exchange/regulatory charges.

Refusals preserve the saved settings: missing/changed provenance or runtime,
pasteurization off, insufficient warmup, absent matching native universe,
missing later calendar boundary, unsupported factor basis, or incompatible
selected SEC/dated-group bindings. The adapter requires at least one warmup
session, OHLCV/factor fields and identity factors over the selected window. It
never silently widens an implicit TOP-N subset. Byte seals and reproduction do
not establish historical economic point-in-time validity; identity, delistings,
adjustments, filing availability and coverage remain source-specific limitations.

## One or two signals, one cash budget

In **Portfolio · one shared account**, select one or two distinct completed
`candidate-validation` attempts. For a single signal use a coefficient such as
`1`; for two, supply fixed capital fractions such as `0.4` and `0.6`. Every
coefficient must be in `[0,1]` and their sum at most one. A smaller sum leaves
capital unallocated. Allocation combines saved target matrices as
`c1 × targets1 + c2 × targets2`; it adds no neutralization or normalization and
does not add independent full-capital return series.

Legs must share the exact price source, execution period and prepared universe;
target dates/symbol axes must match. Two validations of the same Lab attempt
cannot count as two signals. Give the simulation an account ID, initial USD
cash, costs and explicit limits, then preview and run. Each run starts a fresh
historical account; reusing its name does not resume previous positions.

All seven limit keys must be present in structured requests. `null` disables a
limit; `0` enables a zero bound. UI empty fields mean disabled. All bounds are
fractions, with `1 = 100%`:

| Key | Bound |
| --- | --- |
| `gross` | Sum of absolute exposures / positive account equity |
| `net` | Absolute signed net exposure / equity |
| `name` | Absolute exposure of each name / equity |
| `group_gross` | Sum of absolute member exposures per group / equity |
| `group_net` | Absolute signed exposure per group / equity |
| `turnover` | Full absolute traded reference notional / equity, with no half factor |
| `liquidity` | Absolute traded shares / prior-session volume in shares |

Enabled group limits require both a dated `group_overlay` under this root's
`classifications/` and a `group_level` of `sector`, `industry` or `country`.
Groups use the prior decision-session open cutoff; liquidity uses that session's
final volume. Unknown required prices, volumes or groups are missing inputs.
Positive equity and no-loan funding checks apply even with all limits disabled.

Independent pre-order checks assess the joint targets/proposed holdings and
gross proposed trades, including reversals. A failed check blocks that
rebalance and retains actual positions. Post-fill checks report actual holdings,
gross fills, unfilled quantities and residual exposure; they do not undo fills.
A zero target does not erase an untradeable position. Read
`qualification.risk_outcome`, `risk_counts`, `summary.portfolio_risk` and saved
pre/post records: `completed` can coexist with `exceptions_recorded`.
Reconciliation checks the shared signed ledger, not strategy quality.

## Structured CLI and local API

The examples are request shapes, not executed receipts. Replace the uppercase
identity placeholders with returned IDs and use your own data root, request
files and calendar. The shown costs are examples: `cost_per_side: 0.0005` means
**5 basis points per side** and `annual_borrow_rate: 0.004` means **0.4% per
year**; the console displays `5` and `0.4`. Lab's separate `cost_bps` setting
uses basis points. `tubby-qlib` is the installed name of `python -m qlib_yahoo`.

```bash
tubby-qlib --data-root ~/tubby-data workflow attempts --limit 100
tubby-qlib --data-root ~/tubby-data workflow show LAB_ATTEMPT_ID
tubby-qlib --data-root ~/tubby-data workflow preview LAB_ATTEMPT_ID --initial-cash 100000 --cost-per-side 0.0005 --annual-borrow-rate 0.004
tubby-qlib --data-root ~/tubby-data workflow validate LAB_ATTEMPT_ID --initial-cash 100000 --cost-per-side 0.0005 --annual-borrow-rate 0.004
tubby-qlib --data-root ~/tubby-data workflow portfolio-preview --request portfolio.json
tubby-qlib --data-root ~/tubby-data workflow portfolio-run --request portfolio.json
```

The same candidate parameters are the JSON body for
`POST /api/workflow-preview` and `POST /api/workflow-validate`:

```json
{"attempt_id":"LAB_ATTEMPT_ID","initial_cash":100000,"cost_per_side":0.0005,"annual_borrow_rate":0.004}
```

Example `portfolio.json`, also the body for
`POST /api/workflow-portfolio-preview` and `POST /api/workflow-portfolio-run`:

```json
{
  "validation_ids": ["VALIDATION_A_ID", "VALIDATION_B_ID"],
  "coefficients": [0.4, 0.6],
  "account_id": "two-signal-example",
  "initial_cash": 100000,
  "cost_per_side": 0.0005,
  "annual_borrow_rate": 0.004,
  "limits": {"gross":1,"net":1,"name":0.1,"group_gross":null,"group_net":null,"turnover":2,"liquidity":0.01},
  "group_overlay": null,
  "group_level": null
}
```

These illustrative bounds can block a run's rebalances; they are not an accepted
investment policy. For one signal use one validation ID and one coefficient.

API requests use the running console's loopback origin, its authenticated
session cookie and, for POST, a matching `Origin`, the `X-Token` header and a
JSON content type. Use the existing authorized session; never save its token in
requests or documentation. `GET /api/workflow-attempts?limit=100` lists
metadata; `GET /api/workflow-detail?id=ATTEMPT_ID` opens compact evidence;
`POST /api/workflow-recover` with `{}` performs ownership recovery. Execution
POSTs return a job record, not a finished simulation: poll
`GET /api/job?id=JOB_ID&since=0`, advance `since` using `log_index`, and inspect
`state`, `finished`, `error` and `result`. `POST /api/cancel` with `{}` requests
cancellation of the current job.

CLI execution sends progress to stderr and waits for the final evidence seal
before printing its JSON result to stdout. Candidate/portfolio results contain
`run_path`; chronology trial/final results also contain `job_id`, `attempt_id`
and `receipt`. Ctrl+C requests cancellation, including during worker startup;
the CLI waits for the recorded terminal outcome. Job-level success does not
replace inspection of child attempts and risk outcomes.

## Chronology: declare → trial → freeze → final

In Evidence's chronology controls, pin a saved Lab source, preview its windows
and exposure warning, then declare. Refresh the protocol after declaration and
after each job. Submit supplied expressions on development/validation only;
load evaluated candidates, select successful numerical attempts, record the
decision reason and freeze. Review frozen expressions and exposure before final.
Chronology runs Lab screening; it does not itself run signed Qlib validation.

The CLI actions are `chronology-preview`, `chronology-declare`,
`chronology-trial`, `chronology-freeze`, `chronology-final` and `chronology-view`,
each taking `--request FILE.json`; each also has
`POST /api/workflow-chronology-ACTION` with the same body. Example
`declaration.json` (dates must exist in the source's recorded calendar):

```json
{
  "source_attempt_id": "LAB_ATTEMPT_ID",
  "development": {"start":"2021-01-04","end":"2021-02-12"},
  "validation": {"start":"2021-02-16","end":"2021-03-31"},
  "final": {"start":"2021-04-01","end":"2021-05-14"},
  "selection_rule": {"rule":"Review supplied candidates on development and validation; record the selection reason before final."}
}
```

Use the returned `protocol_id` in subsequent request files:

| Action | JSON body |
| --- | --- |
| `trial` | `{"protocol_id":"PROTOCOL_ID","phase":"development","expressions":["rank(close)","-rank(close)"],"workers":1,"force":false}` |
| `trial` on validation | Same shape with `"phase":"validation"` and the explicitly supplied expressions |
| `view` | `{"protocol_id":"PROTOCOL_ID"}` |
| `freeze` | `{"protocol_id":"PROTOCOL_ID","selected_ids":["SUCCESSFUL_LAB_TRIAL_ID"],"decision_reason":"Recorded review of development and validation evidence."}` |
| `final` | `{"protocol_id":"PROTOCOL_ID","selection_id":"SELECTION_ID","workers":1,"force":false,"allow_exposed":false}` |

The declaration freezes the source/settings/recipes; trials can use only those
SEC/group inputs. Windows are inclusive, chronological and nonoverlapping, with
at least 21 sessions each and the exact preceding warmup. The selection-rule
object records human policy; no automatic statistical acceptance is implied.
Without optional `samples`, `overlap_check` is `not_available`: this is a calendar
split, not verified purging. Explicit samples contain `id`, `session`,
`information_start`, `information_end`, and optionally `feature_available_at`,
and must cover every session in all three windows; empty or partial lists are
reported as partial checks and cannot dispatch evaluations. With complete
coverage, any excluded sample excludes that day's portfolio-return observation
while signal calculation, rolling history, weights and costs still use the
continuous window. Filtered summaries annualize the mean daily result of the
retained observations with 252 and take drawdown over their compressed PnL;
they are not a continuous investment-return claim. At least two scored
observations must remain. Filtered candidates can be selected within the
chronology workflow but cannot be sent to Qlib target validation, which needs a
continuous trading candidate. Filtered scores use a separate chronology cache
and do not appear as ordinary results in the Alpha Lab library; their immutable
attempts remain discoverable through Evidence.

Freeze discovers the **complete recorded protocol population**, including
failures, rejected trials, grouping records and reuse, not only the visible
page. Optional `population_ids` must match that full population. Pending trials
prevent freezing, and subsequent training under that protocol is refused. Final
executes only the frozen selected expressions, with their settings and lineage
checked again.

Final access is reserved **before numerical execution and source-byte
verification**, so failure, cancellation or a crash can consume that access.
Labels are `first_recorded_access`, `reused_holdout` or
`already_exposed_holdout`. Overlapping reserved final windows on the same
underlying price lineage count as reuse, including across new protocols.
`force` controls Lab cache reuse; it does not reset holdout history. Known or
uncertain source/trial exposure requires explicit `allow_exposed: true` and
remains labeled exposed. A declaration made after seeing the final interval
cannot restore an untouched holdout. `first_recorded_access` describes this
ledger only: the exposure audit follows other recorded protocols, their sources
and trials, not every unrelated Lab result, external notebook or human
inspection. Use `view` to inspect population, frozen selection and access labels
without hydrating final numerical results.

## Source of these semantics

Examples and semantics come from the source modules shipped in this tree:
`qlib_yahoo/workflow_cli.py` (CLI), `qlib_yahoo/webapp/server.py` (HTTP routes),
`qlib_yahoo/webapp/service.py` (service), `qlib_yahoo/webapp/static/workflow.js`
(Evidence UI), `qlib_yahoo/workflow_records.py` (records),
`qlib_yahoo/candidate_workflow.py` (candidate reconstruction),
`qlib_yahoo/validation_workflow.py` (signed validation),
`qlib_yahoo/portfolio_workflow.py` (portfolio workflow), `qlib_yahoo/portfolio.py`
(risk definitions), `qlib_yahoo/portfolio_runtime.py` (runtime risk hooks),
`qlib_yahoo/chronology_workflow.py` (chronology dispatch) and
`qlib_yahoo/chronology.py` (holdout reservations). Interface availability does
not establish economic readiness; read [limitations](LIMITATIONS.md).

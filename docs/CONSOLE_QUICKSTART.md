# Console quick start: a first session in ten minutes

The console opens on **Start** when the data folder holds no prepared prices.
Start walks through three steps and opens the view each step belongs to, so the
rest of the workbench is never a surprise. Everything runs on this computer:
nothing is traded, no broker is connected and no result is investment advice.

## 1. Get prices

| Way | What happens | When to use it |
| --- | --- | --- |
| **Create demonstration data** | Writes `datasets/demo-planted-200` in a few seconds: 200 fictional stocks over six years with three planted signals (short-term reversal, a volume shock, low volatility). The manifest says `artificial: true` and every view labels it. | Learning the tools, testing an installation, demonstrations. Never research evidence. |
| **Download US daily prices (Yahoo)** | The existing refresh job: Nasdaq screener → Yahoo charts → consolidated table → TOP3000 snapshot → quality scan. About an hour; resumable. | Real research with free data. |
| **Import a table…** | Opens Data with the import section: Stooq, Sharadar or HistoricalData tables you already have. | Licensed or offline data. |

The demonstration data exists so that the examples have something to find. The
discovery demo recovers the planted signals, and the factor-analysis demo's
importance chart should rank `ret_5`, `log_volume_shock` and `vol_20` highly.
If a search finds nothing on it, something is broken.

## 2. Simulate an alpha

**Simulate these three now** fills Alpha Lab with three starter alphas and runs
them on the selected prices:

```
-rank(ts_delta(close, 5))
rank(log(volume) - ts_mean(log(volume), 20))
-rank(ts_std_dev(returns, 20))
```

Alpha Lab is BRAIN-style: bare field names, one alpha per line, a leading minus
is fine. Each run is saved to the library with its settings and the dataset's
identity. The universe menu offers what the selected prices actually contain
(the demonstration data has TOP200; a Yahoo snapshot has TOP3000 and smaller cuts).

Validation (⌘1) uses Qlib's own expression syntax (`$close`, `Mean(...)`),
which has no unary minus on features; when you type one, the editor offers a
one-click rewrite to `-1 * (...)`.

## 3. Let the machine search: Discover (⌘5)

Discover is the console face of the lab harness described in
[LAB_DOMAINS.md](LAB_DOMAINS.md). Two domains share one guided view:

- **Alphas** — autonomous alpha discovery: typed FASTEXPR search (templates,
  random trees, mutation, crossover, your seed ideas, ensembles), screened on
  development data only, validated by Alpha Lab, reviewed with the deflated
  Sharpe ratio and vetoed by cost, sub-period and window-perturbation checks.
- **Models** — machine learning: training recipes (ridge, logistic, gradient
  boosting, random forest, MLP) searched the same way. The default task when
  prices exist is **factor analysis on a price panel**: the factor library
  (22 price/volume features) plus your own FASTEXPR factors predict the next
  session's cross-sectional return rank, and the predictions are traded by the
  unmodified Alpha Lab simulator, so a recipe is judged like an alpha. Built-in
  tabular tasks with known answers and your own CSV tables are also available.

### Setting up a campaign

The form on the right asks for a name, a **preset** (Quick, Standard,
Thorough: round, proposal and validation budgets), the prices and universe,
cost per side and, for alphas, a minimum validation Sharpe and optional seed
expressions. The **Plan** box shows the sealed periods cut from the dataset's
calendar (55 % development, 25 % validation, 20 % reserved holdout after a
130-session warm-up) and the exact campaign file that `tubby-lab` will run.
**Create & run** starts the campaign in its own worker process.

### Reading a campaign

- The stage strip shows where the campaign is: propose → screen → shortlist →
  validate → review → promote → finalize.
- Tiles count rounds, screened and validated trials, promotions, finalized
  candidates and the state of the hash-chained trial ledger.
- **Progress by round** charts the best screen fitness per round; the strip
  beneath it reads screened / validated / promoted for each round.
- **Which factors mattered** (Models) averages the split-gain importance of the
  promoted tree models over the factor library.
- The leaderboard lists every validated candidate with its decision and
  reason. Click a row for its full evidence: development, validation and
  holdout metrics side by side, the robustness probes, the deflated Sharpe
  ratio or Bonferroni level, the recipe's parameters and its feature importance.

### Acting on it

- **Run** continues the campaign for 1 or 3 rounds or until the budget is used;
  **Stop** pauses between reserved trials.
- **Queue your own idea** adds an expression (or a recipe) to the next round's
  inbox; it is screened like every generated candidate.
- **Finalize on holdout…** scores the promoted candidates once on the reserved
  period after an explicit confirmation. A repeated access is refused in the
  console; the command line can label a reuse. Candidates proposed after a
  holdout read are never confirmed.
- **Verify ledger** recomputes the hash chain and compares it with the checkpoint.
- **Open promoted in Alpha Lab** (Alphas) pastes the promoted expressions into
  Alpha Lab on the campaign's prices; **Export** writes a batch file or the
  recipes as JSON under `autoresearch/lab/<domain>/exports/`.
- **Add budget** raises a finished campaign's round, validation and promotion
  limits by one preset step. Gates and the campaign identity stay as they are:
  changing a threshold is a new campaign, never a rescue.

### Where files live

```
<data root>/autoresearch/lab/<domain>/
  campaigns/<slug>.json          the campaign file tubby-lab runs
  campaigns/<slug>.console.json  console bookkeeping (preset, periods, campaign id)
  records/                       the shared record store and trial ledgers
  runtime/                       checkpoints, inboxes, engine caches, saved models
  exports/                       batch files and recipe exports
```

Campaigns of one domain share one record store, so validation-trial counts and
holdout accesses are pooled across campaigns on the same data, as the harness
requires. The same campaign file works from the command line
(`tubby-lab status|report|verify campaign.json`).

## What the demonstration shows

On the planted data, a Quick alpha campaign takes about two seconds per round;
a Quick factor-analysis campaign about 45 seconds per round (features are
computed once, then each recipe trains on up to 200,000 panel rows). Promotion
and confirmation are evidence for deeper research, not a trading recommendation,
and on artificial data they are evidence about the tools, not about markets.

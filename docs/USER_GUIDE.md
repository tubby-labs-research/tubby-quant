# Use the workbench

Launch using [INSTALL.md](INSTALL.md). A data folder without prepared prices
opens on **Start**, which creates demonstration data, simulates three starter
alphas and opens Discover ([console quick start](CONSOLE_QUICKSTART.md)). Then
select your own prepared datasets for market research. No data subscription,
provider account or trade execution is supplied.

## A first session (Start)

When the data folder holds no usable prices the console opens on **Start**
(`⌘0` brings it back at any time). Three cards take you from an empty folder
to a tested idea, and every button opens the view that step belongs to:

1. **Get prices.** *Create demonstration data* writes `datasets/demo-planted-200`
   in a few seconds: 200 fictional stocks over six years with three planted
   signals (a five-day reversal, a volume shock, low volatility). The manifest
   says artificial and the interface labels it as demonstration data wherever it
   appears. *Download US daily prices (Yahoo)* is the real-data refresh (about an
   hour, resumable); *Import a table…* opens the import section of Data.
2. **Simulate an alpha.** *Simulate these three now* fills Alpha Lab with three
   starter expressions and runs them; *Open Alpha Lab* only opens the editor.
3. **Let the machine search.** *Run a short demo campaign* runs a Quick
   alpha-discovery campaign on the demonstration data; *Discover alphas* and
   *Search model recipes* open Discover's new-campaign form.

The checklist on the left marks prices ready, an alpha simulated and a campaign
run; the panel on the right shows the signed-engine state, the data folder and
the version. The demonstration data exists so that the examples have something
to find: the discovery demo should recover the planted signals, and the
factor-analysis demo's importance chart should rank `ret_5`,
`log_volume_shock` and `vol_20` highly. Nothing computed on it has economic
meaning.

## Screen in Alpha Lab

1. Select **Alpha Lab**, then its **Trading prices** snapshot and named universe.
   Its price selection is independent of Qlib Validation's.
2. Enter a FASTEXPR expression, such as `-rank(ts_delta(close, 5))`. Lab uses bare
   field names. Multiple lines form a batch; semicolons separate assignments.
3. Review dates, warm-up, delay, decay, neutralization, stock-weight cap and costs.
4. Run and inspect the leaderboard, coverage, charts and saved library entries.

Delay 1 uses the previous session's signal for weights held over later
close-to-close returns. Delay 0 is idealized same-close execution. Lab uses a
fixed book, default zero turnover cost and no stock-borrow expense. Set costs
explicitly. Unrecognized names or invalid syntax are rejected before a run.
Undefined numeric math yields missing values, not zero signals.

Industry/sector/country neutralization needs compatible group data. Dated labels
must be selected separately from fundamentals. SEC-derived labels use SIC,
coarse SIC divisions and business-address countries. Unknown/expired labels and
singleton groups cannot supply balanced positions. These are not market-beta
neutrality or a universal industry taxonomy.

Use **SEC fields** for fundamentals; see the [field guide](SEC_FIELDS_USER_GUIDE.md).
For scripts, `python -m qlib_yahoo alpha operators` lists supported operators.
Use the installed environment's Python and put global `--data-root` before `alpha`.

## Drive Alpha Lab from an agent (MCP)

The framework ships a Model Context Protocol server so Claude Code, Codex or
any MCP client can check and simulate alphas on your prepared data and read
research reports, without shell access to the machine:

```bash
claude mcp add tubby-alpha-lab -- tubby-alpha-mcp --data-root ~/tubby-data
```

(`python -m qlib_yahoo.alphalab.mcp --data-root ~/tubby-data` without the
package install.) Tools: `list_operators`, `check_expression`, `list_datasets`,
`simulate_alpha`, `simulate_batch` (leaderboard and correlated pairs),
`library_list` and the read-only `research_report`. Datasets are addressed by
folder name under the data root; paths outside it are refused. The server
performs no trading, no model calls, no data downloads, and never scores a
research campaign's reserved holdout. Metrics are Alpha Lab's fixed-book,
close-to-close results after the configured cost: research evidence, not a
trading recommendation.

## Validate with Qlib

Choose **Qlib Validation**, its dataset/universe and a native expression such as
`$close / Ref($close, 20) - 1`. Native expressions use `$` field prefixes and
Qlib operator names. FASTEXPR text is not automatically interchangeable.
Review mode, exposure, dates, costs and report options, then run.

Signed mode uses a verified derivative in a separate worker. It uses prior-close
signals and next-open fills, whole shares, fees, restricted short proceeds and
stock-borrow expense. CLI users must request `--strategy-mode signed`; its default
is long-only. Read [limitations](LIMITATIONS.md) before interpreting performance.

Saved results retain original settings and provenance. Report generation can
fail separately from simulation. Inspect diagnostics and retain the complete
run folder for offline reports. A matching expression alone does not establish
matching inputs, accounting or returns across engines.

## Evidence, validation and portfolios

Every submitted Lab, Qlib and portfolio attempt is recorded in a sealed,
append-only store under the data root. Open **Evidence** to trace an attempt to
its parents, preview and run a signed Qlib validation of a saved Lab result,
combine one or two validated signals in one shared account with explicit risk
limits, or declare a chronological protocol whose final window is scored once.
The `tubby-qlib workflow` commands and loopback API expose the same steps. See
[Evidence](EVIDENCE.md) for the exact contracts and refusals.

## Research campaigns

Research is a separate price/volume workflow using Alpha Lab evaluations.
Select prepared data and a named universe, chronological development/validation
dates, reserved holdout, explicit costs and budgets. **Reviewed source-backed
ideas** evaluates supplied cited plans without a model credential. **Try demo**
uses artificial fixtures. For model proposals, configure a trusted agent through
the [API guide](RESEARCH_AGENT_API.md), explicitly enable calls and start.

New campaign records default to `<data root>/autoresearch/records`; configs and
runtime artifacts also live beneath the external data root. Explicit CLI paths
in a campaign configuration remain in effect. Existing source-folder records
can be inspected without automatic copying or migration. Keep user records out
of the source distribution.

Stop saves bounded-step recovery state; app closure stops its worker. Continuous
mode runs while the app/process and machine remain available, within budgets.
Review retained failures as well as successes. Promotion means further research,
not Qlib validation or a trading recommendation. Research decisions never score
the reserved holdout. When promoted candidates exist, `python -m autoresearch
finalize <campaign.json>` scores each of them once on the holdout and records an
immutable final with a `confirmed` / `not_confirmed` verdict; a repeated access
is refused unless requested and is then labelled `reused`, and later hypotheses
are labelled `post_holdout`. `python -m autoresearch export <campaign.json>
--output promoted.txt` writes promoted (or `--decision confirmed`) expressions
as an Alpha Lab batch file for Lab screening and Qlib validation;
`python -m autoresearch report --markdown` summarises a campaign. The console
offers the same as **Finalize on holdout** and **Send promoted to Alpha Lab**
in the campaign header, shows finals beside each idea, and lists
**OpenAI-compatible · direct model** as a research agent (`OPENAI_API_KEY` on
the app process, or a loopback `QUANT_OPENAI_BASE_URL` for a local server).
Research campaigns do not currently use SEC inputs or dated classification overlays.

## Discover: autonomous alpha discovery and machine learning

Discover runs the lab harness without a campaign file: choose a domain
(**Alphas** searches typed FASTEXPR expressions; **Models** searches training
recipes, by default as factor analysis on a price panel), a preset, the prices
and universe, costs and gates. The campaign runs in its own worker process on
development data only, validates a few shortlisted candidates per round, reviews
them with the deflated Sharpe ratio and robustness vetoes, and scores the
reserved holdout once, after an explicit confirmation. Campaigns of one domain
share a record store and a hash-chained trial ledger, so the same campaign file
also works from the command line (`tubby-lab status|report|verify <campaign.json>`).
The rules, the campaign file format and the `tubby-lab` commands are in
[LAB_DOMAINS.md](LAB_DOMAINS.md); measured results on artificial panels are in
[LAB_BENCHMARKS.md](LAB_BENCHMARKS.md). Demonstration data is artificial and
labelled as such in every view; promotion on it is evidence about the tools,
not about any market.

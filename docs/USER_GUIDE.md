# Use the workbench

Launch using [INSTALL.md](INSTALL.md). Start with **Research → Try demo**, then
select your own prepared datasets for market research. No data subscription,
provider account or trade execution is supplied.

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
not Qlib validation or a trading recommendation. Reserved holdout is not scored.
Research campaigns do not currently use SEC inputs or dated classification overlays.

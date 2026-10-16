# Preview limitations

This early software preview includes a pinned Windows native helper and is
for local research. It places no trades
and provides no broker connection, investment recommendation or complete legal,
security, data-rights or platform clearance.

- **Different simulators:** Alpha Lab holds weights on a fixed book over
  close-to-close returns. Qlib uses an account/share ledger and next-open fills.
  Matching expressions or inputs do not establish return equivalence. Delay 0
  is idealized same-close execution, and default Lab turnover cost is zero.
- **Shorting assumptions:** signed Qlib defaults to 0.4% annual stock borrow,
  with every selected stock assumed borrowable and no recalls. Short-sale
  proceeds are restricted; cash loans are disabled. Broker margin calls,
  collateral revaluation and forced liquidation are absent. Expenses/losses
  can create an unfunded account; recorded accounting is not funding feasibility.
  Borrow costs, transaction fees and the Sharpe cash hurdle are separate.
- **Execution gaps:** impact, realistic liquidity, taxes, settlement and
  broker-specific constraints are incomplete or absent. Targets can differ
  from fills. Lab group neutrality is not a guarantee of actual/beta neutrality.
- **Historical data:** adjustment quality, delistings, changing tickers, share
  units and survivorship depend on the selected source. Current ticker mappings
  and static labels are not a historical security master. A hash verifies
  identity, not economic accuracy or data rights.
- **SEC definitions:** availability is inferred from filing dates; variants,
  missing coverage and revisions matter. Currency, TTM and tag equivalence are
  not automatically inferred. Raw-table access is broader than numeric inputs.
- **Search evidence:** repeated selection contaminates validation evidence.
  Research reserves but does not score holdout; promotion requests more research.
  Exact source quotations do not establish the economic claim or reproduce a paper.
  Campaigns currently exclude SEC and dated classification inputs.
- **Operations:** call/token limits are not a monetary guarantee. Sleep, app
  closure, provider errors or exhausted budgets stop Research. Local records and
  source snapshots need explicit retention; missing charts are not reconstructed.
- **Platform checks:** CPython 3.12 is required for the pinned signed runtimes.
  Windows x64 has isolated installation/recovery, local HTTP and small synthetic
  signed-run evidence, not universal clean-machine or desktop certification.
  macOS is experimental: its separate runtime profile has not been verified on
  a real macOS machine for this preview. Linux covers core checks only, with no
  signed runtime pin. Windows environment CI is not desktop/native-fill acceptance.
  Skips and bounded checks leave corresponding capabilities unverified.
- **Preview stability:** interfaces and persisted formats may change. Back up
  datasets and results separately. Declared dependency pins are not a complete
  transitive lock, and passing offline checks is not a security audit.

Review complete run settings and diagnostics before interpreting statistics.
Short-window annualized metrics and synthetic demo returns are especially weak
evidence. No runtime guarantee is made for your machine or dataset.

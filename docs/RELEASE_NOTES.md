# Release notes

## v0.2.0 — 2026-10-24

The second research preview makes the framework installable as a package,
usable from agents, and closes the research loop end to end.

- **Packaging.** `pip install -e .` installs Alpha Lab, Research, the console and
  the data adapters on Windows, macOS and Linux with the commands `tubby-quant`,
  `tubby-qlib`, `tubby-research` and `tubby-alpha-mcp`. Extras `qlib`, `table`,
  `desktop` and `research` mirror the pinned requirement files. The signed Qlib
  runtime still uses the platform setup scripts and their verification.
- **MCP server.** `tubby-alpha-mcp` exposes `list_operators`, `check_expression`,
  `list_datasets`, `simulate_alpha`, `simulate_batch`, `library_list` and a
  read-only `research_report` over stdio to Claude Code, Codex or any MCP client,
  contained in one data root; no trading, model calls or downloads.
- **Research loop.** `tubby-research finalize` scores promoted hypotheses once on
  the reserved holdout, reserving the access before the run and labelling any
  repeat as reused; `export` writes promoted, confirmed or all candidates as an
  Alpha Lab batch; `report --markdown` summarises a campaign; `"provider": "openai"`
  adds a chat-completions adapter for the OpenAI API and loopback-hosted compatible
  servers. The console offers *Finalize on holdout* and *Send promoted to Alpha
  Lab*, shows finals beside each idea, and lists the OpenAI-compatible provider.
- **Evidence.** The console's new Evidence view and `tubby-qlib workflow`
  commands record every submitted Lab, Qlib and portfolio attempt in a sealed,
  append-only store under the data root; a saved Lab result can be previewed and
  validated in signed Qlib, one or two validated signals can share one account
  under explicit risk limits, and chronological protocols reserve their final
  window before it is scored. See [Evidence](EVIDENCE.md).
- **Alpha Lab engine.** Overflow becomes missing instead of infinite, an empty
  side of `scale` no longer poisons a day, arbitrary series used as groups no
  longer allocate by label value, `group_zscore` uses a two-pass variance,
  `group_rank` is one lexsort, `ts_arg_max/min` work in bounded blocks, `clamp`
  rejects an empty interval and wrong-typed options raise positioned expression
  errors. New reference tests pin these and previously untested operators.
- **Compatibility.** The offline suite runs on macOS through `scripts/run_tests.py`
  (canonical temporary root); the signed holding-series fast path accepts
  pandas 3's default microsecond dates and string dtype.
- **Documentation.** README repositioned around the offline BRAIN-style
  simulator, auditable validation, the research loop and agent access.

### Verification scope for v0.2.0

macOS 15 (arm64) with CPython 3.12. The private source tree's full offline suite
passed in three environments, pinned numpy 1.26.4 / pandas 2.2.3, numpy 2.3 /
pandas 3.1 with PyArrow, and a numpy-free interpreter (2,245 tests each,
synthetic fixtures only). This exported tree was then tested in a disposable
clone: a clean `pip install -e .` environment (numpy 2.5 / pandas 2.3), the
pinned environment and the pandas 3 environment each ran its offline suite
(1,971 tests each, no failures; 522 skipped without the optional runtimes,
380 with PyArrow); the four
commands, the synthetic research demo, the MCP handshake and the token-guarded
console server with its Evidence listing were exercised. In the private tree the
console's finalize path and the hand-off to Alpha Lab were also exercised in a
browser. Hosted checks on the merged private revisions covered the bounded Linux
core suite and Windows import/installer checks. The signed runtime remains
verified on Windows x64 only; macOS remains experimental and Linux has no pinned
signed runtime. Nothing here establishes trading readiness.

## Initial research software preview

Tubby Quant is a local workbench for daily equity alpha research. This preview
includes Alpha Lab screening and batches, Qlib validation with separate long-only
and verified signed modes, saved runs and HTML reports, selected SEC fundamental
definitions with shared `sec_` names, and price/volume Research campaigns. The
synthetic Research demo runs without provider credentials or model calls.
Research campaigns do not integrate SEC or dated classification inputs.

Project-owned framework code uses Apache License 2.0. Third-party components
retain their licenses and notices. Data, credentials, portfolios and generated
research records belong outside the source checkout. Installation supplies no
market dataset. See [installation](INSTALL.md), [data](DATA.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md).

### Verification scope

Windows x64 with CPython 3.12 has isolated installation/recovery evidence,
local HTTP console checks and three small synthetic signed simulations. The
simulations used six instruments over 24 sessions, checked single/batch ledger
agreement and fees, and generated reports. Installer regressions cover UTF-8
subprocess output and launch commands that retain the selected data folder.
The installation exercise included recovery after an interruption; it does not
certify an uninterrupted fresh-machine install or every native desktop window.

Offline suites exercise synthetic fixtures and bounded components. The included
workflow selects core Linux tests with NumPy/pandas and Windows import/installer
checks. Its presence is not evidence of a successful hosted CI run. Skipped
optional or platform checks do not establish acceptance of those capabilities.

macOS is experimental, with a separate pinned runtime and no real-machine
installation or desktop acceptance for this preview. Linux has core checks only,
no pinned signed runtime and no verified desktop installation path. Windows
helper compatibility is limited to CPython 3.12 on Windows x64.

### Research and operational limits

The framework places no trades and provides no broker connection. Lab's weight
simulation and Qlib's cash/share ledger use different timing and accounting;
matching inputs do not imply equal returns. Signed simulation assumes stocks
are borrowable, without recalls or broker margin enforcement. Market impact,
liquidity, funding and historical data limitations remain. SEC availability is
inferred, with missing coverage and source-dependent definitions.

Synthetic returns and historical backtests establish neither investment merit
nor readiness to trade. Interfaces and saved formats may change. This is a local
research preview, not a supported hosted or multi-user service. Read the full
[limitations](LIMITATIONS.md) and [security guidance](../SECURITY.md) before use.

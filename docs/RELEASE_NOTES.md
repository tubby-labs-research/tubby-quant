# Release notes

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

# Agent instructions

Start with [README.md](README.md), [docs/README.md](docs/README.md) and
[CONTRIBUTING.md](CONTRIBUTING.md). Follow the user's current scope and coordinate
file ownership before editing a shared checkout.

- Alpha Lab is the primary fast screening tool. Qlib provides detailed validation.
  Research campaigns are separate and currently use daily price/volume inputs.
- Read the relevant public guide and actual source before changing behavior.
  Preserve expression syntax, timing, units, missing-data rules and run provenance.
- New SEC formulas use verified dictionary `sec_` names. Keep saved legacy
  bindings and their original meanings; do not silently migrate them.
- Keep data, strategies, campaigns, results, credentials and machine settings
  outside source control. Use synthetic fixtures for reproducible examples.
- Preserve unrelated edits and historical outputs. Do not overwrite source data
  or bypass runtime/data hashes to make a test pass.
- Run relevant offline checks; report failures and skips. A unit suite does not
  establish a clean installation, platform support or trading readiness.
- This file grants no standing permission to publish, push, merge, deploy,
  connect accounts, incur model costs, download market data or place trades.
  Obtain task-specific authorization for those actions.

Report changed files, checks and unresolved limits. Publishing a revision,
changing repository settings and declaring a release require separate authority;
local changes or passing tests are not permission to do so.

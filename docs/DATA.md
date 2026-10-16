# Data and local storage

Bring your own data with rights suitable for your use. No market database,
private strategy, campaign result or credential is bundled. Setup downloads
software dependencies and pinned source, not a research universe.

## Use synthetic data first

Choose **Research → Try demo** after [installation](INSTALL.md). It uses artificial
OHLCV and deterministic responses while exercising the real Alpha Lab evaluator.
It makes no model calls or market-data downloads and proves no investment merit.
For a separate command-line demo, from the extracted folder on Windows:

```powershell
.\env\qlib\Scripts\python.exe -m autoresearch demo --root (Join-Path $taskDataRoot 'autoresearch-demo')
.\env\qlib\Scripts\python.exe -m autoresearch report --research-root (Join-Path $taskDataRoot 'autoresearch-demo/research')
```

On macOS:

```bash
env/qlib/bin/python -m autoresearch demo --root "$task_data_root/autoresearch-demo"
env/qlib/bin/python -m autoresearch report --research-root "$task_data_root/autoresearch-demo/research"
```

These demo commands choose their own external fixture record folder. They are
separate from ordinary campaigns' `<data root>/autoresearch/records` default.
Repeated demo invocations resume completed work. `tests/fixtures/` supplies
additional authored accounting/data cases; fixtures are not provider datasets.

## Prepare your inputs

Use **Manage data** / **Data** to inspect the local catalog and available import
actions. Existing Yahoo, Stooq, Sharadar and HistoricalData adapters have distinct
schemas and provenance. Select explicit source paths and a separate writable
data root. Read command help before an import; fetch/prepare commands can access
external providers. Credentials and provider terms remain your responsibility.

A prepared snapshot carries a manifest, price calendar, symbols, field files
and, where supported, dated universe membership. An arbitrary CSV directory is
not a verified snapshot. Named TOP-N universes depend on the snapshot's ranking;
historical-index membership cannot be treated as a ranked subset. Valid syntax
does not prove enough warm-up history, good coverage or correct adjustments.

SEC catalogs, dictionaries and run indexes are separate prepared inputs; see
the [SEC guide](SEC_FIELDS_USER_GUIDE.md). Do not turn missing observations into
zeros, infer unsupported units or assume text metadata is a numeric feature.

## Retain provenance

Store datasets, runs, Alpha Lab libraries, classification sources, SEC preparation
and Research state below your external data root. Keep original source files
separate and preserve recorded hashes, settings and availability conventions.
Back up valuable data/results independently; ignored files are not backed up by
Git. Never distribute user datasets, private campaigns or local settings with a
framework source distribution. See [limitations](LIMITATIONS.md) for historical
identity, survivorship and accounting risks.

<div align="center">

# Tubby Quant

**An offline, auditable workbench for daily equity alpha research.**

BRAIN-style Alpha Lab screening · Qlib-backed long/short validation · a source-grounded research loop with a sealed holdout · an MCP server for agents

[![Framework checks](https://github.com/tubby-labs-research/tubby-quant/actions/workflows/tests.yml/badge.svg)](https://github.com/tubby-labs-research/tubby-quant/actions/workflows/tests.yml) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

[Get started](#get-started) · [User guide](docs/USER_GUIDE.md) · [Documentation](docs/README.md) · [Limitations](docs/LIMITATIONS.md)

</div>

Tubby Quant runs WorldQuant-BRAIN-style FASTEXPR alphas on **your own daily
data, on your own machine**: the same operators, neutralization, truncation,
decay and metric conventions you know from BRAIN, with every input hashed,
every universe rebuilt from what was known at the time, and every result saved
with the settings that produced it. It does not place trades.

> **Real console, synthetic data.** Every screenshot shows the interface on
> purpose-made artificial data, not investment performance.

![Tubby Quant console using purpose-made synthetic toy data; not real-market investment evidence.](https://raw.githubusercontent.com/tubby-labs-research/tubby-quant/main/docs/images/console.jpg)

## Why it exists

- **BRAIN semantics, offline.** Screen ideas without an account, a queue or
  rate limits, on data you are allowed to use, with a simulator whose
  conventions are documented operator by operator.
- **Validation you can defend.** Point-in-time TOP-N universes, survivorship-free
  data paths (Sharadar, HistoricalData), a Qlib derivative that fills at the next
  open with whole shares, borrow costs and restricted short proceeds, hash-pinned
  runtimes and datasets, and reports rebuilt from saved files.
- **Research that keeps its evidence.** Campaigns propose hypotheses only from
  retrieved sources with exact quotes, evaluate them chronologically, refine or
  reject them under explicit gates, score the reserved holdout **once**, and
  keep every record immutable. Agents get bounded tools, never shell or trading
  authority.

## What is in the box

| Tool | Use it for |
| --- | --- |
| **[Alpha Lab](docs/USER_GUIDE.md#screen-in-alpha-lab)** | Fast FASTEXPR screening, batches with a leaderboard and in-batch correlations, saved alpha library. |
| **[Qlib Validation](docs/USER_GUIDE.md#validate-with-qlib)** | Detailed long-only or verified long/short simulation with an account ledger, fees and borrow; saved runs and HTML reports. |
| **[Research](docs/USER_GUIDE.md#research-campaigns)** | Source-grounded hypothesis campaigns (Anthropic, OpenAI-compatible or your own agent), chronological evaluation, sealed holdout finals, hand-off to Lab and Qlib. |
| **[MCP server](docs/USER_GUIDE.md#drive-alpha-lab-from-an-agent-mcp)** | Let Claude Code, Codex or any MCP client check, simulate and compare alphas and read campaign reports inside one data root. |
| **[SEC fields](docs/SEC_FIELDS_USER_GUIDE.md)** | Discoverable fundamental definitions with point-in-time availability for Lab and Validation. |
| **[Data adapters](docs/DATA.md)** | Yahoo, Stooq, Sharadar and HistoricalData imports into hashed Qlib snapshots with dated TOP-N universes. |

Lab and Qlib use different expression syntax, timing and accounting on purpose;
matching inputs do not imply matching returns. Research campaigns currently use
daily price/volume inputs only.

## Alpha Lab results

**Alpha Lab: positive outcome · SYNTHETIC TOY DATA**

![Positive Alpha Lab result in the real console using synthetic toy data; not investment performance.](https://raw.githubusercontent.com/tubby-labs-research/tubby-quant/main/docs/images/positive-result.jpg)

**Alpha Lab: negative outcome · SYNTHETIC TOY DATA**

![Negative Alpha Lab result in the real console using synthetic toy data; not investment performance.](https://raw.githubusercontent.com/tubby-labs-research/tubby-quant/main/docs/images/negative-result.jpg)

Both runs use the same artificial dataset and settings with the signal's sign
reversed: 200 synthetic instruments built from eight deterministic price paths
(seed 7403), TOP200, delay 1, decay 4, MARKET neutralization, 1% cap, 5 bps cost,
evaluated over 2024. The console's own charts, captured on Windows; the inputs
and capture settings are recorded in [screenshot provenance](https://github.com/tubby-labs-research/tubby-quant/blob/main/docs/SCREENSHOTS.md).

## Get started

Install Git and **64-bit CPython 3.12**, then:

```bash
git clone https://github.com/tubby-labs-research/tubby-quant.git
cd tubby-quant
python3.12 -m venv env/core
env/core/bin/python -m pip install -e .
env/core/bin/tubby-quant --browser --data-root ~/tubby-data
```

Open **Research → Try demo** first: it runs the real engine on synthetic data
with deterministic responses and no model calls or credentials. Then bring
your own licensed data ([data guide](docs/DATA.md)) and keep datasets,
campaigns, credentials and reports outside the source folder.

The package install covers Alpha Lab, Research, the console and the data
adapters on Windows, macOS and Linux. The verified **long/short Qlib runtime**
uses the platform setup scripts (Windows x64 has installation and synthetic
signed-run evidence; macOS is experimental; Linux has no pinned signed runtime).
See [installation and recovery](docs/INSTALL.md). On Windows use
`env\core\Scripts\python.exe`.

### Use it from an agent

```bash
claude mcp add tubby-alpha-lab -- tubby-alpha-mcp --data-root ~/tubby-data
```

The server exposes `list_operators`, `check_expression`, `list_datasets`,
`simulate_alpha`, `simulate_batch`, `library_list` and a read-only
`research_report`, all contained in the data root. Research campaigns can also
call an external agent through the [Research Agent API](docs/RESEARCH_AGENT_API.md).

### Run a research campaign from the command line

```bash
tubby-research run campaign.json --once --allow-model     # discover, propose, evaluate, review
tubby-research report --markdown                           # one row per hypothesis
tubby-research finalize campaign.json                      # promoted ideas, scored once on the holdout
tubby-research export campaign.json --output promoted.txt  # hand promoted alphas to Lab and Qlib
```

## How it compares

| | Tubby Quant | WorldQuant BRAIN | Microsoft Qlib + RD-Agent | LLM alpha-mining frameworks | General backtesters (LEAN, vectorbt, Zipline) |
| --- | --- | --- | --- | --- | --- |
| Runs where | your machine, your data | hosted platform, platform data | your machine | your machine | your machine |
| Alpha language | BRAIN-style FASTEXPR, documented operator by operator | FASTEXPR | Qlib expressions, ML models | code or expressions per framework | strategy code |
| Fast screen | fixed-book simulator with BRAIN metric conventions | platform simulator | factor analysis | varies | n/a |
| Detailed validation | signed Qlib derivative: next-open fills, whole shares, fees, borrow | n/a | long-only Top-k by default | varies | event or vectorized execution |
| Research loop | source-grounded proposals, chronological gates, sealed single-use holdout, immutable records | consultant workflow | LLM factor and model R&D | LLM or evolutionary mining, typically reported in-sample | n/a |
| Agent access | MCP server and HTTP agent contract with bounded authority | n/a | agent framework | agent frameworks | n/a |

Scope as of October 2026; each project's own documentation is authoritative
for its current features. Tubby Quant builds on Qlib rather than replacing it.

## Know the assumptions

Lab defaults to zero turnover cost and charges no stock borrow; the screenshots
use 5 bps. Qlib's signed simulation assumes every selected stock is borrowable
at a fixed rate with no recalls or margin calls. Historical data quality,
delistings and adjustment conventions depend on the source. Read the
[limitations](docs/LIMITATIONS.md) before interpreting any statistic.

## Status

Early research software preview: interfaces and saved formats may change.
The offline suite (about 2,200 tests on synthetic fixtures) runs on Windows and
macOS through `scripts/run_tests.py`; CI covers bounded Linux and Windows
checks. Nothing here certifies trading readiness. [Release notes](docs/RELEASE_NOTES.md)
record what each preview verified; [Architecture](docs/ARCHITECTURE.md) describes
the components and storage.

## Contribute and license

Focused fixes and reproducible synthetic examples are welcome. Read
[Contributing](CONTRIBUTING.md) for offline checks and sign-off,
[AGENTS.md](AGENTS.md) for agent guidance and [Security](SECURITY.md) for
reporting vulnerabilities. Keep private data, strategies and credentials out
of issues and contributions.

Project-owned code is licensed under **[Apache License 2.0](LICENSE)**.
Retained third-party code keeps its own licenses; see [NOTICE](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md).

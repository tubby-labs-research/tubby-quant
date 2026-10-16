# Tubby Quant

A local workbench for daily equity alpha research. **Alpha Lab** is the fast
screening tool; **Qlib Validation** provides detailed simulation and accounting.
**Research** manages separate, source-backed price/volume campaigns.
The framework does not place trades.

This is an **early research software preview**, with a pinned Windows native
helper. Interfaces and file formats may change. Project-owned code is
licensed under Apache License 2.0; retained third-party code keeps its own
licenses. See [LICENSE](LICENSE), [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Simulations and synthetic examples
do not establish investment performance or readiness to trade.

## Start locally

Clone the public source or extract a complete source release:

```bash
git clone https://github.com/tubby-labs-research/tubby-quant.git
cd tubby-quant
```

Follow [installation](docs/INSTALL.md) with CPython 3.12. Windows x64 has bounded
installation and synthetic simulation evidence; macOS is experimental and awaits
real-machine installation and desktop verification. Linux covers core checks
only and has no pinned signed runtime. The Windows native helper is bundled
and hash-checked; setup does not require GitHub CLI or private repository access.
Package and pinned public Qlib source downloads still need network access.

After setup, from the extracted folder:

```powershell
.\env\qlib\Scripts\python.exe -m qlib_yahoo.webapp --browser --data-root $taskDataRoot
```

Here `$taskDataRoot` is the external storage folder chosen in the installation
guide. On macOS use `env/qlib/bin/python` and `--data-root "$task_data_root"`.
Open **Research → Try demo** first: it uses artificial data without model calls.
Setup supplies no market dataset. Bring your own licensed data for other runs.

## Guides

- [User guide](docs/USER_GUIDE.md): screening, validation, Research and saved results.
- [SEC fields](docs/SEC_FIELDS_USER_GUIDE.md): discover exact fundamental definitions.
- [Data](docs/DATA.md) and [limitations](docs/LIMITATIONS.md): inputs and assumptions.
- [Release notes](docs/RELEASE_NOTES.md): preview features and verification scope.
- [Architecture](docs/ARCHITECTURE.md) and [Research Agent API](docs/RESEARCH_AGENT_API.md).
- [Contributing](CONTRIBUTING.md), [agent instructions](AGENTS.md) and [security](SECURITY.md).

`qlib_yahoo/` contains the workbench, adapters and simulators; `autoresearch/`
contains the campaign controller. `scripts/` holds setup utilities and `tests/`
contains offline fixtures. Data, campaigns, credentials and generated reports
belong outside the source folder. See the [documentation index](docs/README.md).

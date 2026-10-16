# Architecture

The workbench combines three distinct workflows over prepared local inputs:

| Component | Responsibility |
| --- | --- |
| `qlib_yahoo/webapp/` | Loopback UI/service, job control and saved-report access. |
| `qlib_yahoo/alphalab/` | FASTEXPR screening, fixed-book simulation and local library. |
| Qlib runner and `qlib_yahoo/runtime.py` | Native expression validation, account simulation and verified signed workers. |
| `autoresearch/` | Source discovery, cited proposals, chronological evaluations, decisions and immutable records. |
| SEC adapters | Catalog discovery, fixed dictionary definitions and selected numeric materialization. |
| `scripts/` | Platform setup and verified runtime installation. |

Prices, numeric fundamentals and group classifications have separate identities.
Only requested SEC recipes become daily inputs. Research reuses Alpha Lab's
price/volume evaluator; it does not automatically perform a Qlib accounting pass.

Keep source and writable user state separate. Datasets live under the external
data root's `datasets/`, Qlib outputs under `runs/`, Lab state under `alphalab/`,
and SEC/classification preparation under their respective folders. New Research
records default to `autoresearch/records`, with configurations and runtime state
alongside. Explicit CLI paths remain supported. Legacy source records are
inspectable without automatic migration. One campaign writer owns its runtime;
file sharing is not distributed locking.

Signed workers verify pinned wheel/build identity and installed files before
activation. Windows x64 uses the CPython 3.12 engine plus its bundled hash-pinned
native date helper; macOS uses its separate universal2 engine and portable paths.
The bundle eliminates authenticated helper retrieval, not public package/source
downloads. Normal Qlib and signed packages remain separate; verification failures
must not silently select a different mode.

The UI binds to loopback with launch-token and request guards. External Research
agents are administrator-configured HTTP services with bounded proposal/refinement
authority; they receive evidence and history, not shell/trading permissions.
See [API](RESEARCH_AGENT_API.md), [security](../SECURITY.md) and
[limitations](LIMITATIONS.md). Architecture describes implemented boundaries,
not complete deployment or release acceptance.

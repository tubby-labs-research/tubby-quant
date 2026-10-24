# Install and launch

Install Git and **64-bit CPython 3.12** first. After publication, clone the public
source, or extract a complete source release, and open a terminal in its folder:

```bash
git clone https://github.com/tubby-labs-research/tubby-quant.git
cd tubby-quant
```

The repository URL above is the intended public location; these commands apply
once it has been published.
Setup needs network access for pinned packages and public Qlib source; it does
not download a market universe. Keep writable data outside the extracted folder.
The first install includes Qlib's numerical and notebook dependencies and can
take several minutes. Package progress remains visible. Setup also checks package
metadata and declared files: an interrupted installation can look complete to
the package manager while files are missing. If this check fails, rerun with
`-Recreate` / `--recreate`; it preserves the old environment by moving it aside.

## Windows x64

Windows installation, local HTTP startup and small synthetic signed simulations
have been exercised in an isolated environment. That evidence includes recovery
from an interrupted dependency install; it is not an uninterrupted fresh-machine
or universal desktop acceptance test.

From PowerShell:

```powershell
$taskDataRoot = Join-Path $env:LOCALAPPDATA 'TubbyLabs\FrameworkData'
powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1 -Signed -NoShortcuts -DataRoot $taskDataRoot
.\env\qlib\Scripts\python.exe -m qlib_yahoo.webapp --browser --data-root $taskDataRoot
```

Setup creates `env/qlib`, checks dependencies, prepares the external storage
folder and installs/verifies the pinned signed runtime. `-NoShortcuts` avoids
Desktop and Start menu changes. Omit it when you want shortcuts. `-Python` can
select a particular Python 3.12 executable. The desktop window needs Edge WebView2;
`--browser` uses your browser instead.

Choose the storage folder explicitly with `-DataRoot` as above. Without it,
setup uses an inherited `TUBBY_QLIB_DATA_ROOT`, otherwise `D:\Qlib`, falling back
to `%USERPROFILE%\Qlib` when the default drive is unavailable. Relative roots
are resolved against the extracted checkout. Setup's environment change lasts
only in its own process; it does not save a user or system setting.

At completion, save the printed PowerShell launch command. It includes the
absolute checkout and interpreter paths and the selected `--data-root`, so it
also works from a new terminal in another folder. Add `--browser` to that command
if needed. When created, shortcuts also retain the selected root. A bare
`launch-qlib-console.cmd` does not remember setup's selection: pass
`--data-root 'C:\your\data folder'` or set `TUBBY_QLIB_DATA_ROOT` in each launch
terminal. With neither, the launcher defaults to `D:\Qlib`, even if setup used
the no-D-drive fallback.

The distribution bundles its hash-pinned Windows native helper under
`runtime-artifacts/`. GitHub CLI, authentication and private repository access
are not required. Keep the bundle with the source: a missing/altered archive
fails verification. Other runtime dependencies still download from public sources.

## macOS

Experimental: the setup path and separate runtime pin exist, but installation
and the desktop app have not been verified on a real macOS machine for this preview.

From Terminal:

```bash
task_data_root="$HOME/Library/Application Support/TubbyLabs/FrameworkData"
bash scripts/macos/setup.sh --signed --no-app --data-root "$task_data_root"
export TUBBY_QLIB_DATA_ROOT="$task_data_root"
env/qlib/bin/python -m qlib_yahoo.webapp --browser --data-root "$task_data_root"
```

Use `--python /path/to/python3.12` if detection needs help. Omit `--no-app` to
build the desktop app. macOS selects its own pinned universal2 engine, using
Python date checks and standard-library gzip rather than the Windows helper.
The macOS and Windows runtime pins are different and are not interchangeable.

## Linux

Linux is limited to core offline development checks with Python 3.12 and the
workflow's minimal NumPy/pandas dependencies. There is no pinned Linux signed
runtime or verified desktop/setup path. See [contributing](../CONTRIBUTING.md)
and the [workflow](../.github/workflows/tests.yml) for the bounded checks.

## Install as a Python package (any platform)

Alpha Lab screening, Research campaigns, the loopback console and the data
adapters install as an ordinary package from the source folder:

```bash
python3.12 -m venv env/core
env/core/bin/python -m pip install -e .
env/core/bin/tubby-quant --browser --data-root ~/tubby-data
```

This provides the `tubby-quant` (console), `tubby-qlib` (datasets and Qlib
backtests) and `tubby-research` (campaign worker) commands; the `python -m`
forms keep working. Extras mirror the pinned requirement files: `.[qlib]` for
the Qlib validation stack, `.[table]` for Parquet table imports, `.[desktop]`
for the native window and `.[research]` for the direct Anthropic adapter. The
package path installs no signed Qlib derivative and performs none of the
runtime verification: signed validation still uses the platform setup above.
On Windows use `env\core\Scripts\python.exe`.

## Dependencies and downloads

Windows setup reads `requirements-qlib-windows.txt`; macOS setup reads
`requirements-qlib-macos.txt`. Both include `requirements-qlib-yahoo.txt` and
`requirements-qlib-table.txt`. The declared stack pins NumPy 1.26.4, pandas 2.2.3,
SciPy 1.14.1, MLflow 3.16.1, pyqlib 0.9.7, yahooquery 2.4.1, Plotly 5.24.1,
statsmodels 0.14.4, PyArrow 25.0.1 and pywebview 6.2.1. These files are not a full
transitive lock; setup also upgrades pip, and transitive packages can change.

With the default public package index, installation uses PyPI/Python package
hosting, the [upstream Qlib source](https://github.com/microsoft/qlib) and the
[pinned Qlib fork](https://github.com/tubby-labs-research/qlib). Signed setup
downloads the official base wheel and exact fork source, builds and verifies
the derivative wheel, and verifies its installed files. Windows additionally
downloads hash-pinned compiler/compression wheels from Python package hosting;
its native helper is already bundled. No private account or repository is
required. Local pip/Git configuration or a proxy can change network routing.
Setup does not request provider credentials or download market data.

The signed installer explicitly matches UTF-8 child output and decoding,
including Unicode checkout paths under legacy Windows pipe encodings. Keep
that installer fix when updating; it does not certify every dependency's path
handling. Hash or package-integrity failures require investigation, not bypasses.

## First check and recovery

In **Research**, choose **Try demo**. It uses synthetic data and deterministic
responses without provider credentials or paid calls. Continue with [data](DATA.md)
and the [user guide](USER_GUIDE.md) before importing real data.

Re-running setup checks the existing environment. `-Recreate` / `--recreate`
moves the environment aside rather than deleting it. This does not back up data;
back up valuable datasets and runs separately. On Windows, reuse the saved launch
command or created shortcut; for other commands, pass the same external data root
or set it in that terminal. Setup's environment variable changes are process-local.

For runtime errors, first check Python 3.12, the selected platform and complete
source bundle. Re-run setup with the original data root. Do not edit hashes
or substitute an arbitrary Qlib fork head. Signed mode is explicit and fails
when its verified runtime is unavailable; the CLI defaults to long-only.

The pinned [Qlib fork](https://github.com/tubby-labs-research/qlib) and exact
[Windows](https://github.com/tubby-labs-research/qlib/commit/8ab6249aa6143cb0675259d5ff986d294e918ef4)
and [macOS](https://github.com/tubby-labs-research/qlib/commit/d10a5a8ca397ea088da8f5c996e8bd042eea27e0)
source commits were reachable anonymously during preview preparation. Source
access alone does not establish successful installation. Read the
[release notes](RELEASE_NOTES.md) and [limitations](LIMITATIONS.md) for verification scope.

#!/bin/bash
# One-time macOS setup for the Qlib console: Python 3.12 environment, pinned Qlib
# source for dataset conversion, optional signed engine, and the desktop app.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/macos/setup.sh [--signed] [--no-app] [--python PATH] [--data-root DIR] [--recreate]
  --signed        Also build and verify the pinned signed (long/short) Qlib engine
  --no-app        Do not build "Tubby Qlib Console.app" in ~/Applications
  --python PATH   Python 3.12 to use (default: auto-detect)
  --data-root DIR Data folder (default: ~/Qlib)
  --recreate      Move an existing env/qlib aside and create a fresh one
USAGE
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIGNED=0; APP=1; PYTHON_CHOICE=""; DATA_ROOT="${TUBBY_QLIB_DATA_ROOT:-$HOME/Qlib}"; RECREATE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --signed) SIGNED=1; shift ;;
    --no-app) APP=0; shift ;;
    --python) PYTHON_CHOICE="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --recreate) RECREATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ "$(uname -s)" = "Darwin" ] || { echo "This setup script is for macOS." >&2; exit 1; }
case "$DATA_ROOT" in /*) ;; *) DATA_ROOT="$PWD/$DATA_ROOT" ;; esac  # relative to where it was typed
cd "$ROOT"  # later steps import qlib_yahoo from this checkout, whatever folder the script was started from
step() { printf '\n==> %s\n' "$1"; }

usable_python() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
assert sys.version_info[:2] == (3, 12)
PY
}

step "Finding Python 3.12"
PYTHON=""
for candidate in "$PYTHON_CHOICE" /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 "$(command -v python3.12 || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ] && usable_python "$candidate"; then
    PYTHON="$candidate"; break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "Python 3.12 is required. Install with:  brew install python@3.12" >&2
  echo "or use the python.org 3.12 macOS installer, then re-run this script." >&2
  exit 1
fi
echo "Using $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])'))"

ENV="$ROOT/env/qlib"
if [ -d "$ENV" ] && [ "$RECREATE" = 1 ]; then
  ASIDE="$ENV.old-$(date +%Y%m%d%H%M%S)"
  mv "$ENV" "$ASIDE"
  echo "Moved the previous environment to $ASIDE"
fi
if [ -x "$ENV/bin/python" ] && ! usable_python "$ENV/bin/python"; then
  echo "env/qlib exists but lacks Python 3.12; re-run with --recreate." >&2
  exit 1
fi
if [ ! -x "$ENV/bin/python" ]; then
  step "Creating env/qlib"
  "$PYTHON" -m venv "$ENV"
fi
step "Installing pinned packages (Qlib 0.9.7, pandas, plotly, pyarrow, pywebview)"
"$ENV/bin/python" -m pip install --quiet --upgrade pip
"$ENV/bin/python" -m pip install -r "$ROOT/requirements-qlib-macos.txt"
"$ENV/bin/python" -m pip check
"$ENV/bin/python" "$ROOT/scripts/verify_framework_environment.py"
"$ENV/bin/python" -c "import qlib, pandas, pyarrow, plotly, webview; print('qlib', qlib.__version__, '· pywebview ok')"

step "Preparing the data folder and pinned Qlib source ($DATA_ROOT)"
mkdir -p "$DATA_ROOT"
TUBBY_QLIB_DATA_ROOT="$DATA_ROOT" "$ENV/bin/python" - <<'PY'
from pathlib import Path
from qlib_yahoo import DEFAULT_DATA_ROOT
from qlib_yahoo.dataset import ensure_upstream
print("dump_bin:", ensure_upstream(Path(DEFAULT_DATA_ROOT)))
PY

if [ "$SIGNED" = 1 ]; then
  step "Installing and verifying the accepted signed runtime"
  "$ENV/bin/python" "$ROOT/scripts/install_fast_runtime.py" --data-root "$DATA_ROOT"
fi

if [ "$APP" = 1 ]; then
  step "Building the desktop app"
  if [ "$DATA_ROOT" = "$HOME/Qlib" ]; then
    bash "$ROOT/scripts/macos/build_app.sh"
  else
    bash "$ROOT/scripts/macos/build_app.sh" --data-root "$DATA_ROOT"
  fi
fi

step "Done"
echo "Launch: the Tubby Qlib Console app, launch-qlib-console.command, or"
echo "        env/qlib/bin/python -m qlib_yahoo.webapp"
echo "Import the Stooq table in the app's Data tab, or from Terminal:"
echo "  env/qlib/bin/python -m qlib_yahoo import-stooq-table --table <us_daily.parquet> --start 2010-01-01 --end <today> --make-latest"

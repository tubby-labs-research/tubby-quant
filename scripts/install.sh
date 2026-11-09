#!/bin/bash
# Tubby Quant installer for macOS: get the source, build the pinned Python environment, prepare the data
# folder and the desktop app. Idempotent: run it again to update an existing installation.
#
#   curl -fsSL https://raw.githubusercontent.com/tubby-labs-research/tubby-quant/main/scripts/install.sh | bash
#   bash scripts/install.sh --dir ~/TubbyQuant --data-root ~/Qlib --ref v0.2.0 --signed
#
# Everything stays in two folders you choose: the source checkout (--dir) and the data folder (--data-root).
# Nothing is written elsewhere except the app bundle in ~/Applications and a launcher log under ~/Library/Logs.
# The tool places no trades and needs no account; the installer downloads only the source and pinned packages.
set -euo pipefail

REPO_URL="${TUBBY_REPO_URL:-https://github.com/tubby-labs-research/tubby-quant.git}"
DIR="$HOME/TubbyQuant"
DATA_ROOT="${TUBBY_QLIB_DATA_ROOT:-$HOME/Qlib}"
REF=""
SIGNED=0
APP=1
SOURCE=""
usage() {
  cat <<'USAGE'
Usage: install.sh [--dir DIR] [--data-root DIR] [--ref TAG|BRANCH] [--signed] [--no-app] [--source PATH|URL]
  --dir DIR         Where the source checkout lives (default: ~/TubbyQuant)
  --data-root DIR   Data folder for prices, runs and campaigns (default: ~/Qlib)
  --ref REF         Release tag or branch to install (default: the repository's default branch)
  --signed          Also install and verify the pinned long/short Qlib engine
  --no-app          Do not build "Tubby Qlib Console.app" in ~/Applications
  --source PATH|URL Clone from another location (a local checkout for offline installs, or a fork)
USAGE
}
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --signed) SIGNED=1; shift ;;
    --no-app) APP=0; shift ;;
    --source) SOURCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ "$(uname -s)" = "Darwin" ] || { echo "This installer is for macOS. Windows: scripts\\windows\\setup.ps1; see docs/INSTALL.md." >&2; exit 1; }
case "$DIR" in /*|~*) ;; *) DIR="$PWD/$DIR" ;; esac
case "$DATA_ROOT" in /*|~*) ;; *) DATA_ROOT="$PWD/$DATA_ROOT" ;; esac
DIR="${DIR/#\~/$HOME}"; DATA_ROOT="${DATA_ROOT/#\~/$HOME}"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "$2" >&2; exit 1; }; }

step "Checking prerequisites"
need git "Git is required. Install Xcode's command line tools:  xcode-select --install"
PY=""
for candidate in /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 "$(command -v python3.12 || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.12 is required (64-bit CPython). Install it with:  brew install python@3.12" >&2
  echo "or the python.org 3.12 macOS installer, then run this installer again." >&2
  exit 1
fi
echo "git $(git --version | awk '{print $3}') · $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

step "Getting the source into $DIR"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch --quiet --tags origin
  if [ -n "$REF" ]; then git -C "$DIR" checkout --quiet "$REF"; fi
  if git -C "$DIR" symbolic-ref -q HEAD >/dev/null; then git -C "$DIR" pull --quiet --ff-only || echo "Local changes kept; not fast-forwarded."; fi
elif [ -e "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ]; then
  echo "$DIR exists and is not a Git checkout; choose another --dir or remove it." >&2; exit 1
else
  git clone --quiet "${SOURCE:-$REPO_URL}" "$DIR"
  if [ -n "$REF" ]; then git -C "$DIR" checkout --quiet "$REF"; fi
fi
echo "Source: $(git -C "$DIR" rev-parse --short HEAD) ($(git -C "$DIR" describe --tags --always 2>/dev/null))"

step "Building the environment, data folder and app (several minutes the first time)"
ARGS=(--python "$PY" --data-root "$DATA_ROOT")
[ "$SIGNED" = 1 ] && ARGS+=(--signed)
[ "$APP" = 0 ] && ARGS+=(--no-app)
bash "$DIR/scripts/macos/setup.sh" "${ARGS[@]}"

step "Checking the installation"
# The console runs from the checkout (the app and launcher cd there first); the package is not installed into the environment.
(cd "$DIR" && env/qlib/bin/python -m qlib_yahoo.webapp --help >/dev/null)
(cd "$DIR" && env/qlib/bin/python -m autoresearch.lab.cli domains >/dev/null)
echo "Console and research lab import and start correctly."

step "Installed"
cat <<DONE
Source:  $DIR
Data:    $DATA_ROOT
Launch:  open "$HOME/Applications/Tubby Qlib Console.app"      (if built)
         cd "$DIR" && env/qlib/bin/python -m qlib_yahoo.webapp --data-root "$DATA_ROOT"
First session: the console opens on Start; create demonstration data, simulate three alphas, run a campaign.
Guide: $DIR/docs/USER_GUIDE.md (public release) or docs/QLIB_CONSOLE.md
DONE

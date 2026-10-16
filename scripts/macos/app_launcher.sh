#!/bin/bash
# Executable inside "Tubby Qlib Console.app": start the console from the checkout recorded at build time.
set -u
RESOURCES="$(cd "$(dirname "$0")/../Resources" && pwd)"
REPO="$(cat "$RESOURCES/repo-path" 2>/dev/null || true)"
LOG_DIR="$HOME/Library/Logs/Tubby Qlib Console"
LOG="$LOG_DIR/launcher.log"
mkdir -p "$LOG_DIR"

alert() {
  /usr/bin/osascript - "$1" >/dev/null 2>&1 <<'OSA'
on run argv
  display alert "Tubby Qlib Console" message (item 1 of argv) as critical
end run
OSA
}

if [ -z "$REPO" ] || [ ! -f "$REPO/qlib_yahoo/webapp/__main__.py" ]; then
  alert "The Quant checkout recorded in this app was not found:
$REPO

Run scripts/macos/build_app.sh again from your checkout."
  exit 1
fi
PYTHON="$REPO/env/qlib/bin/python"
if [ ! -x "$PYTHON" ]; then
  alert "The Qlib Python environment is missing. In Terminal run:

cd \"$REPO\" && scripts/macos/setup.sh"
  exit 1
fi
if [ -f "$RESOURCES/data-root" ]; then
  export TUBBY_QLIB_DATA_ROOT="$(cat "$RESOURCES/data-root")"
fi
# Finder starts apps with a minimal PATH; git (pinned-source checks) may live in Homebrew.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONIOENCODING=utf-8
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG")" -gt 2000000 ]; then
  mv -f "$LOG" "$LOG.1"
fi
cd "$REPO" || exit 1
echo "=== $(date '+%Y-%m-%d %H:%M:%S') launching from $REPO" >> "$LOG"
exec "$PYTHON" -m qlib_yahoo.webapp >> "$LOG" 2>&1

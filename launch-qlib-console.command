#!/bin/bash
# macOS: double-click in Finder to open the Qlib workbench from this checkout (Terminal shows its log).
cd "$(dirname "$0")" || exit 1
if [ ! -x env/qlib/bin/python ]; then
  echo "The Qlib environment is missing. Run this once in Terminal:"
  echo "  cd \"$(pwd)\" && scripts/macos/setup.sh"
  read -r -p "Press Return to close." _
  exit 1
fi
exec env/qlib/bin/python -m qlib_yahoo.webapp

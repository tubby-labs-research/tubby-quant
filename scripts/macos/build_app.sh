#!/bin/bash
# Build "Tubby Qlib Console.app" for this checkout (default destination: ~/Applications).
# The bundle is a thin launcher: code, environment and data stay in the checkout/data folder.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/macos/build_app.sh [--dest DIR] [--data-root DIR]
  --dest DIR       Folder for the app (default: ~/Applications)
  --data-root DIR  Data folder the app should use (default: ~/Qlib, or $TUBBY_QLIB_DATA_ROOT)
USAGE
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$HOME/Applications"
DATA_ROOT="${TUBBY_QLIB_DATA_ROOT:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ "$(uname -s)" = "Darwin" ] || { echo "build_app.sh builds a macOS app; run it on macOS." >&2; exit 1; }
PYTHON="$ROOT/env/qlib/bin/python"
[ -x "$PYTHON" ] || { echo "Missing $PYTHON — run scripts/macos/setup.sh first." >&2; exit 1; }

NAME="Tubby Qlib Console"
APP="$DEST/$NAME.app"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
BUNDLE="$WORK/$NAME.app"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources" "$DEST"

VERSION="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo local)"
"$PYTHON" - "$BUNDLE/Contents/Info.plist" "$VERSION" <<'PY'
import plistlib, sys
plistlib.dump({
    "CFBundleName": "Tubby Qlib Console", "CFBundleDisplayName": "Tubby Qlib Console",
    "CFBundleIdentifier": "org.tubbylabs.qlib-console", "CFBundleExecutable": "TubbyQlibConsole",
    "CFBundleIconFile": "AppIcon", "CFBundlePackageType": "APPL", "CFBundleShortVersionString": "1.0",
    "CFBundleVersion": sys.argv[2], "CFBundleInfoDictionaryVersion": "6.0",
    "LSMinimumSystemVersion": "12.0", "LSApplicationCategoryType": "public.app-category.finance",
    "NSHighResolutionCapable": True, "NSRequiresAquaSystemAppearance": False,
    "NSHumanReadableCopyright": "Tubby Labs research tool — local backtests only, no trading.",
}, open(sys.argv[1], "wb"))
PY
"$PYTHON" "$ROOT/scripts/macos/make_icon.py" "$BUNDLE/Contents/Resources/AppIcon.icns" >/dev/null
cp "$ROOT/scripts/macos/app_launcher.sh" "$BUNDLE/Contents/MacOS/TubbyQlibConsole"
chmod 755 "$BUNDLE/Contents/MacOS/TubbyQlibConsole"
printf '%s' "$ROOT" > "$BUNDLE/Contents/Resources/repo-path"
if [ -n "$DATA_ROOT" ]; then
  printf '%s' "$DATA_ROOT" > "$BUNDLE/Contents/Resources/data-root"
fi
codesign --force --sign - "$BUNDLE" >/dev/null 2>&1 || true

if [ -e "$APP" ]; then
  # Replace only a bundle this script built (it records a checkout path).
  if [ ! -f "$APP/Contents/Resources/repo-path" ]; then
    echo "Refusing to replace $APP: it was not built by this script." >&2
    exit 1
  fi
  rm -rf "$APP"
fi
mv "$BUNDLE" "$APP"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[ -x "$LSREGISTER" ] && "$LSREGISTER" -f "$APP" >/dev/null 2>&1 || true
echo "Built: $APP"
echo "Open it from Finder, Launchpad or: open \"$APP\""

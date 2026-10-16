"""Launch the desktop app: ``python -m qlib_yahoo.webapp [--browser]``.

A native window (pywebview/WebKit) shows the UI served on 127.0.0.1. Without
pywebview, or with ``--browser``, the default browser opens it instead.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from qlib_yahoo import DEFAULT_DATA_ROOT
from qlib_yahoo.webapp.server import start_server
from qlib_yahoo.webapp.service import ConsoleService

APP_NAME = "Tubby Qlib Console"


class NativeBridge:
    """Methods exposed to the page as ``window.pywebview.api`` (native dialogs only)."""

    def __init__(self) -> None:
        self._window = None  # private: pywebview exposes public attributes of js_api objects to the page

    def choose_table(self, current: str = "") -> str | None:
        import webview

        start = Path(current).expanduser().parent if current else Path.home() / "Desktop"
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN, directory=str(start if start.is_dir() else Path.home()),
            allow_multiple=False, file_types=("Stooq tables or ASCII zip (*.parquet;*.csv;*.zip)", "All files (*.*)"))
        return str(result[0]) if result else None


def _macos_identity() -> None:
    """Show the app's own name and icon in the menu bar and Dock (Python.app would otherwise appear)."""
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication, NSImage
        from Foundation import NSBundle, NSData

        info = NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = APP_NAME
        info["CFBundleDisplayName"] = APP_NAME
        from qlib_yahoo.icon import png_bytes

        data = NSData.dataWithBytes_length_(png_bytes(512), len(png_bytes(512)))
        NSApplication.sharedApplication().setApplicationIconImage_(NSImage.alloc().initWithData_(data))
    except Exception:  # cosmetic only
        pass


def _windows_window_icon(window) -> None:
    """Give the WinForms window the app icon (pywebview uses the Python executable's otherwise)."""
    if sys.platform != "win32":
        return
    try:
        import clr  # pythonnet, which pywebview uses on Windows

        clr.AddReference("System.Drawing")
        from System import Action
        from System.Drawing import Icon

        from qlib_yahoo.desktop import windows_icon_path

        form, icon = window.native, Icon(str(windows_icon_path()))
        form.Invoke(Action(lambda: setattr(form, "Icon", icon)))
    except Exception:  # cosmetic only
        pass


def _dark_mode() -> bool:
    if sys.platform == "win32":
        from qlib_yahoo.desktop import windows_dark_mode

        return windows_dark_mode()
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True, timeout=3)
        return "Dark" in result.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--browser", action="store_true", help="Open in the default browser instead of a native window.")
    parser.add_argument("--no-open", action="store_true", help="Serve only and print the start URL (for testing).")
    parser.add_argument("--port", type=int, default=0, help="Loopback port (default: random free port).")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--create-shortcuts", action="store_true",
                        help="Windows: add Start menu and Desktop shortcuts that open this workbench, then exit.")
    args = parser.parse_args(argv)
    if args.create_shortcuts:
        from qlib_yahoo.desktop import create_windows_shortcuts

        explicit = "--data-root" in (argv if argv is not None else sys.argv[1:])
        for path in create_windows_shortcuts(Path(__file__).resolve().parents[2], args.data_root if explicit else None):
            print(f"Created {path}")
        return 0

    from qlib_yahoo.desktop import settings_dir

    data_root = args.data_root.expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    service = ConsoleService(data_root, settings_path=settings_dir() / "webapp.json")
    server, _thread = start_server(service, args.port)
    url = server.start_url
    reason = ""
    try:
        webview = None if args.browser or args.no_open else __import__("webview")
    except ImportError:
        webview, reason = None, "The app window needs pywebview (requirements-qlib-windows.txt).\n\n"
    try:
        if webview is None:
            _serve_in_browser(server, url, open_page=not args.no_open, reason=reason)
        else:
            try:
                _native_window(webview, service, url)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # e.g. Windows without the WebView2 runtime: the browser still works
                from qlib_yahoo.desktop import console_logger

                console_logger().exception("native window failed; opening the browser")
                _serve_in_browser(server, url, reason=f"The app window could not open ({exc}).\n\n")
    except KeyboardInterrupt:
        pass
    finally:
        service.research.close()
        if service.busy():
            # Worker tasks stop on cancel; an in-process report cannot be interrupted, so let it finish.
            patience = 600 if getattr(service.current, "kind", "") == "report" else 10
            service.cancel()
            deadline = time.time() + patience
            while service.busy() and time.time() < deadline:
                time.sleep(0.2)
        server.shutdown()
    return 0


def _native_window(webview, service: ConsoleService, url: str) -> None:
    bridge = NativeBridge()
    window = webview.create_window(
        APP_NAME, url, width=1360, height=900, min_size=(1040, 700), js_api=bridge,
        background_color="#1b1b1d" if _dark_mode() else "#f5f5f7", text_select=True,
        localization={"global.quitConfirmation": "A task is still running. Stop it and quit? "
                                                 "(A report that is being written finishes first.)"})
    bridge._window = window
    window.events.shown += lambda: _windows_window_icon(window)

    def closing() -> bool:
        # Runs on the GUI main thread: opening a dialog here and waiting for it deadlocks.
        # pywebview asks for confirmation itself, on this thread, while confirm_close is set.
        window.confirm_close = service.busy()
        return True

    window.events.closing += closing
    webview.start(func=_macos_identity, private_mode=True)


def _serve_in_browser(server, url: str, *, open_page: bool = True, reason: str = "") -> None:
    """Serve until interrupted; started without a console (pythonw), a dialog stands in for Ctrl+C."""
    print(f"{APP_NAME} is running at {server.origin} (press Ctrl+C to stop).", flush=True)
    if not open_page:
        print(url, flush=True)
    else:
        webbrowser.open(url)
    if sys.platform == "win32" and Path(sys.executable).name.lower() == "pythonw.exe":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, f"{reason}The workbench is open in your browser at {server.origin}.\n\n"
                                               "Keep this box open while you use it; press OK to stop the workbench.",
                                         APP_NAME, 0x40)
        return

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())

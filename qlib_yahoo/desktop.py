"""Small cross-platform desktop helpers for the console (paths, opening files, settings, logs)."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

APP_NAME = "Tubby Qlib Console"


def settings_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "tubby-qlib-console"


def logs_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    return settings_dir() / "logs"


def load_settings(path: Path) -> dict[str, Any]:
    """Return saved preferences; a missing or damaged file yields defaults, never an error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def save_settings(path: Path, values: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        logging.getLogger(__name__).warning("could not save console settings to %s", path, exc_info=True)


def console_logger() -> logging.Logger:
    """A rotating file log for the window process; failures to create it are not fatal."""
    logger = logging.getLogger("qlib_yahoo.console")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    try:
        directory = logs_dir()
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(directory / "console.log", maxBytes=2_000_000,
                                                       backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    except OSError:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def open_path(path: Path) -> None:
    """Open a file or folder with the system handler (Finder, Explorer or xdg-open)."""
    path = Path(path)
    if not path.exists():
        raise OSError(f"{path} does not exist")
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True, capture_output=True)
    else:
        subprocess.run(["xdg-open", str(path)], check=True, capture_output=True)


def reveal_path(path: Path) -> None:
    """Select the item in Finder/Explorer; elsewhere open its folder."""
    path = Path(path)
    if not path.exists():
        raise OSError(f"{path} does not exist")
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=True, capture_output=True)
    elif os.name == "nt":
        subprocess.run(["explorer", f"/select,{path}"], capture_output=True)
    else:
        open_path(path if path.is_dir() else path.parent)


def move_to_system_trash(path: Path) -> None:
    """Move a file or folder to the operating system's trash (never deletes it outright)."""
    path = Path(path)
    if not path.exists():
        raise OSError(f"{path} does not exist")
    if sys.platform == "darwin":
        trash = Path.home() / ".Trash"
        target = trash / path.name
        if target.exists():
            from datetime import datetime

            target = trash / f"{path.name} {datetime.now():%Y-%m-%d %H.%M.%S}"
        import shutil

        shutil.move(str(path), str(target))
        return
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                        ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

        fo_delete, allow_undo, no_confirmation, no_error_ui, silent = 3, 0x40, 0x10, 0x400, 0x4
        operation = SHFILEOPSTRUCTW(None, fo_delete, str(path.resolve()) + "\0", None,
                                    allow_undo | no_confirmation | no_error_ui | silent, False, None, None)
        if ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation)) or operation.fAnyOperationsAborted:
            raise OSError(f"Windows could not move {path} to the Recycle Bin")
        return
    # freedesktop.org trash (Linux and other Unix desktops)
    from datetime import datetime
    from urllib.parse import quote

    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "Trash"
    (base / "files").mkdir(parents=True, exist_ok=True)
    (base / "info").mkdir(parents=True, exist_ok=True)
    name, number = path.name, 1
    while (base / "files" / name).exists() or (base / "info" / f"{name}.trashinfo").exists():
        number += 1
        name = f"{path.name}.{number}"
    (base / "info" / f"{name}.trashinfo").write_text(
        f"[Trash Info]\nPath={quote(str(path.resolve()))}\nDeletionDate={datetime.now():%Y-%m-%dT%H:%M:%S}\n",
        encoding="utf-8")
    import shutil

    shutil.move(str(path), str(base / "files" / name))


# ----------------------------------------------------------------------------- Windows desktop integration

SHORTCUT_NAME = f"{APP_NAME}.lnk"


def windows_dark_mode() -> bool:
    """Whether Windows apps use the dark theme (Settings > Personalization > Colors)."""
    if os.name != "nt":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def windows_icon_path(folder: Path | None = None) -> Path:
    """Write the app icon as an .ico beside the settings (rewritten when the drawing changes) and return its path."""
    from qlib_yahoo.icon import ico_bytes

    path = (folder or settings_dir()) / "tubby-qlib.ico"
    data = ico_bytes()
    if not path.is_file() or path.read_bytes() != data:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    return path


def _ps_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def shortcut_script(target: Path, arguments: str, workdir: Path, icon: Path, *, desktop: bool = True) -> str:
    """PowerShell that writes the Start menu (and optionally Desktop) shortcut and prints each path."""
    folders = "[Environment]::GetFolderPath('Programs')" + (", [Environment]::GetFolderPath('Desktop')" if desktop else "")
    return "\n".join([
        "$ErrorActionPreference = 'Stop'",
        "$shell = New-Object -ComObject WScript.Shell",
        f"foreach ($folder in @({folders})) {{",
        f"  $path = Join-Path $folder {_ps_quote(SHORTCUT_NAME)}",
        "  $link = $shell.CreateShortcut($path)",
        f"  $link.TargetPath = {_ps_quote(target)}",
        f"  $link.Arguments = {_ps_quote(arguments)}",
        f"  $link.WorkingDirectory = {_ps_quote(workdir)}",
        f"  $link.IconLocation = {_ps_quote(str(icon) + ',0')}",
        f"  $link.Description = {_ps_quote('Qlib research workbench: backtests, Alpha Lab and market data')}",
        "  $link.Save()",
        "  Write-Output $path",
        "}",
    ])


def create_windows_shortcuts(root: Path, data_root: Path | None = None, *, desktop: bool = True,
                             runner: Any = subprocess.run) -> list[Path]:
    """Start menu (and Desktop) shortcuts that open the workbench in its own window, without a console."""
    if os.name != "nt" and runner is subprocess.run:
        raise OSError("Windows shortcuts can only be created on Windows")
    root = Path(root).resolve()
    target = root / "env" / "qlib" / "Scripts" / "pythonw.exe"
    folder = str(Path(data_root)) if data_root else ""
    if folder.endswith("\\"):
        folder += "\\"  # "D:\\" would otherwise escape the closing quote when Windows splits the arguments
    arguments = "-m qlib_yahoo.webapp" + (f' --data-root "{folder}"' if folder else "")
    script = shortcut_script(target, arguments, root, windows_icon_path(), desktop=desktop)
    result = runner(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
                    capture_output=True, text=True)
    if result.returncode:
        raise OSError(f"could not create the shortcuts: {(result.stderr or result.stdout).strip()}")
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]

"""Lexical Windows cache-I/O paths; no filesystem or security-boundary resolution."""
import ntpath
import os
import re
from pathlib import Path


def windows_extended_path(value: str) -> str:
    """Use the absolute Unicode namespace without resolving symlinks."""
    if value.replace("/", "\\").startswith("\\\\?\\"):
        tail = value[4:]
        drive = re.match(r"^[A-Za-z]:\\", tail)
        parts = tail[4:].split("\\") if tail[:4].upper() == "UNC\\" else []
        unc = len(parts) >= 2 and all(
            part not in {"", ".", ".."} and re.fullmatch(r'[^\\/:<>"|?*\x00]+', part)
            for part in parts[:2]
        )
        if (not value.startswith("\\\\?\\") or "/" in value or "\x00" in value
                or not (drive or unc)):
            raise ValueError("Only extended drive-absolute or UNC cache paths are supported")
        return ntpath.abspath(value)
    value = ntpath.abspath(value)
    if value.startswith("\\\\.\\"):
        raise ValueError("Device paths are not cache directories")
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def cache_io_path(value: Path) -> Path:
    """Only cache callers opt in; other platforms retain their original paths."""
    return Path(windows_extended_path(str(value))) if os.name == "nt" else Path(value)

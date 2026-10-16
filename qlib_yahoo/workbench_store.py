"""Workbench bookkeeping beside the data: stars, labels and notes, and a recoverable trash for runs.

Saved runs and the alpha library stay as the runner wrote them. Annotations live
in ``<data root>/workbench/annotations.json``; "deleting" a run moves its folder
to ``<data root>/trash/runs/`` where it can be restored, and emptying that trash
moves the folders to the operating system's trash. The app never destroys a run
outright. Never imports Qlib.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

KINDS = ("runs", "factors", "alphas")
RUN_NAME = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,127}\Z")
HEX_KEY = re.compile(r"[0-9a-f]{16}\Z")
MAX_LABEL = 120
MAX_NOTE = 2000
_LOCK = threading.Lock()


def factor_key(alpha: str) -> str:
    """A stable key for one alpha expression: whitespace-insensitive, 16 hex characters."""
    text = " ".join(str(alpha or "").split())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Annotations:
    """Stars, labels, notes and soft deletions keyed by run folder name, factor key or alpha id."""

    def __init__(self, data_root: Path):
        self.path = Path(data_root) / "workbench" / "annotations.json"

    def load(self) -> dict[str, dict[str, dict[str, Any]]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = {}
        if not isinstance(value, dict):
            value = {}
        return {kind: value.get(kind) if isinstance(value.get(kind), dict) else {} for kind in KINDS}

    def update(self, kind: str, key: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Apply ``fields`` (starred, label, note, deleted) to one entry; return the entry."""
        if kind not in KINDS:
            raise ValueError("Unknown annotation kind.")
        pattern = RUN_NAME if kind == "runs" else HEX_KEY
        if not isinstance(key, str) or not pattern.fullmatch(key):
            raise ValueError("Invalid annotation key.")
        unknown = set(fields) - {"starred", "label", "note", "deleted"}
        if unknown:
            raise ValueError(f"Unknown annotation fields: {', '.join(sorted(unknown))}.")
        with _LOCK:
            data = self.load()
            entry = dict(data[kind].get(key) or {})
            if "starred" in fields:
                if not isinstance(fields["starred"], bool):
                    raise ValueError("starred must be true or false.")
                entry["starred"] = fields["starred"]
            for name, limit in (("label", MAX_LABEL), ("note", MAX_NOTE)):
                if name in fields:
                    text = " ".join(str(fields[name] or "").split()) if name == "label" else str(fields[name] or "").strip()
                    if len(text) > limit:
                        raise ValueError(f"{name.title()} is limited to {limit} characters.")
                    entry[name] = text
            if "deleted" in fields:
                if not isinstance(fields["deleted"], bool):
                    raise ValueError("deleted must be true or false.")
                entry["deleted_at"] = _now() if fields["deleted"] else ""
            entry = {name: value for name, value in entry.items() if name != "updated_at" and value not in ("", False, None)}
            if entry:
                entry["updated_at"] = _now()
                data[kind][key] = entry
            else:
                data[kind].pop(key, None)
            _write_json(self.path, data)
        return entry


# ----------------------------------------------------------------------------- run trash


def trash_folder(data_root: Path) -> Path:
    return Path(data_root) / "trash" / "runs"


def _index_path(data_root: Path) -> Path:
    return Path(data_root) / "trash" / "index.json"


def _read_index(data_root: Path) -> dict[str, Any]:
    try:
        value = json.loads(_index_path(data_root).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _run_status(folder: Path) -> str:
    try:
        return str(json.loads((folder / "manifest.json").read_text(encoding="utf-8")).get("status", "unknown"))
    except (OSError, ValueError, AttributeError):
        return "unknown"


def trash_runs(data_root: Path, names: Iterable[str], *, busy: bool = False) -> list[str]:
    """Move run folders into the app's trash; a run still being written (while a task runs) is refused."""
    data_root = Path(data_root)
    runs, trash = data_root / "runs", trash_folder(data_root)
    names = list(dict.fromkeys(names))
    for name in names:
        if not RUN_NAME.fullmatch(name) or not (runs / name).is_dir() or (runs / name).is_symlink():
            raise ValueError(f"No saved run named {name}.")
        if busy and _run_status(runs / name) == "running":
            raise ValueError(f"{name} is still being written by the running task.")
    trash.mkdir(parents=True, exist_ok=True)
    moved = []
    with _LOCK:
        index = _read_index(data_root)
        for name in names:
            target = trash / name
            if target.exists():
                raise ValueError(f"The trash already holds a run named {name}; restore or empty it first.")
            os.replace(runs / name, target)
            index[name] = {"trashed_at": _now()}
            moved.append(name)
        _write_json(_index_path(data_root), index)
    return moved


def restore_runs(data_root: Path, names: Iterable[str]) -> list[str]:
    data_root = Path(data_root)
    runs, trash = data_root / "runs", trash_folder(data_root)
    restored = []
    with _LOCK:
        index = _read_index(data_root)
        for name in dict.fromkeys(names):
            if not RUN_NAME.fullmatch(name) or not (trash / name).is_dir():
                raise ValueError(f"No run named {name} in the trash.")
            if (runs / name).exists():
                raise ValueError(f"A run named {name} already exists.")
            runs.mkdir(parents=True, exist_ok=True)
            os.replace(trash / name, runs / name)
            index.pop(name, None)
            restored.append(name)
        _write_json(_index_path(data_root), index)
    return restored


def list_trash(data_root: Path) -> list[dict[str, Any]]:
    """Trashed runs, most recently trashed first, with the fields the run list shows."""
    from qlib_yahoo.workspace import run_row

    trash = trash_folder(data_root)
    if not trash.is_dir():
        return []
    index = _read_index(data_root)
    rows = []
    for folder in trash.iterdir():
        if folder.is_dir() and RUN_NAME.fullmatch(folder.name):
            row = run_row(folder)
            row["trashed_at"] = (index.get(folder.name) or {}).get("trashed_at", "")
            rows.append(row)
    rows.sort(key=lambda row: row["trashed_at"], reverse=True)
    return rows


def empty_trash(data_root: Path, names: Iterable[str] | None = None) -> list[str]:
    """Move trashed runs to the operating system's trash (all of them when ``names`` is None)."""
    from qlib_yahoo.desktop import move_to_system_trash

    data_root = Path(data_root)
    trash = trash_folder(data_root)
    chosen = [folder.name for folder in trash.iterdir() if folder.is_dir()] if names is None and trash.is_dir() else list(names or [])
    emptied = []
    with _LOCK:
        index = _read_index(data_root)
        for name in dict.fromkeys(chosen):
            if not RUN_NAME.fullmatch(name) or not (trash / name).is_dir():
                raise ValueError(f"No run named {name} in the trash.")
            move_to_system_trash(trash / name)
            index.pop(name, None)
            emptied.append(name)
        _write_json(_index_path(data_root), index)
    return emptied


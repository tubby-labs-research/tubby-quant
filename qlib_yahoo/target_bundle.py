"""Sealed close-decision targets for the existing next-open signed account.

Dates are decision dates, never execution dates. No allocation policy is applied
here. Float32 conversion can only reduce a rounding overflow at the gross cap.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import zipfile

from .dataset import DatasetError

MAX_BYTES = 512 * 1024 * 1024
MAX_MANIFEST = 8 * 1024 * 1024
CONTRACT = {"schema_version": 1, "input_kind": "target_weights",
            "decision_timing": "session_close", "execution_timing": "next_session_open"}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise DatasetError("target bundle path must be absolute without parent traversal")
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise DatasetError("target bundle links/reparse points are forbidden")
    return path


def _json(data):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)


def _labels(dates, symbols):
    from datetime import date
    if not isinstance(dates, list) or not dates or any(
            not isinstance(day, str) or date.fromisoformat(day).isoformat() != day for day in dates):
        raise DatasetError("target dates must be nonempty ISO daily decision dates")
    if dates != sorted(set(dates)):
        raise DatasetError("target decision dates must be ordered and unique")
    if (not isinstance(symbols, list) or not symbols or len(set(symbols)) != len(symbols)
            or any(not isinstance(symbol, str) or not symbol or symbol.strip() != symbol for symbol in symbols)):
        raise DatasetError("target symbols must be nonempty unique ordered labels")


def _metadata(value):
    if not isinstance(value, dict) or not _hash(value.get("dataset_manifest_sha256")):
        raise DatasetError("targets require a bound dataset_manifest_sha256")
    if not isinstance(value.get("price_identity"), (dict, str)) or not value["price_identity"]:
        raise DatasetError("targets require price_identity")
    if value.get("weight_policy") not in ("lab_close_targets_v1", "portfolio_fixed_v1"):
        raise DatasetError("unsupported target weight_policy")
    for key in ("candidate_result_digests", "attempt_ids"):
        items = value.get(key)
        if (not isinstance(items, list) or not items or any(not isinstance(x, str) or not x for x in items)
                or (key == "candidate_result_digests" and not all(_hash(x) for x in items))):
            raise DatasetError(f"targets require nonempty {key}")
    if not isinstance(value.get("assumptions"), (dict, list)):
        raise DatasetError("targets require explicit assumptions")
    # Detach caller-owned containers and reject NaN/Infinity anywhere in metadata.
    return _json(json.dumps(value, allow_nan=False))


def _matrix(values, shape, *, writing=False):
    import numpy as np
    array = np.asarray(values)
    if array.shape != shape or array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8):
        raise DatasetError("target weights need a dates-by-symbols float32/float64 matrix")
    if array.size * 4 > MAX_BYTES - 4096 or not np.isfinite(array).all():
        raise DatasetError("target weights exceed size limit or contain nonfinite values")
    totals = [math.fsum(abs(float(x)) for x in row) for row in array]
    tolerance = float(np.finfo(np.float32).eps) if writing and array.dtype.itemsize == 4 else 1e-12
    if any(total > 1 + tolerance for total in totals):
        raise DatasetError("target gross weight exceeds one")
    return array


def write_bundle(directory, *, dates, symbols, weights, metadata):
    """Publish a new directory atomically; never replace an existing bundle.

    Metadata requires dataset_manifest_sha256, price_identity,
    candidate_result_digests (SHA256 list), attempt_ids (string list),
    weight_policy and assumptions (object/list). Extra lineage fields are retained.
    """
    import numpy as np
    staging = None
    try:
        directory = _path(directory)
        dates, symbols = list(dates), list(symbols)
        _labels(dates, symbols)
        metadata = _metadata(metadata)
        original = _matrix(weights, (len(dates), len(symbols)), writing=True)
        matrix = original.astype("<f4", order="C", copy=True)
        corrections = 0
        for row in matrix:
            while math.fsum(abs(float(x)) for x in row) > 1 + 1e-12:
                column = int(np.argmax(np.abs(row)))
                row[column] = np.nextafter(row[column], np.float32(0))
                corrections += 1
        if directory.exists() or not directory.parent.is_dir():
            raise DatasetError("target bundle needs a new directory in an existing parent")
        staging = Path(tempfile.mkdtemp(prefix=".targets-", dir=directory.parent))
        with (staging / "weights.npz").open("xb") as stream:
            np.savez(stream, weights=matrix)
            stream.flush()
            os.fsync(stream.fileno())
        payload = (staging / "weights.npz").read_bytes()
        manifest = {**CONTRACT, "dates": dates, "symbols": symbols, "metadata": metadata,
                    "weights": {"file": "weights.npz", "sha256": _digest(payload),
                                "size_bytes": len(payload), "shape": list(matrix.shape), "dtype": "float32"},
                    "precision": {"storage": "float32", "gross_cap_ulp_reductions": corrections,
                                  "max_absolute_conversion": float(np.max(np.abs(matrix.astype(float) - original)))}}
        encoded = (json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(encoded) > MAX_MANIFEST:
            raise DatasetError("target manifest exceeds size limit")
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename fails if destination exists; do not use replace.
        if directory.exists():
            raise DatasetError("target bundle destination already exists")
        staging.rename(directory)
        staging = None
        return {"path": str(directory), "sha256": _digest(encoded)}
    except (OSError, ValueError, TypeError, OverflowError) as error:
        raise DatasetError(f"invalid target bundle: {error}") from error
    finally:
        if staging is not None:
            shutil.rmtree(staging)


def load_bundle(envelope):
    """Verify envelope, manifest and bounded NPZ bytes without pickle loading."""
    import numpy as np
    try:
        if (not isinstance(envelope, dict) or set(envelope) != {"path", "sha256"}
                or not isinstance(envelope["path"], str) or not _hash(envelope["sha256"])):
            raise DatasetError("targets require a bound path/sha256 envelope")
        directory = _path(envelope["path"])
        manifest_path = _path(directory / "manifest.json")
        if (not directory.is_dir() or not manifest_path.is_file() or manifest_path.stat().st_nlink != 1
                or manifest_path.stat().st_size > MAX_MANIFEST):
            raise DatasetError("invalid target manifest size/directory")
        encoded = manifest_path.read_bytes()
        if _digest(encoded) != envelope["sha256"]:
            raise DatasetError("target manifest digest mismatch")
        manifest = _json(encoded)
        if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
                or any(manifest.get(k) != v for k, v in CONTRACT.items())):
            raise DatasetError("unsupported target bundle schema/timing/input_kind")
        dates, symbols = manifest["dates"], manifest["symbols"]
        _labels(dates, symbols)
        metadata = _metadata(manifest["metadata"])
        record = manifest["weights"]
        if not isinstance(record, dict):
            raise DatasetError("invalid target matrix record")
        shape = (len(dates), len(symbols))
        expected = math.prod(shape) * 4
        if (expected > MAX_BYTES - 4096 or record.get("file") != "weights.npz"
                or record.get("dtype") != "float32" or record.get("shape") != list(shape)
                or type(record.get("size_bytes")) is not int or not _hash(record.get("sha256"))):
            raise DatasetError("invalid target matrix schema/shape/size/dtype")
        path = _path(directory / "weights.npz")
        if (not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size != record["size_bytes"]
                or not 0 < path.stat().st_size <= expected + 4096):
            raise DatasetError("target matrix file size mismatch")
        payload = path.read_bytes()
        if _digest(payload) != record["sha256"]:
            raise DatasetError("target matrix checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if (len(entries) != 1 or entries[0].filename != "weights.npy"
                    or entries[0].file_size > expected + 1024
                    or entries[0].compress_type != zipfile.ZIP_STORED):
                raise DatasetError("invalid target NPZ members/size/compression")
            with archive.open(entries[0]) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise DatasetError("unsupported target NPY header")
                header_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                if header_shape != shape or fortran or dtype != np.dtype("<f4"):
                    raise DatasetError("invalid target NPY shape/dtype/order")
                if entries[0].file_size != stream.tell() + expected:
                    raise DatasetError("target NPY payload length mismatch")
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            weights = _matrix(arrays["weights"], shape).copy()
        weights.flags.writeable = False
        return {"dates": dates, "symbols": symbols, "weights": weights, "metadata": metadata,
                "identity": {**CONTRACT, **envelope, "weights_sha256": record["sha256"],
                             "precision": manifest.get("precision")}}
    except (OSError, ValueError, TypeError, KeyError, OverflowError, zipfile.BadZipFile, EOFError) as error:
        raise DatasetError(f"invalid target bundle: {error}") from error


def align_bundle(bundle, *, sessions, signal_days, symbols, eligible, spans=None):
    """Validate calendar/membership, selecting rows without moving decision dates."""
    import numpy as np
    import pandas as pd
    dates = pd.DatetimeIndex(bundle["dates"])
    sessions = pd.DatetimeIndex(sessions)
    signal_days = pd.DatetimeIndex(signal_days)
    for calendar in (sessions, signal_days):
        if (calendar.hasnans or calendar.has_duplicates or not calendar.is_monotonic_increasing
                or calendar.tz is not None or not calendar.equals(calendar.normalize())):
            raise DatasetError("target calendar must contain ordered unique daily sessions")
    if len(dates.difference(sessions)):
        raise DatasetError("target decision date is not a dataset calendar session")
    if len(pd.DatetimeIndex(signal_days).difference(dates)):
        raise DatasetError("missing target decision date for execution window")
    unknown = set(bundle["symbols"]) - set(symbols)
    if unknown:
        raise DatasetError(f"unknown target symbols: {sorted(unknown)}")
    matrix = bundle["weights"]
    # Check every supplied row, including rows outside this selected run window.
    for column, symbol in enumerate(bundle["symbols"]):
        active = matrix[:, column] != 0
        if spans is not None:
            member = np.zeros(len(dates), dtype=bool)
            for begin, end in spans.get(symbol, ()):
                member |= (dates >= begin) & (dates <= end)
            if np.any(active & ~member):
                raise DatasetError(f"nonzero target outside dated membership: {symbol}")
        if symbol not in eligible and np.any(active[dates.isin(signal_days)]):
            raise DatasetError(f"nonzero target outside eligible symbols: {symbol}")
    rows = dates.get_indexer(signal_days)
    index = pd.MultiIndex.from_product([signal_days, bundle["symbols"]], names=["datetime", "instrument"])
    return pd.Series(matrix[rows].ravel(), index=index, name="weight", dtype=np.float32)

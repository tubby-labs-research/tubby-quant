"""Independent read-only inventory of supplied SEC tabular archives.

Counts TSV records without using the importer, retaining no source rows.
Use --logical for independent CSV parsing of quoted tabs/newlines. The faster
physical line count is only a preliminary inventory. Output is local evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile


def audit_archive(path: Path, *, logical: bool = False) -> dict:
    fingerprint = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            fingerprint.update(chunk)
    tables = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = member.filename
            if Path(name).suffix.lower() not in (".txt", ".tsv"):
                continue
            with archive.open(member) as handle:
                header = handle.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
                count, tail, size = 0, b"", 0
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    count += chunk.count(b"\n")
                    size += len(chunk)
                    tail = chunk[-1:]
                if tail and tail != b"\n":
                    count += 1
            if logical:
                with archive.open(member) as handle:
                    reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline=""),
                                        delimiter="\t", strict=True)
                    parsed_header = next(reader)
                    if parsed_header != header:
                        raise ValueError(f"Header mismatch: {path.name}/{name}")
                    count = 0
                    for row in reader:
                        if len(row) != len(header):
                            raise ValueError(f"Malformed logical record: {path.name}/{name}:{count + 1}")
                        count += 1
            tables[Path(name).stem.lower()] = {
                "member": name, "columns": header, "rows": count,
                "uncompressed_bytes": member.file_size, "body_bytes": size,
                "crc32": member.CRC,
            }
    return {"name": path.name, "sha256": fingerprint.hexdigest(),
            "bytes": path.stat().st_size, "tables": tables}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start", default="2012q1")
    parser.add_argument("--end", default="2026q2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical", action="store_true", help="Parse quoted TSV records independently with csv.reader")
    args = parser.parse_args()
    paths = sorted(p for p in args.source.glob("EDGAR-US-FS-*.zip")
                   if args.start <= p.stem.removeprefix("EDGAR-US-FS-").lower() <= args.end)
    if not paths:
        parser.error("no matching SEC archives")
    if args.output.exists():
        parser.error("output already exists; choose a new evidence file")
    started = time.monotonic()
    records = []
    for i, path in enumerate(paths, 1):
        records.append(audit_archive(path, logical=args.logical))
        print(f"[{i}/{len(paths)}] {path.name}: "
              f"{records[-1]['tables'].get('num', {}).get('rows', 0):,} numeric records", flush=True)
    totals = {}
    for record in records:
        for name, table in record["tables"].items():
            totals[name] = totals.get(name, 0) + table["rows"]
    receipt = {"schema_version": 1, "method": ("independent csv.reader logical TSV records" if args.logical
                                               else "preliminary physical TSV line counts") + " plus ZIP CRC and SHA256",
               "start": args.start, "end": args.end, "archives": records,
               "archive_count": len(records), "rows": totals,
               "elapsed_seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps({k: receipt[k] for k in ("archive_count", "rows", "elapsed_seconds")}))


if __name__ == "__main__":
    main()

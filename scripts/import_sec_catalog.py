"""Explicit local-only SEC catalog import. Does not download or select defaults."""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qlib_yahoo.sec_catalog import build_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-glob', help='Explicit ZIP input glob; quote it in the shell')
    source.add_argument('--source-dir', type=Path, help='Directory containing quarterly ZIPs, or top-level table files')
    parser.add_argument('--start-quarter', help='YYYYqN, requires --end-quarter')
    parser.add_argument('--end-quarter', help='YYYYqN, requires --start-quarter')
    parser.add_argument('--ticker-map', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    def quarter(value):
        if not value or not re.fullmatch(r'\d{4}q[1-4]', value.lower()):
            parser.error('Quarter must be YYYYqN')
        return int(value[:4]) * 4 + int(value[-1]) - 1
    if bool(args.start_quarter) != bool(args.end_quarter):
        parser.error('Specify both range endpoints')
    paths = sorted(Path(p) for p in glob.glob(args.input_glob)) if args.input_glob else sorted(args.source_dir.glob('*.zip'))
    if args.source_dir and not paths:
        paths = [args.source_dir]
    if args.start_quarter:
        first, last = quarter(args.start_quarter), quarter(args.end_quarter)
        if first > last:
            parser.error('Reversed quarter range')
        selected = {}
        for path in paths:
            match = re.fullmatch(r'(?:EDGAR-US-FS-)?(\d{4}q[1-4])\.zip', path.name, re.I)
            if match and first <= quarter(match[1]) <= last:
                key = quarter(match[1])
                if key in selected:
                    parser.error('Duplicate input quarter: ' + match[1])
                selected[key] = path
        if set(selected) != set(range(first, last + 1)):
            parser.error('Requested range has missing quarterly ZIPs')
        paths = [selected[key] for key in sorted(selected)]
    if not paths:
        parser.error('No supplied inputs matched')
    manifest = build_catalog(paths, args.output, ticker_map=args.ticker_map, progress=lambda message: print(message, flush=True))
    print(json.dumps({'output': str(args.output), 'catalog_sha256': manifest['catalog_sha256'],
                      'counts': {name: table['rows'] for name, table in manifest['tables'].items()}}))


if __name__ == '__main__':
    main()

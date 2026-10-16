"""Write the procedural app icon as an .icns file (macOS iconutil) or a PNG iconset."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qlib_yahoo.icon import png_bytes  # noqa: E402

SIZES = (16, 32, 128, 256, 512)


def write_iconset(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        (folder / f"icon_{size}x{size}.png").write_bytes(png_bytes(size))
        (folder / f"icon_{size}x{size}@2x.png").write_bytes(png_bytes(size * 2))
    return folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination .icns file")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        iconset = write_iconset(Path(temporary) / "AppIcon.iconset")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(args.output)], check=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

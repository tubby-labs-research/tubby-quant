"""Procedural application icon: a rounded tile with a rising chart (no binary assets in Git)."""

from __future__ import annotations

import struct
import zlib

_TOP = (30, 58, 138)       # indigo
_BOTTOM = (15, 118, 110)   # teal
_ACCENT = (251, 191, 36)   # amber
_POINTS = ((0.22, 0.70), (0.36, 0.56), (0.48, 0.62), (0.63, 0.44), (0.79, 0.30))


def render_icon(size: int):
    """Return a (size, size, 4) uint8 RGBA array with anti-aliased edges."""
    import numpy as np

    if not 16 <= size <= 2048:
        raise ValueError("icon size must be between 16 and 2048 pixels")
    grid = (np.mgrid[0:size, 0:size] + 0.5) / size
    v, u = grid[0], grid[1]
    pixel = 1.0 / size

    margin, radius = 0.1, 0.19
    qx = np.abs(u - 0.5) - (0.5 - margin - radius)
    qy = np.abs(v - 0.5) - (0.5 - margin - radius)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - radius
    tile = np.clip(0.5 - outside / pixel, 0, 1)
    shade = np.clip((v - margin) / (1 - 2 * margin), 0, 1)[..., None]
    rgb = (1 - shade) * np.array(_TOP, float) + shade * np.array(_BOTTOM, float)

    # Faint horizontal guides.
    for level in (0.34, 0.50, 0.66):
        guide = np.clip(1 - np.abs(v - level) / (0.004 + pixel), 0, 1) * ((u > 0.18) & (u < 0.82))
        rgb = rgb * (1 - 0.10 * guide[..., None]) + 255 * 0.10 * guide[..., None]

    xs = np.array([p[0] for p in _POINTS])
    ys = np.array([p[1] for p in _POINTS])
    line_y = np.interp(u, xs, ys)
    inside_x = (u >= xs[0]) & (u <= xs[-1])
    area = inside_x & (v > line_y) & (v < 0.78)
    fade = np.clip(1 - (v - line_y) / 0.40, 0, 1)
    fill = 0.22 * fade * area
    rgb = rgb * (1 - fill[..., None]) + 255 * fill[..., None]

    distance = np.full(u.shape, np.inf)
    for (x0, y0), (x1, y1) in zip(_POINTS, _POINTS[1:]):
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((u - x0) * dx + (v - y0) * dy) / (dx * dx + dy * dy), 0, 1)
        distance = np.minimum(distance, np.hypot(u - (x0 + t * dx), v - (y0 + t * dy)))
    stroke = np.clip((0.021 - distance) / pixel + 0.5, 0, 1)
    rgb = rgb * (1 - stroke[..., None]) + 255 * stroke[..., None]

    end_x, end_y = _POINTS[-1]
    dot_distance = np.hypot(u - end_x, v - end_y)
    ring = np.clip((0.062 - dot_distance) / pixel + 0.5, 0, 1)
    dot = np.clip((0.046 - dot_distance) / pixel + 0.5, 0, 1)
    rgb = rgb * (1 - ring[..., None]) + 255 * ring[..., None]
    rgb = rgb * (1 - dot[..., None]) + np.array(_ACCENT, float) * dot[..., None]

    image = np.empty((size, size, 4), dtype=np.uint8)
    image[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    image[..., 3] = np.clip(np.rint(tile * 255), 0, 255).astype(np.uint8)
    return image


def png_bytes(size: int) -> bytes:
    """Encode the icon as a PNG using only zlib/struct."""
    image = render_icon(size)
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def ico_bytes(sizes: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)) -> bytes:
    """A Windows .ico holding PNG-compressed images (the format Windows has read since Vista)."""
    images = [png_bytes(size) for size in sizes]
    offset = 6 + 16 * len(images)
    entries, data = b"", b""
    for size, image in zip(sizes, images):
        # width/height 0 means 256; one colour plane, 32 bits per pixel
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(image), offset + len(data))
        data += image
    return struct.pack("<HHH", 0, 1, len(images)) + entries + data

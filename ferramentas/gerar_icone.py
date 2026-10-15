"""Desenha o ícone do Transcritor (transcritor.ico) só com numpy: balão de fala + ondas de som.

Uso: .venv\\Scripts\\python ferramentas\\gerar_icone.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIZES = (16, 24, 32, 48, 64, 128, 256)
SUPERSAMPLE = 4


def rounded_box(x, y, cx, cy, half_w, half_h, radius):
    qx = np.abs(x - cx) - (half_w - radius)
    qy = np.abs(y - cy) - (half_h - radius)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - radius


def triangle(x, y, a, b, c):
    """Distância (negativa dentro) aproximada de um triângulo pelos semiplanos."""
    def edge(p, q):
        nx, ny = q[1] - p[1], p[0] - q[0]
        norm = np.hypot(nx, ny)
        return ((x - p[0]) * nx + (y - p[1]) * ny) / norm
    return np.maximum.reduce([edge(a, b), edge(b, c), edge(c, a)])


def coverage(distance, pixel):
    return np.clip(0.5 - distance / pixel, 0.0, 1.0)


def draw(size: int) -> np.ndarray:
    n = size * SUPERSAMPLE
    coords = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(coords, coords)
    pixel = 1.0 / n
    rgba = np.zeros((n, n, 4), dtype=np.float64)

    # fundo: quadrado arredondado com degradê (azul-petróleo → azul)
    background = coverage(rounded_box(x, y, 0.5, 0.5, 0.47, 0.47, 0.2), pixel)
    top, bottom = np.array([0.0, 0.62, 0.66]), np.array([0.09, 0.33, 0.78])
    color = top * (1 - y[..., None]) + bottom * y[..., None]
    rgba[..., :3] = color
    rgba[..., 3] = background

    # balão de fala branco com "rabinho"
    bubble = np.minimum(
        rounded_box(x, y, 0.5, 0.44, 0.33, 0.24, 0.1),
        triangle(x, y, (0.27, 0.6), (0.43, 0.66), (0.24, 0.82)),
    )
    white = coverage(bubble, pixel)

    # ondas de som (barras verticais de alturas diferentes) dentro do balão
    bars = np.full_like(x, np.inf)
    heights = (0.08, 0.16, 0.24, 0.13, 0.2, 0.1, 0.06)
    for i, h in enumerate(heights):
        cx = 0.29 + i * 0.07
        bars = np.minimum(bars, rounded_box(x, y, cx, 0.44, 0.022, h / 2 + 0.02, 0.02))
    wave = coverage(bars, pixel)

    for channel in range(3):
        rgba[..., channel] = rgba[..., channel] * (1 - white) + 1.0 * white
        rgba[..., channel] = rgba[..., channel] * (1 - wave) + color[..., channel] * wave
    # reduz a resolução (média dos subpixels = bordas suaves)
    small = rgba.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE, 4).mean(axis=(1, 3))
    return (np.clip(small, 0, 1) * 255).round().astype(np.uint8)


def png_bytes(image: np.ndarray) -> bytes:
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    body = chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + body


def write_ico(path: Path) -> None:
    images = [png_bytes(draw(size)) for size in SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, payload = b"", b""
    for size, data in zip(SIZES, images, strict=True):
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset + len(payload))
        payload += data
    path.write_bytes(header + entries + payload)
    print(f"{path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    write_ico(ROOT / "transcritor.ico")

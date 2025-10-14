#!/usr/bin/env python3
"""Convert PNG figures into compact PDFs for arXiv submissions.

The script performs an in-place conversion of PNG files to PDF without external
image processing dependencies.  Only 8-bit RGB or RGBA (non-interlaced) PNGs are
supported, which covers the figures used in this repository.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def iter_pngs(root: Path) -> Iterable[Path]:
    """Yield PNG files under ``root`` sorted by descending file size."""
    files = sorted(root.rglob("*.png"), key=lambda p: p.stat().st_size, reverse=True)
    for path in files:
        yield path


class PngImage:
    """Minimal PNG decoder for 8-bit RGB / RGBA images."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.width = 0
        self.height = 0
        self.bit_depth = 0
        self.color_type = 0
        self.compression = 0
        self.filter_method = 0
        self.interlace = 0
        self._idat_chunks: List[bytes] = []
        self._palette: List[Tuple[int, int, int]] | None = None
        self._transparency: bytes | None = None

    def load(self) -> None:
        with self.path.open("rb") as stream:
            signature = stream.read(8)
            if signature != PNG_SIGNATURE:
                raise SystemExit(f"{self.path} is not a PNG file")

            while True:
                length_bytes = stream.read(4)
                if len(length_bytes) == 0:
                    break
                length = struct.unpack(">I", length_bytes)[0]
                chunk_type = stream.read(4)
                data = stream.read(length)
                stream.read(4)  # CRC

                if chunk_type == b"IHDR":
                    self._parse_ihdr(data)
                elif chunk_type == b"IDAT":
                    self._idat_chunks.append(data)
                elif chunk_type == b"PLTE":
                    self._parse_palette(data)
                elif chunk_type == b"tRNS":
                    self._transparency = data
                elif chunk_type == b"IEND":
                    break

        if not self._idat_chunks:
            raise SystemExit(f"{self.path} does not contain any image data")

        if self.bit_depth != 8:
            raise SystemExit(f"{self.path}: unsupported bit depth {self.bit_depth}")
        if self.interlace != 0:
            raise SystemExit(f"{self.path}: interlaced PNGs are not supported")
        if self.color_type not in (2, 3, 6):
            raise SystemExit(f"{self.path}: unsupported color type {self.color_type}")
        if self.color_type == 3 and not self._palette:
            raise SystemExit(f"{self.path}: indexed PNG is missing a PLTE chunk")

    def _parse_ihdr(self, data: bytes) -> None:
        self.width, self.height, self.bit_depth, self.color_type, self.compression, self.filter_method, self.interlace = struct.unpack(
            ">IIBBBBB", data
        )
        if self.compression != 0 or self.filter_method != 0:
            raise SystemExit(f"{self.path}: unsupported PNG compression parameters")

    def _parse_palette(self, data: bytes) -> None:
        if len(data) % 3 != 0:
            raise SystemExit(f"{self.path}: malformed PLTE chunk")
        entries = len(data) // 3
        if entries == 0 or entries > 256:
            raise SystemExit(f"{self.path}: invalid palette size {entries}")
        self._palette = [tuple(data[i : i + 3]) for i in range(0, len(data), 3)]

    def to_rgb(self) -> bytes:
        decompressed = zlib.decompress(b"".join(self._idat_chunks))
        channels = 1 if self.color_type == 3 else 3 if self.color_type == 2 else 4
        row_bytes = self.width * channels
        result = bytearray(self.width * self.height * 3)

        prev_row = bytearray(row_bytes)
        offset = 0
        out_offset = 0

        alpha_table: List[int] | None = None
        if channels == 1:
            assert self._palette is not None
            alpha_table = [255] * len(self._palette)
            if self._transparency:
                for idx, value in enumerate(self._transparency):
                    if idx < len(alpha_table):
                        alpha_table[idx] = value

        for _ in range(self.height):
            filter_type = decompressed[offset]
            offset += 1
            row_data = bytearray(decompressed[offset : offset + row_bytes])
            offset += row_bytes

            apply_filter(filter_type, row_data, prev_row, channels)

            if channels == 3:
                result[out_offset : out_offset + row_bytes] = row_data
                out_offset += row_bytes
            elif channels == 4:
                for px in range(self.width):
                    r, g, b, a = row_data[px * 4 : px * 4 + 4]
                    if a == 255:
                        result[out_offset : out_offset + 3] = bytes((r, g, b))
                    elif a == 0:
                        result[out_offset : out_offset + 3] = b"\xff\xff\xff"
                    else:
                        result[out_offset] = (r * a + 255 * (255 - a) + 127) // 255
                        result[out_offset + 1] = (g * a + 255 * (255 - a) + 127) // 255
                        result[out_offset + 2] = (b * a + 255 * (255 - a) + 127) // 255
                    out_offset += 3
            else:  # Indexed color
                assert self._palette is not None and alpha_table is not None
                for px in range(self.width):
                    index = row_data[px]
                    try:
                        r, g, b = self._palette[index]
                    except IndexError as exc:  # pragma: no cover - defensive
                        raise SystemExit(f"{self.path}: palette index {index} out of range") from exc
                    a = alpha_table[index]
                    if a == 255:
                        result[out_offset : out_offset + 3] = bytes((r, g, b))
                    elif a == 0:
                        result[out_offset : out_offset + 3] = b"\xff\xff\xff"
                    else:
                        result[out_offset] = (r * a + 255 * (255 - a) + 127) // 255
                        result[out_offset + 1] = (g * a + 255 * (255 - a) + 127) // 255
                        result[out_offset + 2] = (b * a + 255 * (255 - a) + 127) // 255
                    out_offset += 3

            prev_row = row_data

        # Apply simple transparency (tRNS) for RGB images if present.
        if channels == 3 and self._transparency:
            transparent_rgb = tuple(self._transparency[:3])
            for px in range(self.width * self.height):
                base = px * 3
                if tuple(result[base : base + 3]) == transparent_rgb:
                    result[base : base + 3] = b"\xff\xff\xff"

        return bytes(result)


def apply_filter(filter_type: int, row: bytearray, prev_row: bytearray, channels: int) -> None:
    """Reverse the PNG scanline filters."""
    bpp = channels

    if filter_type == 0:  # None
        return
    if filter_type == 1:  # Sub
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + left) & 0xFF
        return
    if filter_type == 2:  # Up
        for i in range(len(row)):
            row[i] = (row[i] + prev_row[i]) & 0xFF
        return
    if filter_type == 3:  # Average
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            up = prev_row[i]
            row[i] = (row[i] + ((left + up) // 2)) & 0xFF
        return
    if filter_type == 4:  # Paeth
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            up = prev_row[i]
            up_left = prev_row[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + paeth_predictor(left, up, up_left)) & 0xFF
        return
    raise SystemExit(f"Unsupported PNG filter type {filter_type}")


def paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def make_pdf(rgb: bytes, width: int, height: int) -> bytes:
    """Build a minimal single-page PDF embedding the RGB image."""
    image_stream = zlib.compress(rgb)
    contents_stream = f"q {width} 0 0 {height} 0 0 cm /Im0 Do Q".encode()

    objects: List[bytes] = []

    def add_object(body: str | bytes) -> None:
        if isinstance(body, str):
            body_bytes = body.encode()
        else:
            body_bytes = body
        objects.append(body_bytes)

    add_object("<< /Type /Catalog /Pages 2 0 R >>")
    add_object("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add_object(
        "<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im0 4 0 R >> /ProcSet [/PDF /ImageC] >> /MediaBox [0 0 {0} {1}] /Contents 5 0 R >>".format(
            width, height
        )
    )
    add_object(
        "<< /Type /XObject /Subtype /Image /Width {0} /Height {1} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {2} >>\nstream\n".format(
            width, height, len(image_stream)
        ).encode()
        + image_stream
        + b"\nendstream"
    )
    add_object(
        "<< /Length {0} >>\nstream\n".format(len(contents_stream)).encode()
        + contents_stream
        + b"\nendstream"
    )

    xref_positions: List[int] = []
    output = bytearray(b"%PDF-1.4\n%\xff\xff\xff\xff\n")

    for obj_number, body in enumerate(objects, start=1):
        xref_positions.append(len(output))
        output.extend(f"{obj_number} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_start = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for pos in xref_positions:
        output.extend(f"{pos:010} 00000 n \n".encode())
    output.extend(b"trailer\n")
    output.extend(
        "<< /Size {0} /Root 1 0 R >>\nstartxref\n{1}\n%%EOF".format(len(objects) + 1, xref_start).encode()
    )

    return bytes(output)


def convert_png(path: Path, apply: bool, remove_original: bool) -> Tuple[int, int]:
    png = PngImage(path)
    png.load()
    rgb = png.to_rgb()
    pdf_bytes = make_pdf(rgb, png.width, png.height)
    original_size = path.stat().st_size

    if apply:
        pdf_path = path.with_suffix(".pdf")
        pdf_path.write_bytes(pdf_bytes)
        if remove_original:
            path.unlink()

    return original_size, len(pdf_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific PNG files to convert. If omitted, process every PNG under --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("figures"),
        help="Directory to scan when no explicit paths are provided (default: figures).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the resulting PDFs. Without this flag the script performs a dry run.",
    )
    parser.add_argument(
        "--remove-original",
        action="store_true",
        help="Delete the source PNGs after successful conversion (requires --apply).",
    )
    args = parser.parse_args()

    if args.paths:
        targets: Sequence[Path] = args.paths
    else:
        if not args.root.exists():
            raise SystemExit(f"Directory {args.root} does not exist")
        targets = list(iter_pngs(args.root))

    if args.remove_original and not args.apply:
        raise SystemExit("--remove-original requires --apply")

    any_changes = False
    for path in targets:
        if not path.exists():
            raise SystemExit(f"File {path} does not exist")
        original, converted = convert_png(path, args.apply, args.remove_original)
        any_changes = True
        status = "(dry run)" if not args.apply else ""
        print(f"{path}: {original / 1024:.1f} KiB -> {converted / 1024:.1f} KiB {status}")

    if not any_changes:
        print("No PNG files found to process.")


if __name__ == "__main__":
    main()
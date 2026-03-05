#!/usr/bin/env python3
"""Konwertuje PNG do pliku SVG z zakodowanym data URI."""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path
from typing import Tuple


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _extract_png_size(data: bytes) -> Tuple[int, int]:
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE):
        raise ValueError("Wejściowy plik nie jest prawidłowym PNG")
    width = int.from_bytes(data[16:20], "big", signed=False)
    height = int.from_bytes(data[20:24], "big", signed=False)
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Osadza PNG w SVG, aby obejść ograniczenia na pliki binarne."
    )
    parser.add_argument("png", type=Path, help="Ścieżka do źródłowego pliku PNG")
    parser.add_argument("svg", type=Path, help="Ścieżka docelowa dla SVG")
    parser.add_argument(
        "--title",
        default="Feed SLA overlay",
        help="Tytuł elementu <title> osadzonego w SVG",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Opcjonalny opis w elemencie <desc>",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    png_path: Path = args.png
    svg_path: Path = args.svg

    data = png_path.read_bytes()
    width, height = _extract_png_size(data)
    encoded = base64.b64encode(data).decode("ascii")
    title = html.escape(str(args.title))
    desc = html.escape(str(args.description)) if args.description else None

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg"'
            f' xmlns:xlink="http://www.w3.org/1999/xlink"'
            f' width="{width}" height="{height}"'
            f' viewBox="0 0 {width} {height}">'
        ),
        f"  <title>{title}</title>",
    ]
    if desc:
        svg_lines.append(f"  <desc>{desc}</desc>")
    svg_lines.append(
        (
            f'  <image width="{width}" height="{height}"'
            f' xlink:href="data:image/png;base64,{encoded}" />'
        )
    )
    svg_lines.append("</svg>")

    svg_path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

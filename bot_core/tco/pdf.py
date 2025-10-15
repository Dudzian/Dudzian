"""Minimalistyczny generator PDF używany do raportów TCO."""
from __future__ import annotations

from typing import Iterable


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_simple_pdf(lines: Iterable[str]) -> bytes:
    """Zwraca jednokartkowy dokument PDF zawierający przekazane linie tekstu."""

    sanitized = [line.rstrip() for line in lines]
    text_lines = ["BT", "/F1 11 Tf", "1 0 0 1 72 720 Tm", "14 TL"]
    first = True
    for line in sanitized:
        if first:
            text_lines.append(f"({_escape(line)}) Tj")
            first = False
            continue
        text_lines.append("T*")
        text_lines.append(f"({_escape(line)}) Tj")
    if first:
        text_lines.append("() Tj")
    text_lines.append("ET")
    stream = "\n".join(text_lines) + "\n"

    objects: list[str] = []
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>"
    )
    objects.append(f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}endstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    parts: list[bytes] = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offset = sum(len(part) for part in parts)
        offsets.append(offset)
        parts.append(f"{index} 0 obj\n".encode("ascii"))
        parts.append(obj.encode("utf-8"))
        parts.append(b"\nendobj\n")
    xref_offset = sum(len(part) for part in parts)
    xref_lines = [f"xref\n0 {len(objects) + 1}", "0000000000 65535 f "]
    for offset in offsets[1:]:
        xref_lines.append(f"{offset:010d} 00000 n ")
    xref_lines.append("trailer\n<< /Size {size} /Root 1 0 R >>".format(size=len(objects) + 1))
    xref_lines.append(f"startxref\n{xref_offset}")
    xref_lines.append("%%EOF")
    parts.append("\n".join(xref_lines).encode("ascii"))
    return b"".join(parts)


__all__ = ["build_simple_pdf"]

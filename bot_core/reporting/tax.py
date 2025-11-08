"""Eksport raportów podatkowych do różnych formatów."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
from pathlib import Path
from typing import Mapping, Tuple

from bot_core.compliance.tax import TaxReport

try:  # pragma: no cover - opcjonalna zależność
    import jsonschema
except ModuleNotFoundError:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]


class TaxReportExporter:
    """Zapisuje raport podatkowy oraz generuje podpis HMAC."""

    def __init__(self, *, default_schema: Path | None = None) -> None:
        if default_schema is None:
            default_schema = Path("docs/schemas/tax_report.json")
        self._default_schema = default_schema

    def export(
        self,
        report: TaxReport,
        *,
        path: Path | str,
        fmt: str = "json",
        hmac_key: bytes | str,
        schema: Mapping[str, object] | Path | None = None,
    ) -> Tuple[Path, Path]:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fmt_lower = fmt.lower()
        if fmt_lower == "json":
            payload = report.to_dict()
            schema_data = self._load_schema(schema)
            self._validate_schema(payload, schema_data)
            with output.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        elif fmt_lower == "csv":
            self._write_csv(report, output)
        elif fmt_lower == "pdf":
            self._write_pdf(report, output)
        else:
            raise ValueError(f"Nieobsługiwany format eksportu: {fmt}")
        signature = self._write_signature(output, hmac_key)
        return output, signature

    # --- Formatowanie -----------------------------------------------------------
    def _write_csv(self, report: TaxReport, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["base_currency", report.base_currency or ""])
            writer.writerow([])
            writer.writerow(
                [
                    "event_id",
                    "asset",
                    "disposal_time",
                    "quantity",
                    "proceeds",
                    "fee",
                    "cost_basis",
                    "realized_gain",
                    "short_term_gain",
                    "long_term_gain",
                    "short_term_quantity",
                    "long_term_quantity",
                    "average_holding_period_days",
                    "short_term_tax",
                    "long_term_tax",
                    "total_tax_liability",
                ]
            )
            for event in report.events:
                writer.writerow(
                    [
                        event.event_id,
                        event.asset,
                        event.disposal_time.isoformat(),
                        event.quantity,
                        event.proceeds,
                        event.fee,
                        event.cost_basis,
                        event.realized_gain,
                        event.short_term_gain,
                        event.long_term_gain,
                        event.short_term_quantity,
                        event.long_term_quantity,
                        event.average_holding_period_days,
                        event.short_term_tax,
                        event.long_term_tax,
                        event.total_tax_liability,
                    ]
                )
            writer.writerow([])
            writer.writerow(["asset_breakdown"])
            writer.writerow(
                [
                    "asset",
                    "disposed_quantity",
                    "proceeds",
                    "cost_basis",
                    "fees",
                    "realized_gain",
                    "open_quantity",
                    "open_cost_basis",
                    "short_term_gain",
                    "long_term_gain",
                    "short_term_quantity",
                    "long_term_quantity",
                    "average_holding_period_days",
                    "open_short_term_quantity",
                    "open_long_term_quantity",
                    "open_short_term_cost_basis",
                    "open_long_term_cost_basis",
                    "open_average_holding_period_days",
                    "short_term_tax",
                    "long_term_tax",
                    "total_tax_liability",
                ]
            )
            for entry in report.asset_breakdown:
                writer.writerow(
                    [
                        entry.asset,
                        entry.disposed_quantity,
                        entry.proceeds,
                        entry.cost_basis,
                        entry.fees,
                        entry.realized_gain,
                        entry.open_quantity,
                        entry.open_cost_basis,
                        entry.short_term_gain,
                        entry.long_term_gain,
                        entry.short_term_quantity,
                        entry.long_term_quantity,
                        entry.average_holding_period_days,
                        entry.open_short_term_quantity,
                        entry.open_long_term_quantity,
                        entry.open_short_term_cost_basis,
                        entry.open_long_term_cost_basis,
                        entry.open_average_holding_period_days,
                        entry.short_term_tax,
                        entry.long_term_tax,
                        entry.total_tax_liability,
                    ]
                )
            writer.writerow([])
            writer.writerow(["venue_breakdown"])
            writer.writerow(
                [
                    "venue",
                    "disposed_quantity",
                    "proceeds",
                    "cost_basis",
                    "fees",
                    "realized_gain",
                    "open_quantity",
                    "open_cost_basis",
                    "short_term_gain",
                    "long_term_gain",
                    "short_term_quantity",
                    "long_term_quantity",
                    "average_holding_period_days",
                    "open_short_term_quantity",
                    "open_long_term_quantity",
                    "open_short_term_cost_basis",
                    "open_long_term_cost_basis",
                    "open_average_holding_period_days",
                    "short_term_tax",
                    "long_term_tax",
                    "total_tax_liability",
                ]
            )
            for entry in report.venue_breakdown:
                writer.writerow(
                    [
                        entry.venue or "",
                        entry.disposed_quantity,
                        entry.proceeds,
                        entry.cost_basis,
                        entry.fees,
                        entry.realized_gain,
                        entry.open_quantity,
                        entry.open_cost_basis,
                        entry.short_term_gain,
                        entry.long_term_gain,
                        entry.short_term_quantity,
                        entry.long_term_quantity,
                        entry.average_holding_period_days,
                        entry.open_short_term_quantity,
                        entry.open_long_term_quantity,
                        entry.open_short_term_cost_basis,
                        entry.open_long_term_cost_basis,
                        entry.open_average_holding_period_days,
                        entry.short_term_tax,
                        entry.long_term_tax,
                        entry.total_tax_liability,
                    ]
                )
            writer.writerow([])
            writer.writerow(["period_breakdown"])
            writer.writerow(
                [
                    "period",
                    "period_start",
                    "period_end",
                    "disposed_quantity",
                    "proceeds",
                    "cost_basis",
                    "fees",
                    "realized_gain",
                    "short_term_gain",
                    "long_term_gain",
                    "short_term_quantity",
                    "long_term_quantity",
                    "average_holding_period_days",
                    "short_term_tax",
                    "long_term_tax",
                    "total_tax_liability",
                ]
            )
            for entry in report.period_breakdown:
                writer.writerow(
                    [
                        entry.period,
                        entry.period_start.isoformat(),
                        entry.period_end.isoformat(),
                        entry.disposed_quantity,
                        entry.proceeds,
                        entry.cost_basis,
                        entry.fees,
                        entry.realized_gain,
                        entry.short_term_gain,
                        entry.long_term_gain,
                        entry.short_term_quantity,
                        entry.long_term_quantity,
                        entry.average_holding_period_days,
                        entry.short_term_tax,
                        entry.long_term_tax,
                        entry.total_tax_liability,
                    ]
                )

    def _write_pdf(self, report: TaxReport, path: Path) -> None:
        content_lines = [
            f"Jurysdykcja: {report.jurisdiction}",
            f"Metoda: {report.method}",
            f"Waluta bazowa: {report.base_currency or '-'}",
            f"Liczba zdarzeń: {len(report.events)}",
            f"Zrealizowany wynik: {report.totals.realized_gain:.2f}",
            f"Zysk krótkoterminowy: {report.totals.short_term_gain:.2f}",
            f"Zysk długoterminowy: {report.totals.long_term_gain:.2f}",
            f"Podatek krótki: {report.totals.short_term_tax:.2f}",
            f"Podatek długi: {report.totals.long_term_tax:.2f}",
            f"Łączne zobowiązanie: {report.totals.total_tax_liability:.2f}",
            (
                "Śr. okres przetrzymania (dni): "
                f"{report.totals.average_holding_period_days:.2f}"
            ),
            f"Otwarte pozycje: {len(report.open_lots)}",
            (
                "Otwarte qty short: "
                f"{report.totals.unrealized_short_term_quantity:.8f}, cost={report.totals.unrealized_short_term_cost_basis:.2f}"
            ),
            (
                "Otwarte qty long: "
                f"{report.totals.unrealized_long_term_quantity:.8f}, cost={report.totals.unrealized_long_term_cost_basis:.2f}"
            ),
            (
                "Śr. okres otwartych (dni): "
                f"{report.totals.average_open_holding_period_days:.2f}"
            ),
        ]
        for event in report.events:
            content_lines.append(
                f"{event.disposal_time.date()} {event.asset} qty={event.quantity} gain={event.realized_gain:.2f}"
            )
            content_lines.append(
                (
                    f"    krótkoterminowo qty={event.short_term_quantity:.8f}, "
                    f"długoterminowo qty={event.long_term_quantity:.8f}, "
                    f"średni okres={event.average_holding_period_days:.2f} dni, "
                    f"podatek short={event.short_term_tax:.2f}, long={event.long_term_tax:.2f}"
                )
            )
        if report.asset_breakdown:
            content_lines.append("Podsumowanie aktywów:")
            for entry in report.asset_breakdown:
                content_lines.append(
                    (
                        f"{entry.asset}: sprzedano {entry.disposed_quantity:.8f}, zysk "
                        f"{entry.realized_gain:.2f}, krótkoterminowy {entry.short_term_gain:.2f}, "
                        f"długoterminowy {entry.long_term_gain:.2f}, otwarte {entry.open_quantity:.8f}, "
                        f"podatek short {entry.short_term_tax:.2f}, long {entry.long_term_tax:.2f}"
                    )
                )
                content_lines.append(
                    (
                        "    ilość short "
                        f"{entry.short_term_quantity:.8f}, ilość long {entry.long_term_quantity:.8f}, "
                        f"średni okres={entry.average_holding_period_days:.2f} dni"
                    )
                )
                content_lines.append(
                    (
                        "    otwarte short qty="
                        f"{entry.open_short_term_quantity:.8f}, cost={entry.open_short_term_cost_basis:.2f}; "
                        "long qty="
                        f"{entry.open_long_term_quantity:.8f}, cost={entry.open_long_term_cost_basis:.2f}; "
                        f"średni okres otwartych={entry.open_average_holding_period_days:.2f} dni, "
                        f"łączny podatek={entry.total_tax_liability:.2f}"
                    )
                )
        if report.venue_breakdown:
            content_lines.append("Podsumowanie wg giełdy:")
            for entry in report.venue_breakdown:
                venue_label = entry.venue or "-"
                content_lines.append(
                    (
                        f"{venue_label}: sprzedano {entry.disposed_quantity:.8f}, zysk {entry.realized_gain:.2f}, "
                        f"opłaty {entry.fees:.2f}, otwarte {entry.open_quantity:.8f}, "
                        f"podatek={entry.total_tax_liability:.2f}"
                    )
                )
                content_lines.append(
                    (
                        "    qty short="
                        f"{entry.short_term_quantity:.8f}, qty long {entry.long_term_quantity:.8f}, "
                        f"średni okres={entry.average_holding_period_days:.2f} dni"
                    )
                )
        if report.period_breakdown:
            content_lines.append("Podsumowanie okresów:")
            for entry in report.period_breakdown:
                content_lines.append(
                    (
                        f"{entry.period}: przychody {entry.proceeds:.2f}, koszt {entry.cost_basis:.2f}, "
                        f"zysk {entry.realized_gain:.2f}, podatek {entry.total_tax_liability:.2f}"
                    )
                )
                content_lines.append(
                    (
                        "    krótkoterminowo zysk="
                        f"{entry.short_term_gain:.2f}, długoterminowo="
                        f"{entry.long_term_gain:.2f}, ilość={entry.disposed_quantity:.8f}, "
                        f"średni okres={entry.average_holding_period_days:.2f} dni, "
                        f"podatek short={entry.short_term_tax:.2f}, long={entry.long_term_tax:.2f}"
                    )
                )
        pdf_bytes = self._build_simple_pdf(content_lines)
        with path.open("wb") as handle:
            handle.write(pdf_bytes)

    def _build_simple_pdf(self, lines: list[str]) -> bytes:
        import io

        buffer = io.BytesIO()
        buffer.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []

        def write_obj(index: int, payload: bytes) -> None:
            offsets.append(buffer.tell())
            buffer.write(f"{index} 0 obj\n".encode("ascii"))
            buffer.write(payload)
            buffer.write(b"\nendobj\n")

        write_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        write_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        write_obj(
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        )
        content_lines = ["BT", "/F1 12 Tf", "14 TL", "72 770 Td"]
        for idx, line in enumerate(lines):
            text = line.replace("(", r"\(").replace(")", r"\)")
            if idx == 0:
                content_lines.append(f"({text}) Tj")
            else:
                content_lines.append("T*")
                content_lines.append(f"({text}) Tj")
        content_lines.append("ET")
        content_stream = "\n".join(content_lines).encode("utf-8")
        offsets.append(buffer.tell())
        buffer.write(b"4 0 obj\n")
        buffer.write(f"<< /Length {len(content_stream)} >>\n".encode("ascii"))
        buffer.write(b"stream\n")
        buffer.write(content_stream)
        buffer.write(b"\nendstream\nendobj\n")
        write_obj(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        xref_pos = buffer.tell()
        buffer.write(b"xref\n0 6\n0000000000 65535 f \n")
        for off in offsets:
            buffer.write(f"{off:010d} 00000 n \n".encode("ascii"))
        buffer.write(b"trailer\n<< /Size 6 /Root 1 0 R >>\n")
        buffer.write(b"startxref\n")
        buffer.write(f"{xref_pos}\n".encode("ascii"))
        buffer.write(b"%%EOF\n")
        return buffer.getvalue()

    # --- Walidacja i podpis -----------------------------------------------------
    def _load_schema(self, candidate: Mapping[str, object] | Path | None) -> Mapping[str, object] | None:
        if candidate is None:
            path = self._default_schema
            if path and path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            return None
        if isinstance(candidate, Mapping):
            return candidate
        path = Path(candidate)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _validate_schema(self, payload: Mapping[str, object], schema: Mapping[str, object] | None) -> None:
        if schema is None:
            return
        if jsonschema is None:
            raise RuntimeError("Walidacja schematu wymaga biblioteki jsonschema")
        jsonschema.validate(instance=payload, schema=schema)

    def _write_signature(self, output: Path, key: bytes | str) -> Path:
        if isinstance(key, str):
            key_bytes = key.encode("utf-8")
        else:
            key_bytes = key
        data = output.read_bytes()
        digest = hmac.new(key_bytes, data, hashlib.sha256).hexdigest()
        signature_path = output.with_suffix(output.suffix + ".sig")
        with signature_path.open("w", encoding="utf-8") as handle:
            handle.write(digest)
        return signature_path

"""Generowanie i podpisywanie raportów TCO."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

from bot_core.security.signing import build_hmac_signature

from .models import ProfileCostSummary, StrategyCostSummary, TCOReport
from .pdf import build_simple_pdf

_DECIMAL_QUANT = Decimal("0.000001")


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(_DECIMAL_QUANT, rounding=ROUND_HALF_UP)


def _format_decimal(value: Decimal) -> str:
    return f"{_quantize(value):f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class SignedArtifact:
    path: Path
    signature_path: Path
    payload: Mapping[str, object]


class TCOReportWriter:
    """Buduje artefakty raportu TCO (CSV, PDF, JSON)."""

    def __init__(self, report: TCOReport) -> None:
        self._report = report

    def build_csv(self) -> str:
        header = [
            "strategy",
            "profile",
            "trade_count",
            "notional",
            "total_cost",
            "commission",
            "slippage",
            "funding",
            "other",
            "cost_per_trade",
            "cost_bps",
        ]
        rows: list[list[str]] = [header]

        for strategy, summary in sorted(self._report.strategies.items()):
            rows.extend(self._rows_for_strategy(strategy, summary))
        rows.append(self._build_row("__TOTAL__", "all", self._report.total))

        output_lines: list[str] = []
        for row in rows:
            output_lines.append(
                ",".join(row)
            )
        output_lines.append("")
        return "\n".join(output_lines)

    def _rows_for_strategy(
        self,
        strategy: str,
        summary: StrategyCostSummary,
    ) -> Iterable[list[str]]:
        for profile_name, profile_summary in sorted(summary.profiles.items()):
            yield self._build_row(strategy, profile_name, profile_summary)
        yield self._build_row(strategy, "all", summary.total)

    def _build_row(
        self,
        strategy: str,
        profile: str,
        summary: ProfileCostSummary,
    ) -> list[str]:
        return [
            strategy,
            profile,
            str(summary.trade_count),
            _format_decimal(summary.notional),
            _format_decimal(summary.total_cost),
            _format_decimal(summary.breakdown.commission),
            _format_decimal(summary.breakdown.slippage),
            _format_decimal(summary.breakdown.funding),
            _format_decimal(summary.breakdown.other),
            _format_decimal(summary.cost_per_trade),
            _format_decimal(summary.cost_bps),
        ]

    def build_pdf(self) -> bytes:
        metadata = self._report.metadata
        lines = [
            "Raport kosztów transakcyjnych (TCO)",
            f"Data generacji: {self._report.generated_at.isoformat()}",
            "",  # odstęp
            "Statystyki:",
            f"  Liczba strategii: {metadata.get('strategy_count', len(self._report.strategies))}",
            f"  Liczba schedulerów: {metadata.get('scheduler_count', len(self._report.schedulers))}",
            f"  Liczba transakcji: {metadata.get('events_count', 0)}",
        ]
        alerts = list(self._report.alerts)
        if alerts:
            lines.append("")
            lines.append("Alerty TCO:")
            for alert in alerts:
                lines.append(f"  - {alert}")
        for strategy, summary in sorted(self._report.strategies.items()):
            lines.append("")
            lines.append(
                (
                    "Strategia {name}: {trades} transakcji, koszty {cost:.6f} "
                    "({bps:.2f} bps)"
                ).format(
                    name=strategy,
                    trades=summary.total.trade_count,
                    cost=float(summary.total.total_cost),
                    bps=float(summary.total.cost_bps),
                )
            )
            for profile, profile_summary in sorted(summary.profiles.items()):
                lines.append(
                    (
                        "  Profil {profile}: transakcji {trades}, koszt {cost:.6f}, {bps:.2f} bps"
                    ).format(
                        profile=profile,
                        trades=profile_summary.trade_count,
                        cost=float(profile_summary.total_cost),
                        bps=float(profile_summary.cost_bps),
                    )
                )
        if not self._report.strategies:
            lines.append("")
            lines.append("Brak danych transakcyjnych do zaprezentowania.")
        if self._report.schedulers:
            lines.append("")
            lines.append("Zestawienie według schedulerów:")
            for scheduler_name, summary in sorted(self._report.schedulers.items()):
                lines.append(
                    (
                        "Scheduler {name}: transakcji {trades}, koszt {cost:.6f}, {bps:.2f} bps"
                    ).format(
                        name=scheduler_name,
                        trades=summary.total.trade_count,
                        cost=float(summary.total.total_cost),
                        bps=float(summary.total.cost_bps),
                    )
                )
                for strategy_name, strategy_summary in sorted(summary.strategies.items()):
                    lines.append(
                        (
                            "  Strategia {strategy}: transakcji {trades}, koszt {cost:.6f}, {bps:.2f} bps"
                        ).format(
                            strategy=strategy_name,
                            trades=strategy_summary.trade_count,
                            cost=float(strategy_summary.total_cost),
                            bps=float(strategy_summary.cost_bps),
                        )
                    )
        return build_simple_pdf(lines)

    def build_json(self) -> dict[str, object]:
        return self._report.to_dict()

    def write_outputs(
        self,
        output_dir: Path,
        *,
        basename: str | None = None,
    ) -> Mapping[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if basename is None:
            basename = f"tco_report_{self._report.generated_at.strftime('%Y%m%dT%H%M%SZ')}"
        csv_path = output_dir / f"{basename}.csv"
        pdf_path = output_dir / f"{basename}.pdf"
        json_path = output_dir / f"{basename}.json"

        csv_path.write_text(self.build_csv(), encoding="utf-8")
        pdf_path.write_bytes(self.build_pdf())
        json_payload = self.build_json()
        json_path.write_text(
            json.dumps(json_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"csv": csv_path, "pdf": pdf_path, "json": json_path}

    def sign_artifacts(
        self,
        artifacts: Mapping[str, Path],
        *,
        signing_key: bytes,
        key_id: str | None = None,
    ) -> Mapping[str, SignedArtifact]:
        if len(signing_key) < 32:
            raise ValueError("Klucz podpisu musi mieć co najmniej 32 bajty")
        signed: MutableMapping[str, SignedArtifact] = {}
        for label, path in artifacts.items():
            payload = {
                "artifact": path.name,
                "artifact_type": label,
                "generated_at": self._report.generated_at.isoformat(),
                "sha256": _sha256(path),
                "events_count": self._report.metadata.get("events_count", 0),
                "strategy_count": self._report.metadata.get(
                    "strategy_count", len(self._report.strategies)
                ),
                "scheduler_count": self._report.metadata.get(
                    "scheduler_count", len(self._report.schedulers)
                ),
            }
            signature = build_hmac_signature(
                payload,
                key=signing_key,
                algorithm="HMAC-SHA256",
                key_id=key_id,
            )
            document = {"payload": payload, "signature": signature}
            signature_path = path.with_suffix(path.suffix + ".sig")
            signature_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            signed[label] = SignedArtifact(path=path, signature_path=signature_path, payload=payload)
        return signed


__all__ = ["SignedArtifact", "TCOReportWriter"]

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

from bot_core.tco import TCOAnalyzer, TCOReportWriter, TradeCostEvent

_LOGGER = logging.getLogger(__name__)


def _as_decimal(value: float | int | Decimal | None) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _normalize_formats(formats: Sequence[str] | None) -> tuple[str, ...]:
    if not formats:
        return ("json",)
    normalized: list[str] = []
    for entry in formats:
        text = str(entry).strip().lower()
        if not text:
            continue
        if text not in {"json", "csv", "pdf"}:
            raise ValueError(
                "Unsupported TCO export format '{}'. Expected one of: json, csv, pdf".format(entry)
            )
        normalized.append(text)
    return tuple(dict.fromkeys(normalized)) or ("json",)


def _calculate_slippage(
    *,
    side: str,
    quantity: float,
    executed_price: float | None,
    reference_price: float | None,
) -> float:
    if executed_price is None or reference_price is None:
        return 0.0
    qty = max(0.0, float(quantity))
    if qty == 0:
        return 0.0
    executed = float(executed_price)
    reference = float(reference_price)
    if side.lower() == "sell":
        diff = reference - executed
    else:
        diff = executed - reference
    return max(0.0, diff) * qty


@dataclass(slots=True)
class RuntimeTCOReporter:
    """Collects trade cost events during runtime and produces signed reports."""

    output_dir: str | Path | None = None
    basename: str | None = None
    export_formats: Sequence[str] | None = None
    flush_events: int | None = None
    clear_after_export: bool = False
    signing_key: bytes | None = None
    signing_key_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    cost_limit_bps: float | Decimal | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    _analyzer: TCOAnalyzer = field(init=False, repr=False)
    _events: list[TradeCostEvent] = field(init=False, repr=False)
    _output_dir: Path | None = field(init=False, repr=False)
    _formats: tuple[str, ...] = field(init=False, repr=False)
    _flush_every: int | None = field(init=False, repr=False)
    _metadata: MutableMapping[str, object] = field(init=False, repr=False)
    _basename: str | None = field(init=False, repr=False)
    _signing_key: bytes | None = field(init=False, repr=False)
    _signing_key_id: str | None = field(init=False, repr=False)
    _last_export: MutableMapping[str, Path] | None = field(init=False, repr=False)
    _clear_after_export: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cost_limit = self.cost_limit_bps
        if cost_limit is not None and not isinstance(cost_limit, Decimal):
            cost_limit = Decimal(str(cost_limit))
        self._analyzer = TCOAnalyzer(cost_limit_bps=cost_limit)
        self._events: list[TradeCostEvent] = []
        self._output_dir = Path(self.output_dir).expanduser() if self.output_dir else None
        self._formats = _normalize_formats(self.export_formats)
        self._flush_every = self.flush_events if self.flush_events and self.flush_events > 0 else None
        self._metadata = dict(self.metadata)
        self._basename = self.basename
        self._signing_key = self.signing_key
        self._signing_key_id = self.signing_key_id
        self._last_export: MutableMapping[str, Path] | None = None
        self._clear_after_export = bool(self.clear_after_export)

    def record_execution(
        self,
        *,
        strategy: str,
        risk_profile: str,
        instrument: str,
        exchange: str,
        side: str,
        quantity: float,
        executed_price: float | None,
        reference_price: float | None,
        commission: float | None = None,
        slippage: float | None = None,
        funding: float | None = None,
        other: float | None = None,
        metadata: Mapping[str, object] | None = None,
        timestamp: datetime | None = None,
    ) -> TradeCostEvent:
        """Registers a fill and optionally exports aggregated artifacts."""

        event = TradeCostEvent(
            timestamp=timestamp or self.clock(),
            strategy=str(strategy),
            risk_profile=str(risk_profile),
            instrument=str(instrument),
            exchange=str(exchange),
            side=str(side).lower(),
            quantity=_as_decimal(quantity),
            price=_as_decimal(executed_price or reference_price or 0.0),
            commission=_as_decimal(commission),
            slippage=_as_decimal(
                slippage
                if slippage is not None
                else _calculate_slippage(
                    side=side,
                    quantity=quantity,
                    executed_price=executed_price,
                    reference_price=reference_price,
                )
            ),
            funding=_as_decimal(funding),
            other=_as_decimal(other),
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        if self._flush_every and self._output_dir is not None:
            if len(self._events) % self._flush_every == 0:
                try:
                    self.export(clear_events=self._clear_after_export)
                except Exception:  # pragma: no cover - export errors should not block trading
                    _LOGGER.exception("Failed to export runtime TCO report")
        return event

    def export(self, *, clear_events: bool | None = None) -> Mapping[str, Path] | None:
        """Aggregates collected events and writes artifacts to disk."""

        if self._output_dir is None:
            return None
        report = self._analyzer.analyze(self._events, metadata=self._metadata)
        writer = TCOReportWriter(report)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        base = self._basename or f"runtime_tco_{report.generated_at.strftime('%Y%m%dT%H%M%SZ')}"
        artifacts: MutableMapping[str, Path] = {}
        if "csv" in self._formats:
            csv_path = self._output_dir / f"{base}.csv"
            csv_path.write_text(writer.build_csv(), encoding="utf-8")
            artifacts["csv"] = csv_path
        if "pdf" in self._formats:
            pdf_path = self._output_dir / f"{base}.pdf"
            pdf_path.write_bytes(writer.build_pdf())
            artifacts["pdf"] = pdf_path
        if "json" in self._formats:
            json_path = self._output_dir / f"{base}.json"
            json_payload = writer.build_json()
            json_path.write_text(
                json.dumps(json_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifacts["json"] = json_path
        if self._signing_key and artifacts:
            try:
                signed = writer.sign_artifacts(artifacts, signing_key=self._signing_key, key_id=self._signing_key_id)
            except Exception:  # pragma: no cover - signing errors should be visible but non-fatal
                _LOGGER.exception("Failed to sign runtime TCO artifacts")
            else:
                for label, artifact in signed.items():
                    artifacts[f"{label}_signature"] = artifact.signature_path
        self._last_export = dict(artifacts)
        should_clear = self._clear_after_export if clear_events is None else bool(clear_events)
        if should_clear:
            self._events.clear()
        return dict(artifacts)

    def events(self) -> tuple[TradeCostEvent, ...]:
        """Returns collected cost events."""

        return tuple(self._events)

    def last_export(self) -> Mapping[str, Path] | None:
        """Returns the mapping of artifacts produced during the last export."""

        return dict(self._last_export) if self._last_export else None

    def clear_events(self) -> None:
        """Usuwa wszystkie zapisane zdarzenia kosztowe (np. po eksporcie)."""

        self._events.clear()

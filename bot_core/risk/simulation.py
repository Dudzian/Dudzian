"""Narzędzia do symulacji scenariuszy ryzyka z użyciem danych Parquet."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.base import RiskEngine, RiskProfile
from bot_core.risk.profiles import (
    AggressiveProfile,
    BalancedProfile,
    ConservativeProfile,
    ManualProfile,
)

__all__ = [
    "DEFAULT_PROFILES",
    "SimulationOrder",
    "RiskScenarioResult",
    "RiskSimulationSuite",
    "DEFAULT_SMOKE_SCENARIOS",
    "write_default_smoke_scenarios",
    "build_profile",
    "load_orders_from_parquet",
    "run_profile_scenario",
]

DEFAULT_PROFILES: Sequence[str] = ("conservative", "balanced", "aggressive", "manual")

_SMOKE_BASE_TIMESTAMP = datetime(2024, 1, 1, 12, 0, 0)
DEFAULT_SMOKE_SCENARIOS: tuple[Mapping[str, object], ...] = (
    {
        "profile": "conservative",
        "timestamp": _SMOKE_BASE_TIMESTAMP.isoformat(),
        "symbol": "BTCUSDT",
        "side": "buy",
        "price": 10000.0,
        "quantity": 0.01,
        "total_equity": 10000.0,
        "available_margin": 8000.0,
        "maintenance_margin": 100.0,
        "atr": 150.0,
        "stop_price": 9850.0,
        "position_value": 100.0,
        "pnl": 25.0,
    },
    {
        "profile": "conservative",
        "timestamp": (_SMOKE_BASE_TIMESTAMP + timedelta(minutes=1)).isoformat(),
        "symbol": "BTCUSDT",
        "side": "buy",
        "price": 10000.0,
        "quantity": 0.1,
        "total_equity": 10000.0,
        "available_margin": 8000.0,
        "maintenance_margin": 100.0,
        "atr": 150.0,
        "stop_price": 9850.0,
        "position_value": 1000.0,
        "pnl": -10.0,
    },
    {
        "profile": "balanced",
        "timestamp": (_SMOKE_BASE_TIMESTAMP + timedelta(minutes=2)).isoformat(),
        "symbol": "ETHUSDT",
        "side": "buy",
        "price": 100.0,
        "quantity": 1.0,
        "total_equity": 20000.0,
        "available_margin": 15000.0,
        "maintenance_margin": 100.0,
        "atr": 5.0,
        "stop_price": 98.0,
        "position_value": 100.0,
        "pnl": 0.0,
    },
    {
        "profile": "aggressive",
        "timestamp": (_SMOKE_BASE_TIMESTAMP + timedelta(minutes=3)).isoformat(),
        "symbol": "SOLUSDT",
        "side": "buy",
        "price": 150.0,
        "quantity": 0.5,
        "total_equity": 5000.0,
        "available_margin": 4000.0,
        "maintenance_margin": 100.0,
        "atr": 8.0,
        "stop_price": 134.0,
        "position_value": 75.0,
        "pnl": 12.0,
    },
    {
        "profile": "manual",
        "timestamp": (_SMOKE_BASE_TIMESTAMP + timedelta(minutes=4)).isoformat(),
        "symbol": "ADAUSDT",
        "side": "buy",
        "price": 50.0,
        "quantity": 2.0,
        "total_equity": 10000.0,
        "available_margin": 9000.0,
        "maintenance_margin": 150.0,
        "atr": 4.0,
        "stop_price": 45.0,
        "position_value": 100.0,
        "pnl": 5.0,
    },
)


def write_default_smoke_scenarios(path: str | Path) -> Path:
    """Zapisuje domyślne scenariusze smoke testu do pliku Parquet."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in DEFAULT_SMOKE_SCENARIOS])
    pq.write_table(table, target)
    return target


@dataclass(slots=True)
class SimulationOrder:
    """Pojedynczy rekord wejściowy używany podczas symulacji."""

    profile: str
    timestamp: datetime
    symbol: str
    side: str
    price: float
    quantity: float
    total_equity: float
    available_margin: float
    maintenance_margin: float
    atr: float | None = None
    stop_price: float | None = None
    position_value: float | None = None
    pnl: float | None = None

    def to_order_request(self) -> OrderRequest:
        return OrderRequest(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type="limit",
            price=self.price,
            stop_price=self.stop_price,
            atr=self.atr,
            metadata={"source": "risk_simulation"},
        )

    def to_account_snapshot(self) -> AccountSnapshot:
        balances: MutableMapping[str, float] = {"USD": float(self.total_equity)}
        return AccountSnapshot(
            balances=balances,
            total_equity=float(self.total_equity),
            available_margin=float(self.available_margin),
            maintenance_margin=float(self.maintenance_margin),
        )

    def to_fill_arguments(self) -> Mapping[str, object]:
        position_value = (
            float(self.position_value)
            if self.position_value is not None
            else abs(float(self.quantity)) * float(self.price)
        )
        return {
            "symbol": self.symbol,
            "side": self.side,
            "position_value": position_value,
            "pnl": float(self.pnl or 0.0),
            "timestamp": self.timestamp,
        }

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "profile": self.profile,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "price": float(self.price),
            "quantity": float(self.quantity),
            "total_equity": float(self.total_equity),
            "available_margin": float(self.available_margin),
            "maintenance_margin": float(self.maintenance_margin),
            "atr": None if self.atr is None else float(self.atr),
            "stop_price": None if self.stop_price is None else float(self.stop_price),
            "position_value": (
                None if self.position_value is None else float(self.position_value)
            ),
            "pnl": None if self.pnl is None else float(self.pnl),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SimulationOrder":
        timestamp_raw = data.get("timestamp")
        timestamp = _parse_timestamp(timestamp_raw)
        return cls(
            profile=str(data["profile"]).lower(),
            timestamp=timestamp,
            symbol=str(data["symbol"]),
            side=str(data["side"]),
            price=float(data["price"]),
            quantity=float(data["quantity"]),
            total_equity=float(data["total_equity"]),
            available_margin=float(data["available_margin"]),
            maintenance_margin=float(data["maintenance_margin"]),
            atr=_coerce_optional_float(data.get("atr")),
            stop_price=_coerce_optional_float(data.get("stop_price")),
            position_value=_coerce_optional_float(data.get("position_value")),
            pnl=_coerce_optional_float(data.get("pnl")),
        )


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(float(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp field cannot be empty")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def _coerce_optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_PROFILE_FACTORY = {
    "conservative": ConservativeProfile,
    "balanced": BalancedProfile,
    "aggressive": AggressiveProfile,
}


def build_profile(profile_name: str, *, manual_overrides: Mapping[str, object] | None = None) -> RiskProfile:
    normalized = profile_name.strip().lower()
    if normalized == "manual":
        if not manual_overrides:
            raise ValueError("Manual profile requires overrides with explicit limits")
        required = {
            "max_positions",
            "max_leverage",
            "drawdown_limit",
            "daily_loss_limit",
            "max_position_pct",
            "target_volatility",
            "stop_loss_atr_multiple",
        }
        missing = [key for key in required if key not in manual_overrides]
        if missing:
            raise ValueError(f"Missing manual profile overrides: {', '.join(missing)}")
        return ManualProfile(
            name=str(manual_overrides.get("name", "manual")),
            max_positions=int(manual_overrides["max_positions"]),
            max_leverage=float(manual_overrides["max_leverage"]),
            drawdown_limit=float(manual_overrides["drawdown_limit"]),
            daily_loss_limit=float(manual_overrides["daily_loss_limit"]),
            max_position_pct=float(manual_overrides["max_position_pct"]),
            target_volatility=float(manual_overrides["target_volatility"]),
            stop_loss_atr_multiple=float(manual_overrides["stop_loss_atr_multiple"]),
        )
    try:
        factory = _PROFILE_FACTORY[normalized]
    except KeyError as exc:  # pragma: no cover - defensywnie
        raise KeyError(f"Unsupported risk profile: {profile_name}") from exc
    return factory()


def load_orders_from_parquet(path: str | Path) -> Sequence[SimulationOrder]:
    table = pq.read_table(path)
    records = table.to_pylist()
    orders = [SimulationOrder.from_mapping(record) for record in records]
    orders.sort(key=lambda order: order.timestamp)
    return tuple(orders)


@dataclass(slots=True)
class RiskScenarioResult:
    """Zbiorczy wynik symulacji pojedynczego profilu."""

    profile: str
    total_orders: int
    accepted_orders: int
    rejected_orders: int
    force_liquidation: bool
    rejection_reasons: Sequence[str]
    decisions: Sequence[Mapping[str, object]]
    final_state: Mapping[str, object] | None

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "profile": self.profile,
            "total_orders": self.total_orders,
            "accepted_orders": self.accepted_orders,
            "rejected_orders": self.rejected_orders,
            "force_liquidation": self.force_liquidation,
            "rejection_reasons": list(self.rejection_reasons),
            "decisions": list(self.decisions),
            "final_state": dict(self.final_state) if self.final_state is not None else None,
        }


@dataclass(slots=True)
class RiskSimulationSuite:
    """Raport zbiorczy obejmujący wszystkie uruchomione scenariusze."""

    scenarios: Sequence[RiskScenarioResult]
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "scenarios": [scenario.to_mapping() for scenario in self.scenarios],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_mapping(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        target = Path(path)
        target.write_text(self.to_json(indent=indent), encoding="utf-8")
        return target

    def render_pdf(self, path: str | Path) -> Path:
        lines: list[str] = [
            "Paper Labs – Risk Simulation Report",
            f"Generated at: {self.generated_at.isoformat()}",
            "",
        ]
        for scenario in self.scenarios:
            lines.extend(
                [
                    f"Profile: {scenario.profile}",
                    f"Total orders: {scenario.total_orders}",
                    f"Accepted: {scenario.accepted_orders}",
                    f"Rejected: {scenario.rejected_orders}",
                    f"Forced liquidation: {'yes' if scenario.force_liquidation else 'no'}",
                ]
            )
            if scenario.rejection_reasons:
                lines.append("Rejection reasons:")
                for reason in scenario.rejection_reasons:
                    lines.append(f" • {reason}")
            lines.append("")
        _write_simple_pdf(path, lines)
        return Path(path)


def run_profile_scenario(
    engine: RiskEngine,
    profile: RiskProfile,
    orders: Iterable[SimulationOrder],
) -> RiskScenarioResult:
    engine.register_profile(profile)
    accepted = 0
    rejected = 0
    rejection_reasons: list[str] = []
    decisions: list[Mapping[str, object]] = []
    for order in orders:
        if order.profile != profile.name:
            continue
        snapshot = order.to_account_snapshot()
        request = order.to_order_request()
        result = engine.apply_pre_trade_checks(
            request,
            account=snapshot,
            profile_name=profile.name,
        )
        decision_payload: MutableMapping[str, object] = {
            "order": order.to_mapping(),
            "allowed": result.allowed,
            "reason": result.reason,
        }
        if result.adjustments is not None:
            decision_payload["adjustments"] = dict(result.adjustments)
        decisions.append(decision_payload)
        if result.allowed:
            accepted += 1
            fill_args = order.to_fill_arguments()
            engine.on_fill(
                profile_name=profile.name,
                symbol=str(fill_args["symbol"]),
                side=str(fill_args["side"]),
                position_value=float(fill_args["position_value"]),
                pnl=float(fill_args["pnl"]),
                timestamp=fill_args["timestamp"],
            )
        else:
            rejected += 1
            if result.reason:
                rejection_reasons.append(result.reason)
    state = engine.snapshot_state(profile.name)
    force_liquidation = False
    try:
        force_liquidation = engine.should_liquidate(profile_name=profile.name)
    except KeyError:  # pragma: no cover - defensywnie
        force_liquidation = False
    return RiskScenarioResult(
        profile=profile.name,
        total_orders=accepted + rejected,
        accepted_orders=accepted,
        rejected_orders=rejected,
        force_liquidation=force_liquidation,
        rejection_reasons=tuple(rejection_reasons),
        decisions=tuple(decisions),
        final_state=state,
    )


def _write_simple_pdf(path: str | Path, lines: Sequence[str]) -> None:
    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    text_commands = ["BT", "/F1 12 Tf", "50 780 Td"]
    first = True
    for line in lines:
        if first:
            text_commands.append(f"({escape(line)}) Tj")
            first = False
        else:
            text_commands.append("T*")
            text_commands.append(f"({escape(line)}) Tj")
    text_commands.append("ET")
    content = "\n".join(text_commands).encode("latin-1", "ignore")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
        b"4 0 obj << /Length "
        + str(len(content)).encode("ascii")
        + b" >> stream\n"
        + content
        + b"\nendstream\nendobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    pdf_parts: list[bytes] = [b"%PDF-1.4\n"]
    offsets: list[int] = []
    for obj in objects:
        current_offset = sum(len(part) for part in pdf_parts)
        offsets.append(current_offset)
        pdf_parts.append(obj)
    body = b"".join(pdf_parts)
    xref_offset = len(body)
    xref_lines = ["xref", "0 6", "0000000000 65535 f "]
    for offset in offsets:
        xref_lines.append(f"{offset:010d} 00000 n ")
    xref = "\n".join(xref_lines).encode("ascii") + b"\n"
    trailer = (
        b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )
    target = Path(path)
    target.write_bytes(body + xref + trailer)

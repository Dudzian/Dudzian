"""Dry-run walidujący konfigurację egzekucji live bez wysyłania zleceń."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from bot_core.auto_trader.app import AutoTrader  # noqa: E402
from bot_core.config.loader import load_core_config  # noqa: E402
from bot_core.config.models import CoreConfig, EnvironmentConfig  # noqa: E402
from bot_core.execution.base import ExecutionContext  # noqa: E402
from bot_core.execution.paper import (  # noqa: E402
    MarketMetadata,
    PaperTradingExecutionService,
)
from bot_core.exchanges.base import (  # noqa: E402
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
)
from bot_core.runtime.bootstrap import get_registered_adapter_factories  # noqa: E402
from bot_core.security.signing import build_hmac_signature  # noqa: E402


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AdapterValidationResult:
    """Wynik próby inicjalizacji adaptera giełdowego."""

    name: str
    status: str
    details: MutableMapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationResult:
    """Podsumowanie symulowanych zleceń PaperTradingExecutionService."""

    symbols: Sequence[str]
    attempted: int
    executed: int
    ledger_rows: int
    total_notional: float


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/core.yaml",
        help="Ścieżka do pliku config/core.yaml",
    )
    parser.add_argument(
        "--environment",
        action="append",
        dest="environments",
        help="Środowisko live do walidacji (można podać wielokrotnie).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Ścieżka pliku JSON z raportem dry-run.",
    )
    parser.add_argument(
        "--decision-log",
        type=Path,
        help="Ścieżka JSONL decision logu do zaktualizowania.",
    )
    parser.add_argument(
        "--decision-log-hmac-key",
        help="Klucz HMAC do podpisu decision logu (wartość bezpośrednio).",
    )
    parser.add_argument(
        "--decision-log-hmac-key-env",
        help="Nazwa zmiennej środowiskowej z kluczem HMAC dla decision logu.",
    )
    parser.add_argument(
        "--decision-log-hmac-key-file",
        type=Path,
        help="Plik zawierający klucz HMAC dla decision logu.",
    )
    parser.add_argument(
        "--decision-log-key-id",
        help="Identyfikator klucza dołączany do podpisu decision logu.",
    )
    parser.add_argument(
        "--notes",
        help="Dodatkowe notatki zapisane w raporcie i decision logu.",
    )
    parser.add_argument(
        "--skip-decision-log",
        action="store_true",
        help="Nie zapisuj wpisu decision logu.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=5,
        help="Maksymalna liczba symboli symulowanych na środowisko (domyślnie 5).",
    )
    return parser.parse_args(argv)


def _ensure_live_environment(env_cfg: EnvironmentConfig, env_name: str) -> None:
    if env_cfg.environment is not Environment.LIVE:
        raise ValueError(
            f"Środowisko {env_name} nie jest typu live (wykryto {env_cfg.environment.value})."
        )


def _resolve_environments(config: CoreConfig, selected: Sequence[str] | None) -> list[str]:
    if selected:
        return [name for name in selected if name in config.environments]
    return [
        name
        for name, env_cfg in config.environments.items()
        if getattr(env_cfg, "environment", None) is Environment.LIVE
    ]


def _resolve_adapter_factories() -> Mapping[str, Callable[..., ExchangeAdapter]]:
    factories = get_registered_adapter_factories()
    return dict(factories)


def _instantiate_adapter(
    exchange: str,
    factory_map: Mapping[str, Callable[..., ExchangeAdapter]],
    env_cfg: EnvironmentConfig,
) -> AdapterValidationResult:
    result = AdapterValidationResult(name=exchange, status="PASS")
    factory = factory_map.get(exchange)
    if factory is None:
        result.status = "FAIL"
        result.details["error"] = "Brak zarejestrowanej fabryki adaptera"
        return result

    credentials = ExchangeCredentials(
        key_id=f"dry-run-{exchange}",
        secret="dry-run",
        environment=Environment.LIVE,
        permissions=env_cfg.required_permissions or (),
    )
    settings = {}
    if isinstance(env_cfg.adapter_settings, Mapping):
        candidate = env_cfg.adapter_settings.get(exchange)
        if isinstance(candidate, Mapping):
            settings = dict(candidate)

    try:
        adapter = factory(credentials, environment=Environment.LIVE, settings=settings)
    except TypeError:
        try:
            adapter = factory(credentials)
        except Exception as exc:  # pragma: no cover - diagnostyka fabryki
            result.status = "FAIL"
            result.details["error"] = f"Inicjalizacja adaptera nie powiodła się: {exc}"
            return result
    except Exception as exc:  # pragma: no cover - diagnostyka fabryki
        result.status = "FAIL"
        result.details["error"] = f"Inicjalizacja adaptera nie powiodła się: {exc}"
        return result

    try:
        adapter.configure_network(ip_allowlist=env_cfg.ip_allowlist or ())
    except Exception as exc:  # pragma: no cover - konfiguracja sieci
        result.status = "FAIL"
        result.details["error"] = f"configure_network nie powiodło się: {exc}"
        return result

    result.details["class"] = f"{adapter.__class__.__module__}.{adapter.__class__.__name__}"
    if hasattr(adapter, "name"):
        result.details["adapter_name"] = getattr(adapter, "name")
    return result


def _collect_live_exchanges(
    *,
    config: CoreConfig,
    env_cfg: EnvironmentConfig,
) -> Sequence[str]:
    routing = getattr(config, "live_routing", None)
    exchanges: list[str] = []
    if routing and getattr(routing, "default_route", None):
        exchanges.extend(str(name) for name in routing.default_route)
        for override in routing.route_overrides.values():
            exchanges.extend(str(name) for name in override)
    if env_cfg.exchange:
        exchanges.append(str(env_cfg.exchange))
    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in exchanges:
        normalized = item.strip()
        if normalized and normalized not in seen:
            deduplicated.append(normalized)
            seen.add(normalized)
    return tuple(deduplicated)


def _derive_markets(
    *,
    config: CoreConfig,
    env_cfg: EnvironmentConfig,
    max_symbols: int,
) -> tuple[Mapping[str, MarketMetadata], Sequence[str]]:
    universe_name = getattr(env_cfg, "instrument_universe", None)
    markets: dict[str, MarketMetadata] = {}
    used_symbols: list[str] = []
    if universe_name and universe_name in getattr(config, "instrument_universes", {}):
        universe = config.instrument_universes[universe_name]
        for instrument in universe.instruments:
            mapping = getattr(instrument, "exchange_symbols", None) or getattr(instrument, "exchanges", {})
            if not isinstance(mapping, Mapping):
                continue
            symbol = mapping.get(env_cfg.exchange)
            if not symbol:
                continue
            base = getattr(instrument, "base_asset", None)
            quote = getattr(instrument, "quote_asset", None)
            if not base or not quote:
                base, quote = AutoTrader._split_symbol(symbol)  # type: ignore[attr-defined]
            markets[symbol] = MarketMetadata(base_asset=str(base), quote_asset=str(quote))
            used_symbols.append(symbol)
            if len(markets) >= max_symbols:
                break
    if not markets:
        base, quote = AutoTrader._split_symbol(env_cfg.default_strategy or "BTCUSDT")  # type: ignore[attr-defined]
        symbol = f"{base}{quote}"
        markets[symbol] = MarketMetadata(base_asset=base, quote_asset=quote)
        used_symbols.append(symbol)
    return markets, tuple(used_symbols)


def _simulate_orders(
    *,
    markets: Mapping[str, MarketMetadata],
    env_name: str,
    env_cfg: EnvironmentConfig,
) -> SimulationResult:
    balances: dict[str, float] = {}
    for metadata in markets.values():
        balances.setdefault(metadata.quote_asset, 0.0)
        balances[metadata.quote_asset] += 500_000.0

    service = PaperTradingExecutionService(markets, initial_balances=balances)
    context = ExecutionContext(
        portfolio_id=env_name,
        risk_profile=getattr(env_cfg, "risk_profile", "unknown"),
        environment=env_cfg.environment.value,
        metadata={"source": "live_execution_dry_run"},
    )

    attempted = 0
    executed = 0
    total_notional = 0.0
    for symbol, metadata in markets.items():
        attempted += 1
        try:
            quantity = max(0.1, metadata.min_quantity or 0.1)
            reference_price = max(1.0, metadata.min_notional or 100.0)
            if metadata.min_notional:
                reference_price = max(reference_price, metadata.min_notional / quantity)
            request = OrderRequest(
                symbol=symbol,
                side="buy",
                quantity=quantity,
                order_type="market",
                price=reference_price,
            )
            result = service.execute(request, context)
        except Exception as exc:
            LOGGER.error("Symulacja zlecenia %s w środowisku %s nie powiodła się: %s", symbol, env_name, exc)
            continue
        executed += 1
        if result.avg_price:
            total_notional += result.filled_quantity * float(result.avg_price)

    ledger_rows = len(list(service.ledger()))
    return SimulationResult(
        symbols=tuple(markets.keys()),
        attempted=attempted,
        executed=executed,
        ledger_rows=ledger_rows,
        total_notional=round(total_notional, 8),
    )


def _resolve_hmac_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    provided = [
        bool(args.decision_log_hmac_key),
        bool(args.decision_log_hmac_key_env),
        bool(args.decision_log_hmac_key_file),
    ]
    if sum(provided) > 1:
        raise ValueError(
            "Podaj klucz HMAC jako jedną z opcji: --decision-log-hmac-key, --decision-log-hmac-key-env lub --decision-log-hmac-key-file."
        )
    key: bytes | None = None
    if args.decision_log_hmac_key:
        key = args.decision_log_hmac_key.encode("utf-8")
    elif args.decision_log_hmac_key_env:
        env_value = os.environ.get(args.decision_log_hmac_key_env)
        if not env_value:
            raise ValueError(
                f"Zmienna środowiskowa {args.decision_log_hmac_key_env} nie zawiera klucza HMAC."
            )
        key = env_value.encode("utf-8")
    elif args.decision_log_hmac_key_file:
        candidate = Path(args.decision_log_hmac_key_file).expanduser()
        if not candidate.is_file():
            raise ValueError(f"Plik klucza HMAC nie istnieje: {candidate}")
        key = candidate.read_bytes().strip()

    if key is not None and len(key) < 32:
        raise ValueError("Klucz HMAC decision logu musi mieć co najmniej 32 bajty")
    key_id = args.decision_log_key_id
    return key, key_id


def _default_report_path(base: Path | None) -> Path:
    root = base or Path("logs/live_dry_run")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"live_execution_dry_run_{timestamp}.json"


def _write_report(report_path: Path, payload: Mapping[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_decision_log(
    *,
    log_path: Path,
    entry: Mapping[str, object],
    key: bytes | None,
    key_id: str | None,
) -> None:
    payload = dict(entry)
    if key is not None:
        payload["signature"] = build_hmac_signature(payload, key=key, key_id=key_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _process_environment(
    *,
    env_name: str,
    config: CoreConfig,
    factory_map: Mapping[str, Callable[..., ExchangeAdapter]],
    max_symbols: int,
) -> Mapping[str, object]:
    env_cfg = config.environments[env_name]
    _ensure_live_environment(env_cfg, env_name)

    exchanges = _collect_live_exchanges(config=config, env_cfg=env_cfg)
    adapter_results: dict[str, Mapping[str, object]] = {}
    overall_status = "PASS"
    for exchange in exchanges:
        result = _instantiate_adapter(exchange, factory_map, env_cfg)
        adapter_results[exchange] = {
            "status": result.status,
            **result.details,
        }
        if result.status != "PASS":
            overall_status = "FAIL"

    markets, symbols = _derive_markets(config=config, env_cfg=env_cfg, max_symbols=max_symbols)
    simulation = _simulate_orders(markets=markets, env_name=env_name, env_cfg=env_cfg)
    if simulation.executed == 0:
        overall_status = "FAIL"

    return {
        "environment": env_name,
        "status": overall_status,
        "exchanges": exchanges,
        "adapters": adapter_results,
        "simulation": {
            "symbols": simulation.symbols,
            "orders_attempted": simulation.attempted,
            "orders_executed": simulation.executed,
            "ledger_rows": simulation.ledger_rows,
            "total_notional": simulation.total_notional,
        },
        "notes": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config_path = args.config.expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Plik konfiguracji nie istnieje: {config_path}")

    config = load_core_config(str(config_path))
    environments = _resolve_environments(config, args.environments)
    if not environments:
        raise SystemExit("Brak środowisk live do walidacji.")

    factory_map = _resolve_adapter_factories()
    environment_reports: list[Mapping[str, object]] = []
    overall_status = "PASS"

    for env_name in environments:
        try:
            report = _process_environment(
                env_name=env_name,
                config=config,
                factory_map=factory_map,
                max_symbols=max(1, int(args.max_symbols)),
            )
        except Exception as exc:  # pragma: no cover - diagnostyka środowiska
            LOGGER.exception("Walidacja środowiska %s zakończyła się błędem", env_name)
            report = {
                "environment": env_name,
                "status": "FAIL",
                "error": str(exc),
            }
        environment_reports.append(report)
        if report.get("status") != "PASS":
            overall_status = "FAIL"

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "generated_at": generated_at,
        "config": config_path.as_posix(),
        "status": overall_status,
        "environments": environment_reports,
        "notes": args.notes,
    }

    report_path = args.report.expanduser() if args.report else _default_report_path(None)
    _write_report(report_path, summary)
    print(f"Raport live execution dry-run zapisany w {report_path}")

    if args.decision_log and not args.skip_decision_log:
        key, key_id = _resolve_hmac_key(args)
        entry = {
            "timestamp": generated_at,
            "category": "live_execution_dry_run",
            "status": overall_status,
            "report": report_path.as_posix(),
            "environments": [
                {
                    "name": item.get("environment"),
                    "status": item.get("status"),
                    "orders_executed": item.get("simulation", {}).get("orders_executed"),
                }
                for item in environment_reports
            ],
            "notes": args.notes,
        }
        _append_decision_log(
            log_path=args.decision_log.expanduser(),
            entry=entry,
            key=key,
            key_id=key_id,
        )
        print(f"Zapisano wpis decision logu: {args.decision_log}")

    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())

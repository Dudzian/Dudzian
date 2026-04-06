#!/usr/bin/env python3
"""Pobiera rzeczywiste snapshoty metryk long-polla z działającego stream gateway."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bot_core.config import load_core_config
from bot_core.exchanges.streaming import LocalLongPollStream
from bot_core.observability.metrics import MetricsRegistry


# Scope jest pobierany z rzeczywistego runtime streamu przez gateway (`/stream/<adapter>/<scope>`).
# W strict path nie odtwarzamy historycznego kształtu fixture (np. deribit=private),
# tylko zapisujemy metryki zgodne z bieżącym źródłem danych runtime.
@dataclass(frozen=True, slots=True)
class SnapshotTarget:
    environment_name: str
    adapter: str
    environment: str
    scope: str


_REQUIRED_TARGETS: tuple[SnapshotTarget, ...] = (
    SnapshotTarget("deribit_futures_paper", "deribit_futures", "paper", "public"),
    SnapshotTarget("deribit_futures_live", "deribit_futures", "live", "public"),
    SnapshotTarget("bitmex_futures_paper", "bitmex_futures", "paper", "public"),
    SnapshotTarget("bitmex_futures_live", "bitmex_futures", "live", "public"),
)


class SnapshotFetchError(RuntimeError):
    """Błąd pobierania snapshotów long-polla."""


@dataclass(frozen=True, slots=True)
class SnapshotEnvironmentConfig:
    adapter_settings: Mapping[str, Any]


def _iso_now_utc() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _load_health_check_public_symbol(config_path: Path, environment_name: str) -> str | None:
    # Narrow fallback on-demand: shared loader nie mapuje `health_check` do CoreConfig.
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - zależność środowiskowa
        raise SnapshotFetchError(
            "Brak zależności 'PyYAML' wymaganej do odczytu fallbacku health_check.public_symbol."
        ) from exc

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SnapshotFetchError(f"Nie znaleziono pliku konfiguracyjnego: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise SnapshotFetchError(
            f"Nie udało się odczytać konfiguracji YAML z pliku: {config_path}"
        ) from exc

    environments = payload.get("environments") if isinstance(payload, Mapping) else {}
    if not isinstance(environments, Mapping):
        return None

    entry = environments.get(environment_name)
    if not isinstance(entry, Mapping):
        return None
    health_check = entry.get("health_check")
    if not isinstance(health_check, Mapping):
        return None
    symbol = health_check.get("public_symbol")
    if isinstance(symbol, str) and symbol.strip():
        return symbol.strip()
    return None


def _load_snapshot_environment_configs(
    config_path: Path,
) -> Mapping[str, SnapshotEnvironmentConfig]:
    try:
        core_config = load_core_config(config_path)
    except Exception as exc:
        raise SnapshotFetchError(
            f"Nie udało się wczytać konfiguracji z {config_path}: {exc}"
        ) from exc

    environments: dict[str, SnapshotEnvironmentConfig] = {}
    for name, environment in core_config.environments.items():
        environments[name] = SnapshotEnvironmentConfig(
            adapter_settings=environment.adapter_settings,
        )
    return environments


def _has_symbol(params: Mapping[str, Any]) -> bool:
    for key in ("symbol", "symbols"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if any(str(item).strip() for item in value):
                return True
    return False


def _is_ticker_channel_requested(channels: Sequence[str]) -> bool:
    normalized = {str(channel).strip().lower() for channel in channels if str(channel).strip()}
    return "ticker" in normalized or "tickers" in normalized


def _resolve_public_stream_params(
    *,
    config_path: Path,
    environment_name: str,
    environment_config: SnapshotEnvironmentConfig,
    channels: Sequence[str],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    adapter_settings = environment_config.adapter_settings
    if isinstance(adapter_settings, Mapping):
        stream = adapter_settings.get("stream")
        if isinstance(stream, Mapping):
            public_params = stream.get("public_params")
            if isinstance(public_params, Mapping):
                params = {str(key): value for key, value in public_params.items()}

    if _is_ticker_channel_requested(channels) and not _has_symbol(params):
        symbol = _load_health_check_public_symbol(config_path, environment_name)
        if not symbol:
            raise SnapshotFetchError(
                "Brak konfiguracji symbolu dla kanału 'ticker' "
                f"w środowisku '{environment_name}'. "
                "Ustaw adapter_settings.stream.public_params.symbol/symbols "
                "lub health_check.public_symbol."
            )
        params["symbol"] = symbol

    return params


def _collect_single_snapshot(
    *,
    base_url: str,
    adapter: str,
    environment: str,
    scope: str,
    channels: Sequence[str],
    params: Mapping[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    registry = MetricsRegistry()
    stream = LocalLongPollStream(
        base_url=base_url,
        path=f"/stream/{adapter}/{scope}",
        channels=channels,
        adapter=adapter,
        scope=scope,
        environment=environment,
        params=params,
        poll_interval=0.1,
        timeout=max(0.5, timeout_seconds),
        max_retries=2,
        backoff_base=0.1,
        backoff_cap=1.0,
        metrics_registry=registry,
    )

    deadline = time.monotonic() + max(1.0, timeout_seconds)
    stream.start()
    try:
        while time.monotonic() < deadline:
            snapshot = stream.export_metrics_snapshot()
            latency = snapshot.get("requestLatency")
            count = latency.get("count") if isinstance(latency, Mapping) else 0
            if isinstance(count, (int, float)) and count > 0:
                snapshot["collected_at"] = _iso_now_utc()
                return snapshot
            time.sleep(0.1)
    finally:
        stream.close()

    raise SnapshotFetchError(
        f"Brak aktywności long-polla dla {adapter}:{environment} (scope={scope}, timeout={timeout_seconds}s)"
    )


def fetch_snapshots(
    *,
    base_url: str,
    config_path: Path,
    output_path: Path,
    channels: Sequence[str],
    timeout_seconds: float,
) -> Path:
    if not channels:
        raise SnapshotFetchError("Wymagane jest podanie co najmniej jednego kanału long-polla")

    environments_config = _load_snapshot_environment_configs(config_path)

    snapshots: list[dict[str, Any]] = []
    for target in _REQUIRED_TARGETS:
        environment_config = environments_config.get(target.environment_name)
        if not isinstance(environment_config, SnapshotEnvironmentConfig):
            raise SnapshotFetchError(
                f"Brak konfiguracji środowiska '{target.environment_name}' w pliku {config_path}"
            )

        params = _resolve_public_stream_params(
            config_path=config_path,
            environment_name=target.environment_name,
            environment_config=environment_config,
            channels=channels,
        )

        snapshot = _collect_single_snapshot(
            base_url=base_url,
            adapter=target.adapter,
            environment=target.environment,
            scope=target.scope,
            channels=channels,
            params=params,
            timeout_seconds=timeout_seconds,
        )
        labels = snapshot.get("labels")
        if not isinstance(labels, Mapping):
            raise SnapshotFetchError(
                f"Snapshot {target.adapter}:{target.environment} nie zawiera poprawnych etykiet"
            )
        snapshot_labels = dict(labels)
        snapshot_labels["adapter"] = str(snapshot_labels.get("adapter") or "").strip()
        snapshot_labels["scope"] = str(snapshot_labels.get("scope") or "").strip() or target.scope
        snapshot_labels["environment"] = str(snapshot_labels.get("environment") or "").strip()
        snapshot["labels"] = snapshot_labels
        snapshots.append(snapshot)

    seen = {
        (
            str(item.get("labels", {}).get("adapter") or "").strip(),
            str(item.get("labels", {}).get("environment") or "").strip(),
        )
        for item in snapshots
    }
    missing = [
        f"{target.adapter}:{target.environment}"
        for target in _REQUIRED_TARGETS
        if (target.adapter, target.environment) not in seen
    ]
    if missing:
        raise SnapshotFetchError("Brak wymaganych snapshotów: " + ", ".join(missing))

    payload = {
        "collected_at": _iso_now_utc(),
        "snapshots": snapshots,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def _cleanup_long_poll_runtime() -> None:
    """Domyka współdzielone zasoby HTTP/AnyIO używane przez long-poll fetcher."""

    with contextlib.suppress(Exception):
        from bot_core.exchanges import http_client as exchange_http_client

        shutdown_client_cache = getattr(exchange_http_client, "_shutdown_client_cache", None)
        if callable(shutdown_client_cache):
            shutdown_client_cache()

    with contextlib.suppress(Exception):
        from core.network import sync as network_sync

        shutdown_portal = getattr(network_sync, "_shutdown_portal", None)
        if callable(shutdown_portal):
            shutdown_portal()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="Adres stream gateway")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/core.yaml"),
        help="Plik konfiguracyjny środowisk wykorzystywany do parametryzacji streamu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/metrics/long_poll_snapshots.json"),
        help="Docelowy plik JSON ze snapshotami long-polla.",
    )
    parser.add_argument(
        "--channels",
        default="ticker",
        help="Kanały long-polla do odpytania (rozdzielone przecinkiem).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="Limit czasu pobierania pojedynczego snapshotu.",
    )
    args = parser.parse_args(argv)

    channels = tuple(item.strip() for item in str(args.channels).split(",") if item.strip())

    try:
        path = fetch_snapshots(
            base_url=str(args.base_url),
            config_path=args.config,
            output_path=args.output,
            channels=channels,
            timeout_seconds=float(args.timeout_seconds),
        )
    except SnapshotFetchError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        _cleanup_long_poll_runtime()

    print(f"Pobrano rzeczywiste snapshoty long-polla: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

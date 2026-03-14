#!/usr/bin/env python3
"""Pobiera rzeczywiste snapshoty metryk long-polla z działającego stream gateway."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bot_core.exchanges.streaming import LocalLongPollStream
from bot_core.observability.metrics import MetricsRegistry

# Scope jest pobierany z rzeczywistego runtime streamu przez gateway (`/stream/<adapter>/<scope>`).
# W strict path nie odtwarzamy historycznego kształtu fixture (np. deribit=private),
# tylko zapisujemy metryki zgodne z bieżącym źródłem danych runtime.
_REQUIRED_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("deribit_futures", "paper", "public"),
    ("deribit_futures", "live", "public"),
    ("bitmex_futures", "paper", "public"),
    ("bitmex_futures", "live", "public"),
)


class SnapshotFetchError(RuntimeError):
    """Błąd pobierania snapshotów long-polla."""


def _iso_now_utc() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _collect_single_snapshot(
    *,
    base_url: str,
    adapter: str,
    environment: str,
    scope: str,
    channels: Sequence[str],
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
    output_path: Path,
    channels: Sequence[str],
    timeout_seconds: float,
) -> Path:
    if not channels:
        raise SnapshotFetchError("Wymagane jest podanie co najmniej jednego kanału long-polla")

    snapshots: list[dict[str, Any]] = []
    for adapter, environment, scope in _REQUIRED_TARGETS:
        snapshot = _collect_single_snapshot(
            base_url=base_url,
            adapter=adapter,
            environment=environment,
            scope=scope,
            channels=channels,
            timeout_seconds=timeout_seconds,
        )
        labels = snapshot.get("labels")
        if not isinstance(labels, Mapping):
            raise SnapshotFetchError(
                f"Snapshot {adapter}:{environment} nie zawiera poprawnych etykiet"
            )
        snapshot_labels = dict(labels)
        snapshot_labels["adapter"] = str(snapshot_labels.get("adapter") or "").strip()
        snapshot_labels["scope"] = str(snapshot_labels.get("scope") or "").strip() or scope
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
        f"{adapter}:{environment}"
        for adapter, environment, _scope in _REQUIRED_TARGETS
        if (adapter, environment) not in seen
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="Adres stream gateway")
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
            output_path=args.output,
            channels=channels,
            timeout_seconds=float(args.timeout_seconds),
        )
    except SnapshotFetchError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Pobrano rzeczywiste snapshoty long-polla: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

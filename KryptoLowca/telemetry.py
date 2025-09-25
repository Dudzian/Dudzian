"""Utility helpers for persisting and streaming telemetry snapshots."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - pandas optional
    _pd = None

try:  # pragma: no cover - grpc may be optional in tests
    import grpc  # type: ignore
except Exception:  # pragma: no cover
    grpc = None  # type: ignore

try:  # pragma: no cover
    from grpc import aio as grpc_aio  # type: ignore
except Exception:  # pragma: no cover
    grpc_aio = None  # type: ignore

from KryptoLowca.telemetry_pb import (
    ApiTelemetrySnapshot,
    TelemetryAck,
    PROTOBUF_AVAILABLE,
    build_snapshot_message,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TelemetryWriter:
    """Persist telemetry snapshots to local storage and optionally stream via gRPC."""

    def __init__(
        self,
        *,
        storage_path: Path | str,
        exchange: str,
        mode: str = "paper",
        grpc_target: Optional[str] = None,
        aggregate_intervals: Sequence[int] = (1, 10, 60),
    ) -> None:
        self.storage_dir = Path(storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.exchange = exchange
        self.mode = mode
        self.aggregate_intervals = tuple(int(x) for x in aggregate_intervals if int(x) > 0)
        self.sqlite_path = self.storage_dir / "telemetry.sqlite"
        self.parquet_path = self.storage_dir / "telemetry.parquet"
        self.proto_path = self.storage_dir / "latest_snapshot.pb"
        self._has_pandas = _pd is not None
        self._ensure_sqlite()

        self._grpc_target = grpc_target
        self._grpc_channel = None
        self._grpc_stub = None
        if grpc_target and grpc is not None and PROTOBUF_AVAILABLE:
            try:  # pragma: no cover - zależne od środowiska
                self._grpc_channel = grpc.insecure_channel(grpc_target)
                self._grpc_stub = self._grpc_channel.unary_unary(
                    "/telemetry.TelemetryStream/PushSnapshot",
                    request_serializer=ApiTelemetrySnapshot.SerializeToString,
                    response_deserializer=TelemetryAck.FromString,
                )
            except Exception:
                logger.exception("Nie udało się skonfigurować kanału gRPC")
                self._grpc_channel = None
                self._grpc_stub = None
        elif grpc_target:
            logger.warning(
                "gRPC lub protobuf nie są dostępne – snapshoty będą zapisywane lokalnie (target=%s)",
                grpc_target,
            )

    # -------------------- Public API --------------------
    def write_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Persist telemetry snapshot and forward it to optional sinks."""

        message: Any | None = None
        if PROTOBUF_AVAILABLE:
            try:
                message = build_snapshot_message(snapshot, exchange=self.exchange, mode=self.mode)
            except Exception:
                logger.exception("Nie udało się zbudować wiadomości protobuf – pomijam serializację binarną")
                message = None
        else:
            logger.debug("Protobuf nie jest dostępny – zapisuję jedynie dane lokalne")

        if message is not None:
            self._write_proto(message)
        self._write_sqlite(snapshot)
        self._write_parquet(snapshot)
        if message is not None:
            self._send_grpc(message)

    # -------------------- Helpers --------------------
    def _write_proto(self, message: Any) -> None:
        try:
            self.proto_path.write_bytes(message.SerializeToString())
        except Exception:  # pragma: no cover - zależne od IO
            logger.exception("Nie udało się zapisać binarnego snapshotu telemetryjnego")

    def _ensure_sqlite(self) -> None:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS telemetry_snapshots (
                        timestamp_ns INTEGER,
                        schema_version INTEGER,
                        exchange TEXT,
                        mode TEXT,
                        total_calls INTEGER,
                        total_errors INTEGER,
                        avg_latency_ms REAL,
                        max_latency_ms REAL,
                        last_latency_ms REAL,
                        consecutive_errors INTEGER,
                        window_calls INTEGER,
                        window_errors INTEGER,
                        last_endpoint TEXT,
                        current_window_usage REAL,
                        rate_limit_window_seconds REAL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rate_limit_snapshots (
                        timestamp_ns INTEGER,
                        bucket_name TEXT,
                        capacity INTEGER,
                        count INTEGER,
                        usage REAL,
                        max_usage REAL,
                        reset_in_seconds REAL,
                        window_seconds REAL,
                        alert_active INTEGER
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS telemetry_aggregates (
                        interval_s INTEGER,
                        bucket_ts_ns INTEGER,
                        exchange TEXT,
                        mode TEXT,
                        avg_latency_ms REAL,
                        total_calls INTEGER,
                        total_errors INTEGER,
                        PRIMARY KEY(interval_s, bucket_ts_ns, exchange, mode)
                    )
                    """
                )
        finally:
            conn.close()

    def _write_sqlite(self, snapshot: Dict[str, Any]) -> None:
        conn = sqlite3.connect(self.sqlite_path)
        ts_ns = int(snapshot.get("timestamp_ns", 0))
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO telemetry_snapshots (
                        timestamp_ns, schema_version, exchange, mode, total_calls, total_errors,
                        avg_latency_ms, max_latency_ms, last_latency_ms, consecutive_errors,
                        window_calls, window_errors, last_endpoint, current_window_usage, rate_limit_window_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts_ns,
                        int(snapshot.get("schema_version", 1)),
                        self.exchange,
                        self.mode,
                        int(snapshot.get("total_calls", 0)),
                        int(snapshot.get("total_errors", 0)),
                        float(snapshot.get("avg_latency_ms", 0.0)),
                        float(snapshot.get("max_latency_ms", 0.0)),
                        float(snapshot.get("last_latency_ms", 0.0)),
                        int(snapshot.get("consecutive_errors", 0)),
                        int(snapshot.get("window_calls", 0)),
                        int(snapshot.get("window_errors", 0)),
                        snapshot.get("last_endpoint"),
                        float(snapshot.get("current_window_usage") or 0.0),
                        float(snapshot.get("rate_limit_window_seconds", 0.0)),
                    ),
                )
                for bucket in snapshot.get("rate_limit_buckets", []):
                    conn.execute(
                        """
                        INSERT INTO rate_limit_snapshots (
                            timestamp_ns, bucket_name, capacity, count, usage, max_usage,
                            reset_in_seconds, window_seconds, alert_active
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ts_ns,
                            bucket.get("name"),
                            int(bucket.get("capacity", 0)),
                            int(bucket.get("count", 0)),
                            float(bucket.get("usage", 0.0)),
                            float(bucket.get("max_usage", 0.0)),
                            float(bucket.get("reset_in_seconds", 0.0)),
                            float(bucket.get("window_seconds", 0.0)),
                            1 if bucket.get("alert_active") else 0,
                        ),
                    )
                for interval in self.aggregate_intervals:
                    bucket_ts_ns = (ts_ns // (interval * 1_000_000_000)) * (interval * 1_000_000_000)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO telemetry_aggregates (
                            interval_s, bucket_ts_ns, exchange, mode, avg_latency_ms, total_calls, total_errors
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            interval,
                            bucket_ts_ns,
                            self.exchange,
                            self.mode,
                            float(snapshot.get("avg_latency_ms", 0.0)),
                            int(snapshot.get("total_calls", 0)),
                            int(snapshot.get("total_errors", 0)),
                        ),
                    )
        except Exception:
            logger.exception("Nie udało się zapisać snapshotu do SQLite")
        finally:
            conn.close()

    def _write_parquet(self, snapshot: Dict[str, Any]) -> None:
        if not self._has_pandas:
            return
        data = {
            "timestamp": datetime.fromtimestamp(snapshot.get("timestamp_ns", 0) / 1_000_000_000 or 0, tz=timezone.utc),
            "exchange": self.exchange,
            "mode": self.mode,
            "avg_latency_ms": snapshot.get("avg_latency_ms", 0.0),
            "max_latency_ms": snapshot.get("max_latency_ms", 0.0),
            "total_calls": snapshot.get("total_calls", 0),
            "total_errors": snapshot.get("total_errors", 0),
            "current_window_usage": snapshot.get("current_window_usage", 0.0),
        }
        frame = _pd.DataFrame([data])  # type: ignore[arg-type]
        try:
            if self.parquet_path.exists():
                existing = _pd.read_parquet(self.parquet_path)  # type: ignore[assignment]
                frame = _pd.concat([existing, frame], ignore_index=True)  # type: ignore[assignment]
            frame.to_parquet(self.parquet_path, index=False)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - zależne od pyarrow
            logger.exception("Nie udało się zapisać snapshotów do Parquet – fallback CSV")
            csv_path = self.parquet_path.with_suffix(".csv")
            header = not csv_path.exists()
            frame.to_csv(csv_path, mode="a", header=header, index=False)  # type: ignore[arg-type]

    def _send_grpc(self, message: Any) -> None:
        if self._grpc_stub is None:
            return
        try:  # pragma: no cover - zależne od środowiska
            self._grpc_stub(message, timeout=1.0)
        except Exception:
            logger.exception("Wysłanie snapshotu telemetryjnego przez gRPC nie powiodło się")


__all__ = ["TelemetryWriter"]

#!/usr/bin/env python3
"""Synthetic probes sprawdzające DR CloudOrchestratora i alarmy alertingu."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import grpc
from google.protobuf import empty_pb2

from bot_core.generated import trading_pb2_grpc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic probes CloudOrchestratora (DR)")
    parser.add_argument(
        "--endpoint",
        default="localhost:50051",
        help="Adres gRPC CloudOrchestratora (domyślnie localhost:50051)",
    )
    parser.add_argument(
        "--alertmanager-url",
        dest="alertmanager_url",
        help="Opcjonalny endpoint Alertmanagera do walidacji alarmów",
    )
    parser.add_argument(
        "--prometheus-url",
        dest="prometheus_url",
        help="Opcjonalny endpoint Prometheusa do walidacji metryk (_health/_lastError)",
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=int,
        default=5000,
        help="Maksymalne dopuszczalne opóźnienie RPC HealthService.Check w milisekundach (domyślnie 5000)",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=5,
        help="Timeout (s) dla gRPC HealthService.Check oraz zestawienia kanału (domyślnie 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/ci/dr_probes",
        help="Katalog docelowy raportu (domyślnie reports/ci/dr_probes)",
    )
    return parser.parse_args(argv)


def _http_json(url: str, *, timeout: int = 10) -> Any:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _probe_health(
    endpoint: str,
    *,
    latency_threshold_ms: int,
    health_timeout: int,
) -> tuple[dict[str, Any], bool]:
    snapshot: dict[str, Any] = {
        "endpoint": endpoint,
        "status": "unknown",
        "workers": [],
        "headers": {},
    }
    ok = False
    channel = grpc.insecure_channel(endpoint)
    stub = trading_pb2_grpc.HealthServiceStub(channel)
    start = time.perf_counter()
    try:
        grpc.channel_ready_future(channel).result(timeout=health_timeout)
        response, call = stub.Check.with_call(empty_pb2.Empty(), timeout=health_timeout)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        snapshot["latencyMs"] = latency_ms
        initial = {key: value for key, value in (call.initial_metadata() or [])}
        trailing = {key: value for key, value in (call.trailing_metadata() or [])}
        headers = {**initial, **trailing}
        snapshot["headers"] = headers
        snapshot["status"] = response.cloud_health.status or "unknown"
        snapshot["workers"] = [
            {
                "name": worker.name,
                "enabled": worker.enabled,
                "lastRunAt": worker.last_run_at,
                "lastError": worker.last_error,
                "intervalSeconds": worker.interval_seconds,
            }
            for worker in response.cloud_health.workers
        ]
        last_error = headers.get("x-bot-cloud-last-error") or response.cloud_health.last_error
        health_flag = headers.get("x-bot-cloud-health", "1") != "0"
        snapshot["_health"] = bool(health_flag)
        snapshot["_lastError"] = last_error
        latency_ok = latency_ms <= latency_threshold_ms
        ok = bool(health_flag and not last_error and latency_ok)
        if not ok:
            reasons = []
            if last_error:
                reasons.append(last_error)
            if not latency_ok:
                reasons.append(f"latency_ms_exceeded:{latency_ms}>{latency_threshold_ms}")
            if not health_flag:
                reasons.append("cloud_unhealthy")
            snapshot["failureReason"] = ";".join(reasons) or "cloud_unhealthy"
    except grpc.FutureTimeoutError as exc:  # pragma: no cover - zależne od środowiska
        snapshot["failureReason"] = f"health_rpc_timeout: {exc}"
    except Exception as exc:  # pragma: no cover - diagnostyka probe
        snapshot["failureReason"] = f"health_rpc_error: {exc}"
    finally:
        try:
            channel.close()
        except Exception:
            pass
    return snapshot, ok


def _probe_alertmanager(alertmanager_url: str | None) -> tuple[dict[str, Any], bool]:
    if not alertmanager_url:
        return {"skipped": True, "reason": "alertmanager_url_missing"}, True
    payload: dict[str, Any] = {"endpoint": alertmanager_url, "alerts": [], "firing": 0}
    ok = False
    try:
        alerts = _http_json(alertmanager_url.rstrip("/") + "/api/v2/alerts")
        relevant = []
        for alert in alerts:
            labels = alert.get("labels") or {}
            if labels.get("service") == "cloud-orchestrator" or "CloudWorker" in labels.get("alertname", ""):
                relevant.append(alert)
        payload["alerts"] = relevant
        payload["firing"] = len(relevant)
        ok = True
    except URLError as exc:  # pragma: no cover - zależne od środowiska
        payload["failureReason"] = f"alertmanager_unreachable: {exc}"
    except Exception as exc:  # pragma: no cover - zależne od środowiska
        payload["failureReason"] = f"alertmanager_error: {exc}"
    return payload, ok


def _probe_prometheus(prometheus_url: str | None) -> tuple[dict[str, Any], bool]:
    if not prometheus_url:
        return {"skipped": True, "reason": "prometheus_url_missing"}, True
    payload: dict[str, Any] = {"endpoint": prometheus_url, "samples": {}, "validation": {}}
    ok = False
    try:
        for metric in ("bot_cloud_health_status", "bot_cloud_last_error"):
            query = urlencode({"query": metric})
            url = prometheus_url.rstrip("/") + f"/api/v1/query?{query}"
            response = _http_json(url)
            payload["samples"][metric] = response
        health_samples = payload["samples"].get("bot_cloud_health_status", {}).get("data", {}).get("result") or []
        last_error_samples = payload["samples"].get("bot_cloud_last_error", {}).get("data", {}).get("result") or []
        payload["validation"] = {
            "healthValues": [float(sample.get("value", [0, 0])[1]) for sample in health_samples],
            "lastErrorValues": [float(sample.get("value", [0, 0])[1]) for sample in last_error_samples],
        }
        health_ok = bool(payload["validation"]["healthValues"] and all(v >= 1.0 for v in payload["validation"]["healthValues"]))
        last_error_ok = not payload["validation"]["lastErrorValues"] or all(
            v <= 0.0 for v in payload["validation"]["lastErrorValues"]
        )
        ok = health_ok and last_error_ok
        if not ok:
            payload["failureReason"] = "prometheus_health_failed"
    except URLError as exc:  # pragma: no cover - zależne od środowiska
        payload["failureReason"] = f"prometheus_unreachable: {exc}"
    except Exception as exc:  # pragma: no cover - zależne od środowiska
        payload["failureReason"] = f"prometheus_error: {exc}"
    return payload, ok


def _write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"cloud_orchestrator_probe_{timestamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    latency_threshold_ms = max(args.latency_threshold_ms, 0)
    health_timeout = max(args.health_timeout, 1)

    health_snapshot, health_ok = _probe_health(
        args.endpoint,
        latency_threshold_ms=latency_threshold_ms,
        health_timeout=health_timeout,
    )
    alert_payload, alert_ok = _probe_alertmanager(args.alertmanager_url)
    prometheus_payload, prom_ok = _probe_prometheus(args.prometheus_url)

    report = {
        "timestamp": int(time.time()),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "cloud": health_snapshot,
        "alertmanager": alert_payload,
        "prometheus": prometheus_payload,
        "summary": {
            "healthOk": health_ok,
            "alertmanagerOk": alert_ok,
            "prometheusOk": prom_ok,
        },
    }
    report_path = _write_report(Path(args.output_dir), report)
    print(f"DR probe report written to {report_path}")

    if health_ok and alert_ok and prom_ok:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

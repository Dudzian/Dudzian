#!/usr/bin/env python3
"""Failover validation dla Alertmanager/Prometheus z porównaniem `_lastError`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bot_core.observability.dr_failover import (
    FailoverComparison,
    FailoverState,
    compare_last_error,
    evaluate_outcome,
)
from scripts.dr_synthetic_probes import _probe_alertmanager, _probe_health, _probe_prometheus


_DEF_OUTPUT = Path("reports/ci/dr_failover")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wymusza failover DR alertingu i porównuje _lastError")
    parser.add_argument("--endpoint", default="localhost:50051", help="Adres gRPC CloudOrchestratora")
    parser.add_argument("--alertmanager-url", dest="alertmanager_url", help="Primary Alertmanager")
    parser.add_argument("--prometheus-url", dest="prometheus_url", help="Primary Prometheus")
    parser.add_argument("--dr-alertmanager-url", dest="dr_alertmanager_url", help="DR Alertmanager")
    parser.add_argument("--dr-prometheus-url", dest="dr_prometheus_url", help="DR Prometheus")
    parser.add_argument(
        "--failover-webhook",
        help="Opcjonalny webhook wymuszający failover (POST JSON {triggeredAt})",
    )
    parser.add_argument("--wait-seconds", type=int, default=20, help="Czas na propagację failoveru")
    parser.add_argument(
        "--output-dir",
        default=_DEF_OUTPUT,
        type=Path,
        help=f"Katalog na raport porównawczy (domyślnie {_DEF_OUTPUT})",
    )
    parser.add_argument(
        "--snapshot-prefix",
        type=Path,
        help=(
            "Prefiks ścieżki dla snapshotów before/after (_lastError, rulesDigest). "
            "Gdy nie ustawione, snapshoty zapisują się w katalogu output pod prefiksem 'failover_snapshot'."
        ),
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=int,
        default=5000,
        help="Limit opóźnienia gRPC HealthService.Check w milisekundach",
    )
    parser.add_argument("--health-timeout", type=int, default=5, help="Timeout HealthService.Check")
    return parser.parse_args(argv)


def _trigger_failover(webhook: str | None) -> dict[str, Any]:
    if not webhook:
        return {"skipped": True}
    payload = json.dumps({"triggeredAt": int(time.time())}).encode("utf-8")
    request = Request(
        webhook,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            return {"status": response.status, "reason": response.reason}
    except URLError as exc:  # pragma: no cover - zależne od środowiska
        return {"error": f"webhook_unreachable:{exc}"}
    except Exception as exc:  # pragma: no cover - diagnostyka
        return {"error": f"webhook_error:{exc}"}


def _extract_state(snapshot: dict[str, Any]) -> FailoverState:
    cloud = snapshot.get("cloud") or {}
    prometheus = snapshot.get("prometheus") or {}
    alertmanager = snapshot.get("alertmanager") or {}
    rules_digest = None
    firing = None
    if isinstance(alertmanager, dict):
        firing = alertmanager.get("firing")
    if isinstance(prometheus, dict):
        samples = prometheus.get("samples") or {}
        raw_rules = samples.get("rulesDigest") or ""
        if isinstance(raw_rules, str):
            rules_digest = raw_rules or None
    return FailoverState(
        last_error=cloud.get("_lastError") or cloud.get("lastError"),
        rules_digest=rules_digest,
        firing_alerts=firing,
    )


def _collect_probe(
    *,
    endpoint: str,
    alertmanager_url: str | None,
    prometheus_url: str | None,
    latency_threshold_ms: int,
    health_timeout: int,
) -> tuple[dict[str, Any], bool]:
    health_snapshot, health_ok = _probe_health(
        endpoint,
        latency_threshold_ms=latency_threshold_ms,
        health_timeout=health_timeout,
    )
    alert_payload, alert_ok = _probe_alertmanager(alertmanager_url)
    prom_payload, prom_ok = _probe_prometheus(prometheus_url)

    if prom_payload and isinstance(prom_payload, dict):
        try:
            rules_query = urlencode({"query": "sum(federate_rules_digest)"})
            prom_payload.setdefault("samples", {})["rulesDigest"] = _http_scalar(
                prometheus_url, rules_query
            )
        except Exception:  # pragma: no cover - brak federacji rules
            prom_payload.setdefault("validation", {})["rulesDigest"] = "unavailable"

    snapshot = {"cloud": health_snapshot, "alertmanager": alert_payload, "prometheus": prom_payload}
    ok = bool(health_ok and alert_ok and prom_ok)
    return snapshot, ok


def _http_scalar(prometheus_url: str | None, query: str) -> str | None:
    if not prometheus_url:
        return None
    url = prometheus_url.rstrip("/") + f"/api/v1/query?{query}"
    data = _http_json(url)
    result = (data.get("data", {}) or {}).get("result") or []
    if result:
        return str(result[0].get("value", [None, None])[1])
    return None


def _http_json(url: str, *, timeout: int = 10) -> Any:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _write_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _state_to_dict(state: FailoverState) -> dict[str, Any]:
    return {
        "lastError": state.last_error,
        "rulesDigest": state.rules_digest,
        "firingAlerts": state.firing_alerts,
    }


def _write_state_snapshot(path: Path, state: FailoverState) -> Path:
    return _write_report(path, _state_to_dict(state))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dr_failover_report.json"
    snapshot_prefix = args.snapshot_prefix or output_dir / "failover_snapshot"

    before, before_ok = _collect_probe(
        endpoint=args.endpoint,
        alertmanager_url=args.alertmanager_url,
        prometheus_url=args.prometheus_url,
        latency_threshold_ms=max(args.latency_threshold_ms, 0),
        health_timeout=max(args.health_timeout, 1),
    )

    webhook = _trigger_failover(args.failover_webhook)
    time.sleep(max(args.wait_seconds, 0))

    after, after_ok = _collect_probe(
        endpoint=args.endpoint,
        alertmanager_url=args.dr_alertmanager_url or args.alertmanager_url,
        prometheus_url=args.dr_prometheus_url or args.prometheus_url,
        latency_threshold_ms=max(args.latency_threshold_ms, 0),
        health_timeout=max(args.health_timeout, 1),
    )

    before_state = _extract_state(before)
    after_state = _extract_state(after)

    before_snapshot_path = _write_state_snapshot(
        snapshot_prefix.with_name(f"{snapshot_prefix.name}_before.json"), before_state
    )
    after_snapshot_path = _write_state_snapshot(
        snapshot_prefix.with_name(f"{snapshot_prefix.name}_after.json"), after_state
    )

    comparison: FailoverComparison = compare_last_error(before_state, after_state)
    status, failure_reasons = evaluate_outcome(
        comparison, before_ok=before_ok, after_ok=after_ok
    )
    summary = {
        "beforeOk": before_ok,
        "afterOk": after_ok,
        "comparison": comparison.to_dict(),
        "webhook": webhook,
        "status": status,
        "failureReasons": failure_reasons,
        "states": {
            "before": _state_to_dict(before_state),
            "after": _state_to_dict(after_state),
        },
    }

    payload = {
        "generatedAt": int(time.time()),
        "before": before,
        "after": after,
        "summary": summary,
        "states": summary["states"],
        "snapshots": {
            "before": str(before_snapshot_path),
            "after": str(after_snapshot_path),
        },
    }
    _write_report(report_path, payload)
    print(f"Failover validation report written to {report_path}")

    if failure_reasons:
        print(
            "DR failover validation failed:",
            ", ".join(failure_reasons),
        )
        return 1

    print("DR failover validation passed: comparison healthy, probes ok")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

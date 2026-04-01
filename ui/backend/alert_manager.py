"""Komponent zarządzania polityką alertów feed health i Risk Journal."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Protocol

from .feed_health_tracker import FeedHealthTracker


class AlertEventSink(Protocol):
    def emit_feed_health_event(
        self,
        *,
        severity: str,
        title: str,
        body: str,
        context: Mapping[str, str],
        payload: Mapping[str, object],
    ) -> None: ...


class AlertManager:
    def __init__(
        self,
        *,
        runtime_config_loader: Callable[[], object | None],
        active_profile_loader: Callable[[], str],
        sink_loader: Callable[[], object | None],
        risk_diagnostics_normalizer: Callable[[Mapping[str, object]], dict[str, object]],
        mapping_normalizer: Callable[[object], Mapping[str, object]],
        history_changed: Callable[[], None],
        channels_changed: Callable[[], None],
    ) -> None:
        self._runtime_config_loader = runtime_config_loader
        self._active_profile_loader = active_profile_loader
        self._sink_loader = sink_loader
        self._risk_diagnostics_normalizer = risk_diagnostics_normalizer
        self._mapping_normalizer = mapping_normalizer
        self._history_changed = history_changed
        self._channels_changed = channels_changed
        self._feed_alert_history: deque[dict[str, object]] = deque(maxlen=12)
        self._feed_alert_channels: list[dict[str, object]] = []
        self._feed_thresholds: dict[str, float | None] = {}
        self.reload_feed_thresholds()
        self._feed_alert_state: dict[str, str] = {
            "latency": "ok",
            "reconnects": "ok",
            "downtime": "ok",
        }
        self._risk_journal_alert_state = "ok"

    @property
    def feed_alert_history(self) -> deque[dict[str, object]]:
        return self._feed_alert_history

    @property
    def feed_alert_channels(self) -> list[dict[str, object]]:
        return self._feed_alert_channels

    @property
    def feed_thresholds(self) -> dict[str, float | None]:
        return self._feed_thresholds

    @property
    def feed_alert_state(self) -> dict[str, str]:
        return self._feed_alert_state

    def classify_threshold(
        self,
        value: float | None,
        *,
        warning: float | None,
        critical: float | None,
    ) -> str:
        return FeedHealthTracker.classify_threshold(value, warning=warning, critical=critical)

    def aggregate_sla_state(self, states: Iterable[str]) -> str:
        return FeedHealthTracker.aggregate_sla_state(states)

    def evaluate_feed_health_alerts(
        self,
        *,
        payload: Mapping[str, object],
        latency_p95: float | None,
        adapter: str,
        last_error: str,
    ) -> None:
        thresholds = self._feed_thresholds
        status = str(payload.get("status", "unknown"))
        reconnects_raw = payload.get("reconnects", 0)
        downtime_raw = payload.get("downtimeMs", 0.0)
        try:
            reconnects_value = int(reconnects_raw)
        except (TypeError, ValueError):
            reconnects_value = 0
        try:
            downtime_ms = float(downtime_raw)
        except (TypeError, ValueError):
            downtime_ms = 0.0
        downtime_seconds = max(0.0, downtime_ms / 1000.0)

        latency_state = self.classify_threshold(
            latency_p95,
            warning=thresholds["latency_warning_ms"],
            critical=thresholds["latency_critical_ms"],
        )
        self.maybe_emit_feed_alert(
            "latency",
            latency_state,
            metric_label="Latencja p95 decision feedu",
            unit="ms",
            value=latency_p95,
            warning=thresholds["latency_warning_ms"],
            critical=thresholds["latency_critical_ms"],
            status=status,
            adapter=adapter,
            reconnects=reconnects_value,
            downtime_seconds=downtime_seconds,
            latency_p95=latency_p95,
            last_error=last_error,
        )

        reconnect_state = self.classify_threshold(
            float(reconnects_value),
            warning=thresholds["reconnects_warning"],
            critical=thresholds["reconnects_critical"],
        )
        self.maybe_emit_feed_alert(
            "reconnects",
            reconnect_state,
            metric_label="Liczba reconnectów decision feedu",
            unit="",
            value=float(reconnects_value),
            warning=thresholds["reconnects_warning"],
            critical=thresholds["reconnects_critical"],
            status=status,
            adapter=adapter,
            reconnects=reconnects_value,
            downtime_seconds=downtime_seconds,
            latency_p95=latency_p95,
            last_error=last_error,
        )

        downtime_state = self.classify_threshold(
            downtime_seconds,
            warning=thresholds["downtime_warning_seconds"],
            critical=thresholds["downtime_critical_seconds"],
        )
        self.maybe_emit_feed_alert(
            "downtime",
            downtime_state,
            metric_label="Łączny downtime decision feedu",
            unit="s",
            value=downtime_seconds,
            warning=thresholds["downtime_warning_seconds"],
            critical=thresholds["downtime_critical_seconds"],
            status=status,
            adapter=adapter,
            reconnects=reconnects_value,
            downtime_seconds=downtime_seconds,
            latency_p95=latency_p95,
            last_error=last_error,
        )

    def maybe_emit_feed_alert(
        self,
        key: str,
        severity: str,
        *,
        channel: str = "decision_journal",
        metric_label: str,
        unit: str,
        value: float | None,
        warning: float | None,
        critical: float | None,
        status: str,
        adapter: str,
        reconnects: int,
        downtime_seconds: float,
        latency_p95: float | None,
        last_error: str,
    ) -> None:
        previous = self._feed_alert_state.get(key, "ok")
        if severity == previous:
            return
        self._feed_alert_state[key] = severity
        sink = self._sink_loader()
        emit = getattr(sink, "emit_feed_health_event", None) if sink is not None else None
        if not callable(emit):
            sink = None
            emit = None
        router = None
        if sink is not None:
            try:
                router = getattr(sink, "router", None)
            except Exception:
                router = None
            if router is None:
                try:
                    router = getattr(sink, "_router", None)
                except Exception:
                    router = None

        severity_label = severity
        state = severity
        if severity == "ok":
            severity_label = "info"
            state = "recovered"
        else:
            state = "degraded"

        threshold_value: float | None = None
        if severity == "critical" and critical is not None:
            threshold_value = critical
        elif severity == "warning" and warning is not None:
            threshold_value = warning
        elif previous in {"critical", "warning"}:
            threshold_value = (
                critical if previous == "critical" and critical is not None else warning
            )

        body_parts: list[str] = []
        metric_value_text = self.format_feed_metric(value, unit)
        threshold_text = (
            self.format_feed_metric(threshold_value, unit) if threshold_value is not None else None
        )
        if severity == "ok":
            title = f"{metric_label} w normie"
            body_parts.append(f"{metric_label} powróciła do normy ({metric_value_text}).")
            if threshold_text:
                body_parts.append(f"Poprzedni próg odniesienia: {threshold_text}.")
        else:
            title = f"{metric_label} przekroczyła próg"
            if severity == "critical":
                title = f"Krytyczne odchylenie – {metric_label}"
            if threshold_text:
                body_parts.append(
                    f"{metric_label} wynosi {metric_value_text} (próg {threshold_text})."
                )
            else:
                body_parts.append(f"{metric_label} wynosi {metric_value_text}.")
        if last_error:
            body_parts.append(f"Ostatni błąd: {last_error}.")
        body = " ".join(body_parts)

        context: dict[str, str] = {
            "adapter": adapter,
            "status": status,
            "metric": key,
            "metric_label": metric_label,
            "metric_unit": unit,
            "state": state,
            "reconnects": str(reconnects),
            "downtime_seconds": f"{downtime_seconds:.3f}",
            "channel": channel,
        }
        if value is not None:
            context["metric_value"] = (
                f"{value:.3f}" if unit in {"ms", "s"} else str(int(round(value)))
            )
        if warning is not None:
            context["warning_threshold"] = (
                f"{warning:.3f}" if unit in {"ms", "s"} else str(int(round(warning)))
            )
        if critical is not None:
            context["critical_threshold"] = (
                f"{critical:.3f}" if unit in {"ms", "s"} else str(int(round(critical)))
            )
        if latency_p95 is not None:
            context["latency_p95_ms"] = f"{latency_p95:.3f}"
        if last_error:
            context["last_error"] = last_error

        payload: dict[str, object] = {
            "metric": key,
            "metric_label": metric_label,
            "metric_unit": unit,
            "metric_value": value,
            "warning_threshold": warning,
            "critical_threshold": critical,
            "status": status,
            "adapter": adapter,
            "reconnects": reconnects,
            "downtime_seconds": downtime_seconds,
            "latency_p95_ms": latency_p95,
            "state": state,
            "channel": channel,
        }
        if last_error:
            payload["last_error"] = last_error

        if callable(emit):
            emit(
                severity=severity_label,
                title=title,
                body=body,
                context=context,
                payload=payload,
            )

        self.record_feed_alert(
            severity=severity_label,
            state=state,
            metric=key,
            label=metric_label,
            unit=unit,
            value=value,
            warning=warning,
            critical=critical,
            adapter=adapter,
            status=status,
            reconnects=reconnects,
            downtime_seconds=downtime_seconds,
            latency_p95=latency_p95,
            last_error=last_error,
            router=router,
        )

    @staticmethod
    def format_feed_metric(value: float | None, unit: str) -> str:
        if value is None:
            return "brak danych"
        if unit == "ms":
            return f"{value:.1f} ms"
        if unit == "s":
            return f"{value:.1f} s"
        return f"{int(round(value))}"

    def maybe_emit_risk_journal_alert(
        self,
        *,
        diagnostics: Mapping[str, object],
        logger_warning: Callable[[str], None],
        logger_info: Callable[[str], None],
        metrics_record: Callable[..., None],
    ) -> None:
        normalized_diagnostics = self._risk_diagnostics_normalizer(diagnostics)
        incomplete_entries = int(normalized_diagnostics.get("incompleteEntries", 0))
        samples = normalized_diagnostics.get("incompleteSamples", [])
        incomplete_samples_count = int(
            normalized_diagnostics.get("incompleteSamplesCount", len(samples)) or 0
        )
        risk_flag_counts = self._mapping_normalizer(
            normalized_diagnostics.get("riskFlagCounts", {})
        )
        severity = "warning" if incomplete_entries else "ok"
        previous = self._risk_journal_alert_state
        if severity == previous:
            return
        self._risk_journal_alert_state = severity

        environment = self._active_profile_loader() or "default"

        body = (
            "wykryto niekompletne wpisy Risk Journal wymagające pól risk_flags/stress_overrides lub risk_action"
            if incomplete_entries
            else "wpisy Risk Journal zawierają wymagane pola"
        )
        if incomplete_entries and samples:
            body = f"{body} (przykłady: {json.dumps(samples, ensure_ascii=False)})"
        if incomplete_entries:
            logger_warning(body)
        else:
            logger_info(body)

        metrics_record(
            state=severity,
            incomplete_entries=incomplete_entries,
            incomplete_samples=incomplete_samples_count,
            risk_flag_counts=risk_flag_counts,
            labels={"environment": environment},
        )

        sink = self._sink_loader()
        emit = getattr(sink, "emit_feed_health_event", None) if sink is not None else None
        if not callable(emit):
            return

        emit(
            severity="warning" if incomplete_entries else "info",
            title="Risk Journal completeness",
            body=body,
            context={
                "channel": "risk_journal",
                "environment": environment,
                "state": severity,
            },
            payload={
                "channel": "risk_journal",
                "environment": environment,
                "state": severity,
                "incomplete_entries": incomplete_entries,
                "incompleteEntries": incomplete_entries,
                "incomplete_samples": incomplete_samples_count,
                "incompleteSamples": incomplete_samples_count,
                "riskFlagCounts": dict(risk_flag_counts),
                "samples": samples,
            },
        )

    def load_feed_thresholds(self) -> dict[str, float | None]:
        defaults: dict[str, float | None] = {
            "latency_warning_ms": 2500.0,
            "latency_critical_ms": 5000.0,
            "reconnects_warning": 3.0,
            "reconnects_critical": 6.0,
            "downtime_warning_seconds": 30.0,
            "downtime_critical_seconds": 120.0,
        }

        def _normalize(value: float | int | None) -> float | None:
            if value is None:
                return None
            number = float(value)
            if number <= 0:
                return None
            return number

        runtime_config = None
        try:
            runtime_config = self._runtime_config_loader()
        except Exception:
            runtime_config = None
        if runtime_config is not None:
            observability = getattr(runtime_config, "observability", None)
            feed_sla = getattr(observability, "feed_sla", None) if observability else None
            if feed_sla is not None:
                defaults.update(
                    {
                        "latency_warning_ms": _normalize(
                            getattr(feed_sla, "latency_warning_ms", None)
                        )
                        or defaults["latency_warning_ms"],
                        "latency_critical_ms": _normalize(
                            getattr(feed_sla, "latency_critical_ms", None)
                        )
                        or defaults["latency_critical_ms"],
                        "reconnects_warning": _normalize(
                            getattr(feed_sla, "reconnects_warning", None)
                        )
                        or defaults["reconnects_warning"],
                        "reconnects_critical": _normalize(
                            getattr(feed_sla, "reconnects_critical", None)
                        )
                        or defaults["reconnects_critical"],
                        "downtime_warning_seconds": _normalize(
                            getattr(feed_sla, "downtime_warning_seconds", None)
                        )
                        or defaults["downtime_warning_seconds"],
                        "downtime_critical_seconds": _normalize(
                            getattr(feed_sla, "downtime_critical_seconds", None)
                        )
                        or defaults["downtime_critical_seconds"],
                    }
                )

        def _env_float(name: str, current: float | None) -> float | None:
            raw = os.environ.get(name)
            if raw is None or str(raw).strip() == "":
                return current
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return current
            if value <= 0:
                return None
            return value

        def _env_int(name: str, current: float | None) -> float | None:
            raw = os.environ.get(name)
            if raw is None or str(raw).strip() == "":
                return current
            try:
                value = int(float(raw))
            except (TypeError, ValueError):
                return current
            if value <= 0:
                return None
            return float(value)

        defaults["latency_warning_ms"] = _env_float(
            "BOT_CORE_UI_FEED_LATENCY_P95_WARNING_MS",
            defaults["latency_warning_ms"],
        )
        defaults["latency_critical_ms"] = _env_float(
            "BOT_CORE_UI_FEED_LATENCY_P95_CRITICAL_MS",
            defaults["latency_critical_ms"],
        )
        defaults["reconnects_warning"] = _env_int(
            "BOT_CORE_UI_FEED_RECONNECT_WARNING",
            defaults["reconnects_warning"],
        )
        defaults["reconnects_critical"] = _env_int(
            "BOT_CORE_UI_FEED_RECONNECT_CRITICAL",
            defaults["reconnects_critical"],
        )
        defaults["downtime_warning_seconds"] = _env_float(
            "BOT_CORE_UI_FEED_DOWNTIME_WARNING_SECONDS",
            defaults["downtime_warning_seconds"],
        )
        defaults["downtime_critical_seconds"] = _env_float(
            "BOT_CORE_UI_FEED_DOWNTIME_CRITICAL_SECONDS",
            defaults["downtime_critical_seconds"],
        )
        return defaults

    def reload_feed_thresholds(self) -> dict[str, float | None]:
        loaded = self.load_feed_thresholds()
        self._feed_thresholds.clear()
        self._feed_thresholds.update(loaded)
        return self._feed_thresholds

    def record_feed_alert(
        self,
        *,
        severity: str,
        state: str,
        metric: str,
        label: str,
        unit: str,
        value: float | None,
        warning: float | None,
        critical: float | None,
        adapter: str,
        status: str,
        reconnects: int,
        downtime_seconds: float,
        latency_p95: float | None,
        last_error: str,
        router: object | None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entry: dict[str, object] = {
            "id": f"{metric}:{int(time.time() * 1000)}",
            "timestamp": timestamp,
            "severity": severity,
            "state": state,
            "metric": metric,
            "label": label,
            "unit": unit,
            "value": value,
            "formattedValue": self.format_feed_metric(value, unit),
            "warning": warning,
            "critical": critical,
            "adapter": adapter,
            "status": status,
            "reconnects": reconnects,
            "downtimeSeconds": downtime_seconds,
            "latencyP95": latency_p95,
        }
        if last_error:
            entry["lastError"] = last_error
        self._feed_alert_history.appendleft(entry)
        self._history_changed()
        self.refresh_alert_channels(router)

    def refresh_alert_channels(self, router: object | None) -> None:
        channels: list[dict[str, object]] = []
        try:
            health_snapshot = (
                router.health_snapshot() if router and hasattr(router, "health_snapshot") else {}
            )
        except Exception:
            health_snapshot = {}
        if isinstance(health_snapshot, Mapping):
            for name, payload in health_snapshot.items():
                record = {"name": str(name)}
                if isinstance(payload, Mapping):
                    for key, value in payload.items():
                        record[str(key)] = value
                channels.append(record)
        if channels != self._feed_alert_channels:
            self._feed_alert_channels = channels
            self._channels_changed()

"""Automatyzacja self-healing po ćwiczeniach failover Stage6."""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Protocol, Sequence

from bot_core.security.signing import build_hmac_signature

from .drill import FailoverDrillSummary


_REPORT_SCHEMA = "stage6.resilience.self_healing.report"
_REPORT_SIGNATURE_SCHEMA = "stage6.resilience.self_healing.report.signature"
_SCHEMA_VERSION = "1.0"


def _timestamp() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} powinien być obiektem mapującym")
    return value  # type: ignore[return-value]


def _ensure_sequence(value: object, *, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{context} nie może być napisem")
    if not isinstance(value, Sequence):
        raise ValueError(f"{context} powinien być sekwencją")
    return value


def _ensure_tags(raw: object, *, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    sequence = _ensure_sequence(raw, context=field)
    tags: list[str] = []
    for item in sequence:
        if not isinstance(item, str):
            raise ValueError(f"Element {field} musi być napisem")
        if not item:
            raise ValueError(f"Element {field} nie może być pusty")
        tags.append(item)
    return tuple(dict.fromkeys(tags))


def _ensure_statuses(raw: object, *, field: str) -> tuple[str, ...]:
    if raw is None:
        return ("failed",)
    sequence = _ensure_sequence(raw, context=field)
    statuses: list[str] = []
    for item in sequence:
        if not isinstance(item, str) or not item:
            raise ValueError(f"Element {field} musi być niepustym napisem")
        statuses.append(item.lower())
    if not statuses:
        raise ValueError(f"Lista {field} nie może być pusta")
    return tuple(dict.fromkeys(statuses))


def _ensure_float(value: object | None, *, field: str, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except Exception as exc:  # noqa: BLE001 - walidacja wejścia
        raise ValueError(f"{field} musi być liczbą") from exc
    if number < 0:
        raise ValueError(f"{field} nie może być ujemne")
    return number


@dataclass(slots=True)
class SelfHealingActionConfig:
    """Deklaracja pojedynczego restartu runtime."""

    module: str
    command: tuple[str, ...] | None
    delay_seconds: float = 0.0
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SelfHealingRule:
    """Reguła mapująca usługę planu failover na czynności self-healing."""

    name: str | None
    service_pattern: str
    statuses: tuple[str, ...]
    actions: tuple[SelfHealingActionConfig, ...]
    severity: str | None
    tags: tuple[str, ...]
    metadata: Mapping[str, object]

    @staticmethod
    def from_mapping(document: Mapping[str, object]) -> "SelfHealingRule":
        service_pattern = document.get("service_pattern")
        if not isinstance(service_pattern, str) or not service_pattern.strip():
            raise ValueError("Reguła musi zawierać pole 'service_pattern'")
        name_value = document.get("name")
        if name_value is not None and (not isinstance(name_value, str) or not name_value.strip()):
            raise ValueError("Pole 'name' musi być niepustym napisem lub None")
        statuses = _ensure_statuses(document.get("statuses"), field="statuses")
        severity_value = document.get("severity")
        if severity_value is not None and (not isinstance(severity_value, str) or not severity_value.strip()):
            raise ValueError("Pole 'severity' musi być niepustym napisem lub None")
        tags = _ensure_tags(document.get("tags"), field="tags")
        metadata_value = document.get("metadata")
        metadata = dict(_ensure_mapping(metadata_value, context="metadata")) if metadata_value else {}

        actions_value = document.get("actions")
        actions_seq = _ensure_sequence(actions_value, context="actions")
        actions: list[SelfHealingActionConfig] = []
        for raw_action in actions_seq:
            mapping = _ensure_mapping(raw_action, context="action")
            module_value = mapping.get("module")
            if not isinstance(module_value, str) or not module_value.strip():
                raise ValueError("Pole 'module' w akcji musi być niepustym napisem")
            command_value = mapping.get("command")
            command: tuple[str, ...] | None
            if command_value is None:
                command = None
            else:
                sequence = _ensure_sequence(command_value, context="command")
                parts: list[str] = []
                for part in sequence:
                    if not isinstance(part, str) or not part:
                        raise ValueError("Pole 'command' musi zawierać niepuste napisy")
                    parts.append(part)
                if not parts:
                    raise ValueError("Pole 'command' nie może być puste")
                command = tuple(parts)
            delay_seconds = _ensure_float(mapping.get("delay_seconds"), field="delay_seconds", default=0.0)
            action_tags = _ensure_tags(mapping.get("tags"), field="action.tags")
            action_metadata_value = mapping.get("metadata")
            action_metadata = (
                dict(_ensure_mapping(action_metadata_value, context="action.metadata"))
                if action_metadata_value
                else {}
            )
            actions.append(
                SelfHealingActionConfig(
                    module=module_value,
                    command=command,
                    delay_seconds=delay_seconds,
                    tags=action_tags,
                    metadata=action_metadata,
                )
            )
        if not actions:
            raise ValueError("Reguła self-healing musi zawierać co najmniej jedną akcję")
        return SelfHealingRule(
            name=name_value if isinstance(name_value, str) else None,
            service_pattern=service_pattern,
            statuses=statuses,
            actions=tuple(actions),
            severity=severity_value if isinstance(severity_value, str) else None,
            tags=tags,
            metadata=metadata,
        )


def load_self_healing_rules(path: Path) -> tuple[SelfHealingRule, ...]:
    """Ładuje konfigurację self-healing z pliku JSON."""

    path = path.expanduser()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # noqa: BLE001 - komunikat użytkownika
        raise ValueError(f"Nie znaleziono konfiguracji self-healing: {path}") from exc
    except json.JSONDecodeError as exc:  # noqa: BLE001 - komunikat użytkownika
        raise ValueError(f"Konfiguracja self-healing nie jest poprawnym JSON: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ValueError("Konfiguracja self-healing musi być obiektem JSON")
    rules_value = raw.get("rules")
    rules_seq = _ensure_sequence(rules_value, context="rules")
    rules: list[SelfHealingRule] = []
    for item in rules_seq:
        mapping = _ensure_mapping(item, context="rule")
        rules.append(SelfHealingRule.from_mapping(mapping))
    if not rules:
        raise ValueError("Konfiguracja self-healing nie zawiera reguł")
    return tuple(rules)


@dataclass(slots=True)
class SelfHealingAction:
    """Akcja self-healing wygenerowana na podstawie wyników drillu."""

    service: str
    service_status: str
    rule_name: str | None
    module: str
    command: tuple[str, ...] | None
    delay_seconds: float
    severity: str | None
    tags: tuple[str, ...]
    metadata: Mapping[str, object]
    reason: str
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "service": self.service,
            "service_status": self.service_status,
            "rule_name": self.rule_name,
            "module": self.module,
            "command": list(self.command) if self.command else None,
            "delay_seconds": self.delay_seconds,
            "severity": self.severity,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "reason": self.reason,
            "issues": list(self.issues),
        }
        return payload


@dataclass(slots=True)
class SelfHealingPlan:
    """Plan self-healing dla drillu failover."""

    drill_name: str
    generated_at: str
    summary_status: str
    actions: tuple[SelfHealingAction, ...]
    metadata: Mapping[str, object]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "drill_name": self.drill_name,
            "generated_at": self.generated_at,
            "summary_status": self.summary_status,
            "actions": [action.to_dict() for action in self.actions],
            "metadata": dict(self.metadata),
        }


def build_self_healing_plan(
    summary: FailoverDrillSummary,
    rules: Sequence[SelfHealingRule],
) -> SelfHealingPlan:
    """Tworzy plan self-healing dopasowując reguły do wyników drillu."""

    actions: list[SelfHealingAction] = []
    for service in summary.services:
        for rule in rules:
            if not fnmatch(service.name, rule.service_pattern):
                continue
            if service.status.lower() not in rule.statuses:
                continue
            base_tags = tuple(dict.fromkeys(rule.tags))
            issues = tuple(service.issues)
            reason_parts = [f"status={service.status}"]
            if issues:
                reason_parts.append("issues=" + " | ".join(issues))
            reason = "; ".join(reason_parts)
            for action_cfg in rule.actions:
                combined_tags = tuple(dict.fromkeys((*base_tags, *action_cfg.tags)))
                metadata: MutableMapping[str, object] = {
                    "rule": rule.name,
                    "service_pattern": rule.service_pattern,
                }
                metadata.update(rule.metadata)
                metadata.update(action_cfg.metadata)
                actions.append(
                    SelfHealingAction(
                        service=service.name,
                        service_status=service.status,
                        rule_name=rule.name,
                        module=action_cfg.module,
                        command=action_cfg.command,
                        delay_seconds=action_cfg.delay_seconds,
                        severity=rule.severity,
                        tags=combined_tags,
                        metadata=dict(metadata),
                        reason=reason,
                        issues=issues,
                    )
                )
    plan = SelfHealingPlan(
        drill_name=summary.drill_name,
        generated_at=_timestamp(),
        summary_status=summary.status,
        actions=tuple(actions),
        metadata=dict(summary.metadata),
    )
    return plan


@dataclass(slots=True)
class SelfHealingExecution:
    """Wynik wykonania akcji self-healing."""

    action: SelfHealingAction
    status: str
    started_at: str | None
    completed_at: str | None
    exit_code: int | None
    output: str | None
    error: str | None
    notes: str | None = None

    def to_dict(self) -> Mapping[str, object]:
        return {
            "action": self.action.to_dict(),
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "exit_code": self.exit_code,
            "output": self.output,
            "error": self.error,
            "notes": self.notes,
        }


@dataclass(slots=True)
class SelfHealingReport:
    """Podsumowanie planu lub wykonania self-healing."""

    drill_name: str
    generated_at: str
    mode: str
    status: str
    actions: tuple[SelfHealingExecution, ...]
    metadata: Mapping[str, object]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "schema": _REPORT_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "drill_name": self.drill_name,
            "generated_at": self.generated_at,
            "mode": self.mode,
            "status": self.status,
            "actions": [entry.to_dict() for entry in self.actions],
            "metadata": dict(self.metadata),
        }


class SelfHealingExecutor(Protocol):
    """Prosty interfejs restartu modułów runtime."""

    def __call__(self, action: SelfHealingAction) -> SelfHealingExecution:  # pragma: no cover - protokół
        raise NotImplementedError


def summarize_self_healing_plan(plan: SelfHealingPlan) -> SelfHealingReport:
    """Zamienia plan w raport trybu planowania (bez wykonania)."""

    if not plan.actions:
        status = "noop"
    else:
        status = "planned"
    entries = tuple(
        SelfHealingExecution(
            action=action,
            status="planned",
            started_at=None,
            completed_at=None,
            exit_code=None,
            output=None,
            error=None,
            notes="plan_only",
        )
        for action in plan.actions
    )
    return SelfHealingReport(
        drill_name=plan.drill_name,
        generated_at=_timestamp(),
        mode="plan",
        status=status,
        actions=entries,
        metadata=dict(plan.metadata),
    )


class SubprocessSelfHealingExecutor:
    """Wykonuje restart modułu uruchamiając polecenie systemowe."""

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        capture_output: bool = True,
    ) -> None:
        self._env = dict(env) if env is not None else None
        self._cwd = cwd
        self._timeout = timeout
        self._capture_output = capture_output

    def __call__(self, action: SelfHealingAction) -> SelfHealingExecution:
        if not action.command:
            return SelfHealingExecution(
                action=action,
                status="skipped",
                started_at=None,
                completed_at=None,
                exit_code=None,
                output=None,
                error="Brak polecenia restartu",
                notes="no_command",
            )
        started_at = _timestamp()
        try:
            result = subprocess.run(
                action.command,
                check=False,
                capture_output=self._capture_output,
                text=True,
                env=self._env,
                cwd=str(self._cwd) if self._cwd else None,
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001 - komunikat o błędzie
            return SelfHealingExecution(
                action=action,
                status="error",
                started_at=started_at,
                completed_at=_timestamp(),
                exit_code=None,
                output=None,
                error=str(exc),
                notes="executor_exception",
            )
        completed_at = _timestamp()
        status = "success" if result.returncode == 0 else "failed"
        output_text = result.stdout if self._capture_output else None
        error_text = result.stderr if self._capture_output else None
        return SelfHealingExecution(
            action=action,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            exit_code=result.returncode,
            output=output_text,
            error=error_text,
            notes=None,
        )


def execute_self_healing_plan(
    plan: SelfHealingPlan,
    executor: SelfHealingExecutor,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> SelfHealingReport:
    """Uruchamia plan self-healing używając dostarczonego wykonawcy."""

    entries: list[SelfHealingExecution] = []
    overall_status = "noop"
    if plan.actions:
        overall_status = "success"

    for action in plan.actions:
        if action.delay_seconds > 0:
            sleep(action.delay_seconds)
        result = executor(action)
        entries.append(result)
        if result.status in {"failed", "error"}:
            overall_status = "failed"
        elif result.status == "skipped" and overall_status == "success":
            overall_status = "warning"
    return SelfHealingReport(
        drill_name=plan.drill_name,
        generated_at=_timestamp(),
        mode="execute",
        status=overall_status,
        actions=tuple(entries),
        metadata=dict(plan.metadata),
    )


def write_self_healing_report(report: SelfHealingReport, output_path: Path) -> Mapping[str, object]:
    """Zapisuje raport self-healing do pliku JSON."""

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_self_healing_signature(
    report_payload: Mapping[str, object],
    output_path: Path,
    *,
    key: bytes,
    key_id: str | None = None,
    target: str | None = None,
) -> Mapping[str, object]:
    """Zapisuje podpis HMAC raportu self-healing."""

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": _REPORT_SIGNATURE_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "signed_at": _timestamp(),
        "target": target or output_path.name,
        "signature": build_hmac_signature(report_payload, key=key, key_id=key_id),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "SelfHealingActionConfig",
    "SelfHealingRule",
    "load_self_healing_rules",
    "SelfHealingAction",
    "SelfHealingPlan",
    "build_self_healing_plan",
    "SelfHealingExecution",
    "SelfHealingReport",
    "summarize_self_healing_plan",
    "SubprocessSelfHealingExecutor",
    "execute_self_healing_plan",
    "write_self_healing_report",
    "write_self_healing_signature",
]


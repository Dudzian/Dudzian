"""Kontroler odpowiedzialny za mapowanie alertów guardrail na runbooki."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, MutableMapping

import yaml
from PySide6.QtCore import QObject, Property, Signal, Slot

from core.reporting.guardrails_reporter import (
    GuardrailLogRecord,
    GuardrailQueueSummary,
    GuardrailReport,
    GuardrailReportEndpoint,
)
from ui.backend.logging import get_runbook_logger


_DEFAULT_ACTION_DIRECTORY = Path("scripts/runbook_actions")


@dataclass(frozen=True)
class RunbookAction:
    """Opis pojedynczej akcji automatycznej powiązanej z runbookiem."""

    identifier: str
    label: str
    script: Path
    description: str = ""
    confirm: str | None = None


@dataclass(frozen=True)
class RunbookMetadata:
    """Struktura metadanych runbooka."""

    identifier: str
    title: str
    path: Path
    manual_steps: tuple[str, ...]
    actions: tuple[RunbookAction, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "title": self.title,
            "path": str(self.path),
            "manualSteps": list(self.manual_steps),
            "automaticActions": [
                {
                    "id": action.identifier,
                    "label": action.label,
                    "description": action.description,
                    "confirmMessage": action.confirm or "",
                }
                for action in self.actions
            ],
        }


def _now_local_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class RunbookController(QObject):
    """Udostępnia QML listę alertów oraz przypisane runbooki."""

    alertsChanged = Signal()
    lastUpdatedChanged = Signal()
    errorMessageChanged = Signal()
    actionStatusChanged = Signal()

    def __init__(
        self,
        *,
        report_endpoint: GuardrailReportEndpoint | None = None,
        runbook_directory: str | Path | None = None,
        actions_directory: str | Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._endpoint = report_endpoint or GuardrailReportEndpoint()
        self._alerts: list[dict[str, object]] = []
        self._last_updated = ""
        self._error_message = ""
        self._action_status = ""
        self._actions_directory = Path(actions_directory).expanduser() if actions_directory else _DEFAULT_ACTION_DIRECTORY
        self._logger = get_runbook_logger()
        self._runbooks = self._load_runbooks(runbook_directory)

    # ------------------------------------------------------------------
    @Property("QVariantList", notify=alertsChanged)
    def alerts(self) -> list[dict[str, object]]:  # type: ignore[override]
        return list(self._alerts)

    @Property(str, notify=lastUpdatedChanged)
    def lastUpdated(self) -> str:  # type: ignore[override]
        return self._last_updated

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return self._error_message

    @Property(str, notify=actionStatusChanged)
    def actionStatus(self) -> str:  # type: ignore[override]
        return self._action_status

    # ------------------------------------------------------------------
    @Slot(result=bool)
    def refreshAlerts(self) -> bool:
        """Aktualizuje listę alertów na podstawie bieżącego raportu guardrail."""

        try:
            report = self._endpoint.build_report()
        except Exception as exc:  # pragma: no cover - diagnostyka środowiska
            self._error_message = str(exc)
            self.errorMessageChanged.emit()
            return False

        self._error_message = ""
        self.errorMessageChanged.emit()

        self._alerts = self._map_alerts(report)
        self.alertsChanged.emit()

        self._last_updated = _now_local_iso()
        self.lastUpdatedChanged.emit()
        return True

    @Slot(str, str, result=bool)
    def runAction(self, runbook_id: str, action_id: str) -> bool:
        """Uruchamia wskazaną akcję automatyczną dla danego runbooka."""

        if not runbook_id or not action_id:
            return False

        metadata = self._runbooks.get(runbook_id)
        if metadata is None:
            self._set_action_status(
                json.dumps(
                    {
                        "status": "error",
                        "message": f"Nie znaleziono runbooka {runbook_id}",
                    },
                    ensure_ascii=False,
                )
            )
            return False

        action = next((item for item in metadata.actions if item.identifier == action_id), None)
        if action is None:
            self._set_action_status(
                json.dumps(
                    {
                        "status": "error",
                        "message": f"Nie znaleziono akcji {action_id}",
                    },
                    ensure_ascii=False,
                )
            )
            return False

        script_path = self._resolve_action_script(action.script)
        if not script_path.exists():
            self._logger.error("Brak skryptu akcji runbooka: %s", script_path)
            self._set_action_status(
                json.dumps(
                    {
                        "status": "error",
                        "message": f"Skrypt akcji {action_id} jest niedostępny",
                    },
                    ensure_ascii=False,
                )
            )
            return False

        try:
            self._logger.info(
                "Uruchamiam akcję runbooka", extra={"runbook_id": runbook_id, "action_id": action_id, "script": str(script_path)}
            )
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            self._logger.error(
                "Skrypt runbooka zakończył się błędem", extra={"runbook_id": runbook_id, "action_id": action_id, "returncode": exc.returncode}
            )
            self._set_action_status(
                json.dumps(
                    {
                        "status": "error",
                        "message": exc.stderr or str(exc),
                    },
                    ensure_ascii=False,
                )
            )
            return False
        except OSError as exc:  # pragma: no cover - błędy środowiska
            self._logger.exception("Nie udało się uruchomić akcji runbooka")
            self._set_action_status(
                json.dumps(
                    {
                        "status": "error",
                        "message": str(exc),
                    },
                    ensure_ascii=False,
                )
            )
            return False

        payload = {
            "status": "success",
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "runbook_id": runbook_id,
            "action_id": action_id,
        }
        self._set_action_status(json.dumps(payload, ensure_ascii=False))
        self._logger.info(
            "Akcja runbooka zakończona sukcesem",
            extra={"runbook_id": runbook_id, "action_id": action_id, "stdout": result.stdout.strip()},
        )
        return True

    @Slot(str, result=bool)
    def openRunbook(self, path: str) -> bool:
        """Zwraca ``True`` gdy ścieżka do runbooka istnieje."""

        if not path:
            return False
        resolved = Path(path).expanduser()
        return resolved.exists()

    # ------------------------------------------------------------------
    def _map_alerts(self, report: GuardrailReport) -> list[dict[str, object]]:
        alerts: list[dict[str, object]] = []
        used_ids: set[str] = set()

        for summary in report.summaries:
            entry = self._summary_to_alert(summary)
            if entry is None:
                continue
            if entry["id"] in used_ids:
                continue
            used_ids.add(entry["id"])
            alerts.append(entry)

        for record in report.logs:
            entry = self._log_to_alert(record)
            if entry is None or entry["id"] in used_ids:
                continue
            used_ids.add(entry["id"])
            alerts.append(entry)

        for index, recommendation in enumerate(report.recommendations):
            entry = self._recommendation_to_alert(recommendation, index)
            if entry is None or entry["id"] in used_ids:
                continue
            used_ids.add(entry["id"])
            alerts.append(entry)

        return alerts

    # ------------------------------------------------------------------
    def _summary_to_alert(self, summary: GuardrailQueueSummary) -> dict[str, object] | None:
        severity = summary.severity()
        if severity == "normal":
            return None
        message = (
            "Kolejka {queue} w środowisku {env} zgłosiła {timeouts:.0f} timeoutów i "
            "{waits:.0f} oczekiwań na limit.".format(
                queue=summary.queue,
                env=summary.environment,
                timeouts=summary.timeout_total,
                waits=summary.rate_limit_wait_total,
            )
        )
        runbook = self._select_runbook(message, severity)
        return {
            "id": f"summary:{summary.environment}:{summary.queue}",
            "source": "summary",
            "severity": severity,
            "message": message,
            "environment": summary.environment,
            "queue": summary.queue,
            "runbookTitle": runbook.get("title", ""),
            "runbookPath": runbook.get("path", ""),
            "manualSteps": runbook.get("manualSteps", []),
            "automaticActions": runbook.get("automaticActions", []),
            "runbookId": runbook.get("id", ""),
            "timestamp": "",
        }

    def _log_to_alert(self, record: GuardrailLogRecord) -> dict[str, object] | None:
        severity = "error" if record.level == "ERROR" else "warning"
        message = record.message or record.event
        runbook = self._select_runbook(message, severity)
        return {
            "id": f"log:{record.timestamp.isoformat()}:{record.event}",
            "source": "log",
            "severity": severity,
            "message": message,
            "environment": record.metadata.get("environment", "") if record.metadata else "",
            "queue": record.metadata.get("queue", "") if record.metadata else "",
            "runbookTitle": runbook.get("title", ""),
            "runbookPath": runbook.get("path", ""),
            "manualSteps": runbook.get("manualSteps", []),
            "automaticActions": runbook.get("automaticActions", []),
            "runbookId": runbook.get("id", ""),
            "timestamp": record.timestamp.isoformat(),
        }

    def _recommendation_to_alert(self, recommendation: str, index: int) -> dict[str, object] | None:
        message = recommendation.strip()
        if not message:
            return None
        runbook = self._select_runbook(message, "info")
        return {
            "id": f"recommendation:{index}",
            "source": "recommendation",
            "severity": "info",
            "message": message,
            "environment": "",
            "queue": "",
            "runbookTitle": runbook.get("title", ""),
            "runbookPath": runbook.get("path", ""),
            "manualSteps": runbook.get("manualSteps", []),
            "automaticActions": runbook.get("automaticActions", []),
            "runbookId": runbook.get("id", ""),
            "timestamp": "",
        }

    # ------------------------------------------------------------------
    def _select_runbook(self, text: str, severity: str) -> Mapping[str, str]:
        normalized = text.lower()
        for keywords, identifier in _KEYWORD_MAP:
            if any(keyword in normalized for keyword in keywords):
                return self._as_payload(identifier)
        if severity == "error":
            return self._as_payload("strategy_incident_playbook")
        if severity == "warning":
            return self._as_payload("autotrade_threshold_calibration")
        if self._runbooks:
            default_identifier = next(iter(self._runbooks))
            return self._as_payload(default_identifier)
        return {}

    # ------------------------------------------------------------------
    def _load_runbooks(self, directory: str | Path | None) -> MutableMapping[str, RunbookMetadata]:
        search_dir = Path(directory).expanduser() if directory else Path(__file__).resolve().parents[2] / "docs" / "operations" / "runbooks"
        results: MutableMapping[str, RunbookMetadata] = {}
        if not search_dir.exists():
            return results

        metadata_dir = search_dir / "metadata"
        metadata_map = self._load_metadata(metadata_dir)

        for path in sorted(search_dir.glob("*.md")):
            identifier = path.stem
            title = self._extract_title(path)
            manual_steps: Iterable[str]
            actions: Iterable[RunbookAction]
            metadata = metadata_map.get(identifier)
            if metadata:
                manual_steps = metadata.get("manual_steps", [])
                actions = metadata.get("actions", [])
            else:
                manual_steps = []
                actions = []
            results[identifier] = RunbookMetadata(
                identifier=identifier,
                title=title,
                path=path,
                manual_steps=tuple(manual_steps),
                actions=tuple(actions),
            )
        return results

    def _load_metadata(self, directory: Path) -> dict[str, dict[str, Iterable[Any]]]:
        if not directory.exists():
            return {}

        payload: dict[str, dict[str, Iterable[Any]]] = {}
        for path in sorted(directory.glob("*.yml")):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:  # pragma: no cover - błędy składniowe YAML
                logging.getLogger(__name__).warning("Nie udało się sparsować metadanych runbooka %s: %s", path, exc)
                continue

            identifier = str(raw.get("id") or path.stem)
            manual_steps = tuple(str(step) for step in raw.get("manual_steps", []) if step)
            actions = tuple(self._parse_action(identifier, entry, path) for entry in raw.get("automatic_actions", []) if entry)
            payload[identifier] = {
                "manual_steps": manual_steps,
                "actions": tuple(action for action in actions if action is not None),
            }
        return payload

    def _parse_action(self, runbook_id: str, entry: Mapping[str, Any], origin: Path) -> RunbookAction | None:
        identifier = str(entry.get("id") or "").strip()
        label = str(entry.get("label") or "").strip()
        script_name = str(entry.get("script") or "").strip()
        if not identifier or not label or not script_name:
            logging.getLogger(__name__).warning(
                "Pominięto niepełną definicję akcji w %s dla runbooka %s", origin, runbook_id
            )
            return None
        description = str(entry.get("description") or "").strip()
        confirm = entry.get("confirm")
        confirm_message = str(confirm).strip() if confirm else None
        script_path = Path(script_name)
        return RunbookAction(identifier=identifier, label=label, script=script_path, description=description, confirm=confirm_message)

    @staticmethod
    def _extract_title(path: Path) -> str:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    return stripped.lstrip("# ")
        except OSError:  # pragma: no cover - błędy IO
            return path.stem
        return path.stem.replace("_", " ")

    def _resolve_action_script(self, script: Path) -> Path:
        if script.is_absolute():
            return script
        candidate = self._actions_directory / script
        return candidate

    def _as_payload(self, identifier: str) -> Mapping[str, object]:
        metadata = self._runbooks.get(identifier)
        if metadata is None:
            return {}
        payload = metadata.to_payload()
        payload["id"] = identifier
        return payload

    def _set_action_status(self, value: str) -> None:
        self._action_status = value
        self.actionStatusChanged.emit()


_KEYWORD_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("timeout", "awarie", "guardrail"), "strategy_incident_playbook"),
    (("limit", "rate", "oczekiw"), "autotrade_threshold_calibration"),
    (("licenc", "fingerprint", "oem"), "oem_license_provisioning"),
)


__all__ = ["RunbookController", "RunbookAction", "RunbookMetadata"]

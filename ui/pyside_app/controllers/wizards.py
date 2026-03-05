"""Kontroler odpowiedzialny za kreatory trybów pracy w PySide6."""

from __future__ import annotations


import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping

import yaml
from PySide6.QtCore import QObject, Property, Signal, Slot

from bot_core.config.loader import load_runtime_app_config
from bot_core.runtime.cloud_profiles import resolve_runtime_cloud_client

if TYPE_CHECKING:  # pragma: no cover - typy tylko dla mypy
    from ui.backend.runtime_service import RuntimeService  # type: ignore
    from ..config import UiAppConfig

_LOGGER = logging.getLogger(__name__)


class ModeWizardController(QObject):
    """Zarządza definicjami kreatorów i rekomendacjami AI."""

    modesChanged = Signal()
    recommendationChanged = Signal()
    activeModeChanged = Signal()
    resultsChanged = Signal()

    def __init__(
        self,
        runtime_service: QObject,
        ui_config: "UiAppConfig",
        *,
        definitions_path: Path | None = None,
        storage_path: Path | None = None,
        runtime_config_path: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        repo_root = Path(__file__).resolve().parents[3]
        self._runtime_service = runtime_service
        self._ui_config = ui_config
        self._definitions_path = (
            definitions_path or repo_root / "config" / "ui" / "mode_wizards"
        ).expanduser()
        self._storage_path = (
            storage_path or repo_root / "var" / "ui_mode_wizard_state.json"
        ).expanduser()
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage: MutableMapping[str, Any] = self._load_storage()
        self._modes = self._load_mode_definitions()
        self._mode_lookup = {entry["id"]: entry for entry in self._modes if entry.get("id")}
        self._active_mode = str(self._storage.get("active_mode") or "")
        if not self._active_mode or self._active_mode not in self._mode_lookup:
            self._active_mode = self._modes[0]["id"] if self._modes else ""
        self._storage["active_mode"] = self._active_mode
        self._ai_profiles = self._load_ai_profiles(runtime_config_path)
        self._recommendation = str(self._storage.get("last_recommendation") or self._active_mode)
        self._persist_storage()
        if hasattr(self._runtime_service, "decisionsChanged"):
            try:
                self._runtime_service.decisionsChanged.connect(self._handle_decisions)
            except Exception:  # pragma: no cover - środowiska testowe bez sygnału
                _LOGGER.debug("Runtime service nie wspiera sygnału decisionsChanged")
        self._handle_decisions()

    @Property("QVariantList", notify=modesChanged)
    def modes(self) -> list[dict[str, Any]]:  # pragma: no cover - wykorzystywane w QML
        return [dict(mode) for mode in self._modes]

    @Property(str, notify=activeModeChanged)
    def activeModeId(self) -> str:  # pragma: no cover - getter dla QML
        return self._active_mode

    @Property(str, notify=recommendationChanged)
    def recommendedModeId(self) -> str:  # pragma: no cover
        return self._recommendation or self._active_mode

    @Property("QVariantMap", notify=recommendationChanged)
    def recommendedModeSummary(self) -> dict[str, Any]:  # pragma: no cover - QML binding
        mode = self._mode_lookup.get(self.recommendedModeId)
        if not mode:
            return {}
        payload = {key: value for key, value in mode.items() if key != "steps"}
        payload["stepsCount"] = len(mode.get("steps", []))
        payload["ai_profiles"] = self._ai_profiles
        return payload

    @Property("QVariantMap", notify=resultsChanged)
    def savedResults(self) -> dict[str, Any]:  # pragma: no cover - QML binding
        return dict(self._storage.get("results", {}))

    @Slot(str)
    def setActiveMode(self, mode_id: str) -> None:
        mode_id = (mode_id or "").strip()
        if not mode_id:
            return
        if mode_id not in self._mode_lookup:
            _LOGGER.debug("Nieznany tryb %s", mode_id)
            return
        if self._active_mode == mode_id:
            return
        self._active_mode = mode_id
        self._storage["active_mode"] = mode_id
        self._persist_storage()
        self.activeModeChanged.emit()

    @Slot(result="QVariantMap")
    def aiProfiles(self) -> dict[str, Any]:  # pragma: no cover - wykorzystywane w QML
        return dict(self._ai_profiles)

    @Slot(str, result="QVariantMap")
    def modeDetails(self, mode_id: str) -> dict[str, Any]:  # pragma: no cover
        mode = self._mode_lookup.get(mode_id)
        if not mode:
            return {}
        return dict(mode)

    @Slot(str, result="QVariantMap")
    def savedAnswers(self, mode_id: str) -> dict[str, Any]:  # pragma: no cover - QML bridge
        results = self._storage.get("results", {})
        entry = results.get(mode_id, {}) if isinstance(results, Mapping) else {}
        return dict(entry)

    @Slot(str, "QVariantMap")
    def saveResult(self, mode_id: str, answers: Mapping[str, Any] | None = None) -> None:
        mode_id = (mode_id or "").strip()
        if not mode_id:
            return
        if mode_id not in self._mode_lookup:
            return
        results = self._storage.setdefault("results", {})
        if not isinstance(results, MutableMapping):
            self._storage["results"] = {}
            results = self._storage["results"]
        payload = {str(key): answers[key] for key in (answers or {})}
        results[mode_id] = payload
        self._persist_storage()
        self.resultsChanged.emit()

    def _load_mode_definitions(self) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        if not self._definitions_path.exists():
            _LOGGER.warning("Brak katalogu z kreatorami UI: %s", self._definitions_path)
            return definitions
        for entry in sorted(self._definitions_path.glob("*.yaml")):
            try:
                data = yaml.safe_load(entry.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # pragma: no cover - diagnostyka
                _LOGGER.error("Nie udało się odczytać %s: %s", entry, exc)
                continue
            if not isinstance(data, Mapping):
                continue
            mode_id = str(data.get("id") or entry.stem)
            normalized: dict[str, Any] = {
                "id": mode_id,
                "title": data.get("title") or mode_id,
                "icon": data.get("icon") or "mode_wizard",
                "description": data.get("description") or "",
                "badge": data.get("badge") or "",
                "ai_profile_hint": data.get("ai_profile_hint") or "",
                "telemetry_rules": data.get("telemetry_rules") or {},
                "recommendations": data.get("recommendations") or {},
            }
            steps_payload: list[dict[str, Any]] = []
            for step in data.get("steps") or []:
                if not isinstance(step, Mapping):
                    continue
                step_entry: dict[str, Any] = {
                    "id": step.get("id") or f"step-{len(steps_payload)}",
                    "title": step.get("title") or "",
                    "description": step.get("description") or "",
                    "inputs": [],
                }
                for input_cfg in step.get("inputs") or []:
                    if not isinstance(input_cfg, Mapping):
                        continue
                    option_payload: list[dict[str, str]] = []
                    for opt in input_cfg.get("options") or []:
                        if not isinstance(opt, Mapping):
                            continue
                        option_payload.append(
                            {
                                "id": opt.get("id") or opt.get("value") or "option",
                                "label": opt.get("label") or opt.get("id") or "",
                                "helper": opt.get("helper") or "",
                            }
                        )
                    step_entry["inputs"].append(
                        {
                            "id": input_cfg.get("id") or "input",
                            "type": input_cfg.get("type") or "choice",
                            "label": input_cfg.get("label") or "",
                            "options": option_payload,
                        }
                    )
                steps_payload.append(step_entry)
            normalized["steps"] = steps_payload
            definitions.append(normalized)
        return definitions

    def _load_storage(self) -> MutableMapping[str, Any]:
        if not self._storage_path.exists():
            return {"results": {}}
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - błędy IO
            return {"results": {}}
        if not isinstance(data, MutableMapping):
            return {"results": {}}
        data.setdefault("results", {})
        return data

    def _persist_storage(self) -> None:
        try:
            self._storage_path.write_text(
                json.dumps(self._storage, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover - środowiska read-only
            _LOGGER.warning("Nie można zapisać stanu kreatorów: %s", exc)

    def _load_ai_profiles(self, runtime_config_path: Path | None) -> dict[str, Any]:
        runtime_path = runtime_config_path or Path(
            self._ui_config.payload.get("runtime_config_path")
            or Path(__file__).resolve().parents[3] / "config" / "runtime.yaml"
        )
        profiles: dict[str, Any] = {}
        try:
            runtime_config = load_runtime_app_config(runtime_path)
            cloud_section = getattr(runtime_config, "cloud", None)
            if cloud_section:
                for name, profile in (getattr(cloud_section, "profiles", {}) or {}).items():
                    profiles[str(name)] = {
                        "mode": getattr(profile, "mode", "local"),
                        "entrypoint": getattr(profile, "entrypoint", None),
                        "description": getattr(profile, "description", None),
                    }
        except Exception as exc:  # pragma: no cover - konfiguracja opcjonalna
            _LOGGER.debug("Nie udało się wczytać runtime.yaml: %s", exc)
        try:
            selection = resolve_runtime_cloud_client(runtime_path)
            if selection:
                profiles.setdefault(
                    selection.profile_name,
                    {
                        "mode": getattr(selection.profile, "mode", "remote"),
                        "entrypoint": getattr(
                            selection.profile, "entrypoint", selection.profile.entrypoint
                        ),
                        "description": getattr(selection.profile, "description", None),
                    },
                )
        except Exception as exc:  # pragma: no cover - brak profili remote
            _LOGGER.debug("resolve_runtime_cloud_client nie powiodło się: %s", exc)
        return profiles

    def _handle_decisions(self) -> None:
        decisions = list(getattr(self._runtime_service, "decisions", []) or [])
        recommendation = self._compute_recommendation(decisions)
        if recommendation and recommendation != self._recommendation:
            self._recommendation = recommendation
            self._storage["last_recommendation"] = recommendation
            self._persist_storage()
            self.recommendationChanged.emit()

    def _compute_recommendation(self, decisions: list[Mapping[str, Any]]) -> str:
        if not decisions:
            return self._active_mode
        trade_count = 0
        long_count = 0
        short_count = 0
        volatility: list[float] = []
        holding: list[float] = []
        futures_signal = False
        copy_signal = False
        for entry in decisions:
            decision_payload = entry.get("decision") if isinstance(entry, Mapping) else {}
            if isinstance(decision_payload, Mapping) and decision_payload.get("shouldTrade"):
                trade_count += 1
            side = str(entry.get("side") or "").lower()
            if side in {"buy", "long"}:
                long_count += 1
            if side in {"sell", "short"}:
                short_count += 1
            regime = entry.get("marketRegime") if isinstance(entry, Mapping) else {}
            for token in ("volatilityScore", "volatility", "sigma"):
                value = regime.get(token) if isinstance(regime, Mapping) else None
                if value is not None:
                    try:
                        volatility.append(float(value))
                    except Exception:
                        pass
                    break
            hold_value = None
            if isinstance(decision_payload, Mapping):
                hold_value = decision_payload.get("holdingMinutes") or decision_payload.get(
                    "holdingPeriodMinutes"
                )
            if hold_value is not None:
                try:
                    holding.append(float(hold_value))
                except Exception:
                    pass
            env_tokens = " ".join(
                str(entry.get(key) or "")
                for key in ("environment", "portfolio", "strategy", "status")
            ).lower()
            if any(token in env_tokens for token in ("future", "perp", "derivative")):
                futures_signal = True
            ai_payload = entry.get("ai") if isinstance(entry, Mapping) else {}
            if not isinstance(ai_payload, Mapping):
                ai_payload = {}
            ai_signature = " ".join(
                str(ai_payload.get(field) or "") for field in ("strategy", "mode", "label")
            ).lower()
            if "copy" in ai_signature:
                copy_signal = True
        total = len(decisions)
        trade_ratio = trade_count / total if total else 0.0
        avg_vol = sum(volatility) / len(volatility) if volatility else 0.4
        avg_hold = sum(holding) / len(holding) if holding else 30.0
        if futures_signal:
            return "futures" if "futures" in self._mode_lookup else self._active_mode
        if copy_signal:
            return "copy_trading" if "copy_trading" in self._mode_lookup else self._active_mode
        if long_count > 0 and short_count > 0 and avg_vol >= 0.35:
            if "hedge" in self._mode_lookup:
                return "hedge"
        if trade_ratio >= 0.55 or avg_hold <= 15:
            if "scalping" in self._mode_lookup:
                return "scalping"
        if avg_hold >= 45 or avg_vol <= 0.35:
            if "swing" in self._mode_lookup:
                return "swing"
        return self._active_mode or (self._modes[0]["id"] if self._modes else "")


__all__ = ["ModeWizardController"]

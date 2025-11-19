"""Kontroler zarządzania strategiami i marketplace dla PySide6."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from PySide6.QtCore import QObject, Property, Signal, Slot

from bot_core.config.loader import load_runtime_app_config
from bot_core.marketplace import PresetRepository
from bot_core.security.hwid import HwIdProvider
from bot_core.strategies.installer import MarketplaceInstallResult, MarketplacePresetInstaller
from bot_core.ui.api import MarketplaceService

import logging

import yaml

_LOGGER = logging.getLogger(__name__)


def _resolve_repo_path(config_path: Path, presets_path: str | None) -> Path:
    base_dir = config_path.parent
    if not presets_path:
        presets_path = "config/marketplace/presets"
    candidate = Path(presets_path)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _load_signing_keys(config_path: Path, settings: Mapping[str, str] | None) -> dict[str, bytes]:
    keys: dict[str, bytes] = {}
    base_dir = config_path.parent
    if settings:
        for key_id, raw_value in settings.items():
            payload = _coerce_key_payload(raw_value, base_dir)
            if payload is None:
                continue
            keys[str(key_id)] = payload
    default_keys_dir = base_dir / "config" / "marketplace" / "keys"
    if default_keys_dir.exists():
        for key_file in default_keys_dir.glob("*.key"):
            try:
                keys[key_file.stem] = key_file.read_text(encoding="utf-8").strip().encode("utf-8")
            except OSError:
                _LOGGER.debug("Nie udało się wczytać klucza %s", key_file, exc_info=True)
    return keys


def _coerce_key_payload(value: object, base_dir: Path) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    text = str(value).strip()
    if not text:
        return None
    candidate_path = Path(text)
    if not candidate_path.is_absolute():
        candidate_path = (base_dir / candidate_path).resolve()
    if candidate_path.exists():
        try:
            payload = candidate_path.read_bytes()
            return payload.strip() or payload
        except OSError:
            _LOGGER.debug("Nie udało się wczytać klucza z %s", candidate_path, exc_info=True)
    if all(ch in "0123456789abcdefABCDEF" for ch in text) and len(text) % 2 == 0:
        try:
            return bytes.fromhex(text)
        except ValueError:
            pass
    return text.encode("utf-8")


def _resolve_catalog_path(repo_root: Path) -> Path:
    candidate = repo_root.parent
    catalog = candidate / "catalog.yaml"
    if catalog.exists():
        return candidate
    fallback = Path(__file__).resolve().parents[3] / "bot_core" / "strategies" / "marketplace"
    return fallback


def _build_marketplace_service(config_path: Path) -> MarketplaceService | None:
    try:
        runtime_config = load_runtime_app_config(config_path)
    except Exception as exc:  # pragma: no cover - diagnostyka środowisk
        _LOGGER.error("Nie udało się wczytać runtime.yaml: %s", exc)
        return None
    marketplace_settings = runtime_config.marketplace
    if marketplace_settings is None or not marketplace_settings.enabled:
        _LOGGER.info("Marketplace jest wyłączony w konfiguracji runtime")
        return None
    repo_root = _resolve_repo_path(config_path, marketplace_settings.presets_path)
    repository = PresetRepository(repo_root)
    signing_keys = _load_signing_keys(config_path, marketplace_settings.signing_keys)
    meta_root = repository.root / ".meta"
    licenses_dir = meta_root / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    installer = MarketplacePresetInstaller(
        repository,
        catalog_path=_resolve_catalog_path(repo_root),
        licenses_dir=licenses_dir,
        signing_keys=signing_keys or None,
        hwid_provider=HwIdProvider(),
    )
    return MarketplaceService(installer, repository)


class StrategyManagementController(QObject):
    """Udostępnia operacje marketplace i zarządzania strategiami do QML."""

    presetsChanged = Signal()
    statusMessageChanged = Signal()
    busyChanged = Signal()
    bundlePathChanged = Signal()
    cloudRuntimeEnabledChanged = Signal()

    def __init__(
        self,
        *,
        marketplace_service: MarketplaceService | None = None,
        runtime_config_path: str | Path = "config/runtime.yaml",
    ) -> None:
        super().__init__()
        self._runtime_path = Path(runtime_config_path).expanduser()
        self._service = marketplace_service or _build_marketplace_service(self._runtime_path)
        self._presets: list[dict[str, Any]] = []
        self._status_message = ""
        self._busy = False
        config_dir = self._runtime_path.resolve().parent
        project_root = config_dir.parent if config_dir.parent != config_dir else config_dir
        self._project_root = project_root
        self._bundle_root = (self._project_root / "var" / "runtime" / "bundles").resolve()
        self._bundle_root.mkdir(parents=True, exist_ok=True)
        self._last_bundle_path = ""
        self._cloud_runtime_enabled = self._read_cloud_runtime_flag()
        if self._service is None:
            self._status_message = "Marketplace nie jest skonfigurowany"
        else:
            self.refreshMarketplace()

    # ------------------------------------------------------------------
    @Property("QVariantList", notify=presetsChanged)
    def presets(self) -> list[dict[str, Any]]:  # type: ignore[override]
        return list(self._presets)

    # ------------------------------------------------------------------
    @Property(str, notify=statusMessageChanged)
    def statusMessage(self) -> str:  # type: ignore[override]
        return self._status_message

    # ------------------------------------------------------------------
    @Property(bool, notify=busyChanged)
    def busy(self) -> bool:  # type: ignore[override]
        return self._busy

    # ------------------------------------------------------------------
    @Property(str, notify=bundlePathChanged)
    def lastBundlePath(self) -> str:  # type: ignore[override]
        return self._last_bundle_path

    # ------------------------------------------------------------------
    @Property(bool, notify=cloudRuntimeEnabledChanged)
    def cloudRuntimeEnabled(self) -> bool:  # type: ignore[override]
        return self._cloud_runtime_enabled

    # ------------------------------------------------------------------
    @Slot(result=bool)
    def available(self) -> bool:
        return self._service is not None

    # ------------------------------------------------------------------
    def _set_busy(self, value: bool) -> None:
        if self._busy != value:
            self._busy = value
            self.busyChanged.emit()

    # ------------------------------------------------------------------
    def _set_status(self, message: str) -> None:
        if self._status_message != message:
            self._status_message = message
            self.statusMessageChanged.emit()

    # ------------------------------------------------------------------
    def _set_presets(self, payload: list[dict[str, Any]]) -> None:
        self._presets = payload
        self.presetsChanged.emit()

    # ------------------------------------------------------------------
    def _set_last_bundle_path(self, path: str) -> None:
        if self._last_bundle_path != path:
            self._last_bundle_path = path
            self.bundlePathChanged.emit()

    # ------------------------------------------------------------------
    def _set_cloud_runtime_enabled(self, value: bool) -> None:
        if self._cloud_runtime_enabled != value:
            self._cloud_runtime_enabled = value
            self.cloudRuntimeEnabledChanged.emit()

    # ------------------------------------------------------------------
    @Slot()
    def refreshMarketplace(self) -> None:
        if self._service is None:
            self._set_status("Marketplace nieaktywny")
            return
        self._set_busy(True)
        try:
            data = self._service.list_presets_payload()
        except Exception as exc:  # pragma: no cover - diagnostyka
            _LOGGER.error("Błąd odczytu Marketplace: %s", exc)
            self._set_status(f"Błąd odczytu Marketplace: {exc}")
        else:
            self._set_presets(list(data))
            self._set_status(f"Załadowano {len(data)} presetów Marketplace")
        finally:
            self._set_busy(False)

    # ------------------------------------------------------------------
    @Slot(str, result="QVariantMap")
    def presetDetails(self, preset_id: str) -> dict[str, Any]:
        preset_id = (preset_id or "").strip()
        for entry in self._presets:
            if entry.get("presetId") == preset_id:
                return dict(entry)
        return {}

    # ------------------------------------------------------------------
    @Slot(str, str, result="QVariantMap")
    def assignPresetToPortfolio(self, preset_id: str, portfolio_id: str) -> dict[str, Any]:
        if self._service is None:
            return {"success": False, "message": "Marketplace nieaktywny"}
        preset_id = (preset_id or "").strip()
        portfolio_id = (portfolio_id or "").strip()
        if not preset_id or not portfolio_id:
            return {"success": False, "message": "Wymagany preset i portfel"}
        try:
            assigned = self._service.assign_to_portfolio(preset_id, portfolio_id)
        except Exception as exc:  # pragma: no cover - diagnostyka
            return {"success": False, "message": str(exc)}
        self.refreshMarketplace()
        return {
            "success": True,
            "presetId": preset_id,
            "assignedPortfolios": list(assigned),
        }

    # ------------------------------------------------------------------
    @Slot(str, str, result="QVariantMap")
    def activateAndAssign(self, preset_id: str, portfolio_id: str) -> dict[str, Any]:
        if self._service is None:
            return {"success": False, "message": "Marketplace nieaktywny"}
        preset_id = (preset_id or "").strip()
        portfolio_id = (portfolio_id or "").strip()
        if not preset_id:
            return {"success": False, "message": "Identyfikator presetu jest wymagany"}
        self._set_busy(True)
        try:
            result = self._service.install_from_catalog(preset_id)
        except Exception as exc:
            payload = {"success": False, "message": str(exc), "presetId": preset_id}
            self._set_busy(False)
            self._set_status(payload["message"])
            return payload
        assigned: tuple[str, ...] = ()
        assign_error: str | None = None
        if portfolio_id:
            try:
                assigned = self._service.assign_to_portfolio(preset_id, portfolio_id)
            except Exception as exc:  # pragma: no cover - diagnostyka
                assign_error = str(exc)
        self._set_busy(False)
        self.refreshMarketplace()
        payload = _install_result_payload(result)
        payload["assignedPortfolios"] = list(assigned)
        if assign_error:
            payload["assignmentError"] = assign_error
        if payload.get("success"):
            if portfolio_id:
                self._set_status(
                    f"Zainstalowano {preset_id} i przypisano do {portfolio_id}"
                )
            else:
                self._set_status(f"Zainstalowano {preset_id}")
        else:
            self._set_status("Instalacja presetu nie powiodła się")
        return payload

    # ------------------------------------------------------------------
    @Slot(str, "QVariantList", "QVariantMap", result="QVariantMap")
    def createPresetBundle(
        self,
        bundle_name: str,
        entries: Iterable[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        bundle_name = (bundle_name or "").strip()
        normalized_name = bundle_name or f"bundle-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        safe_name = _slugify(normalized_name)
        if not safe_name:
            return {"success": False, "message": "Nazwa bundla jest wymagana"}
        selection = _normalize_bundle_entries(entries)
        if not selection:
            return {"success": False, "message": "Wybierz co najmniej jeden preset"}
        opts = dict(options or {})
        bundle_mode = str(opts.get("bundleMode") or "sequential")
        cloud_enabled = opts.get("cloudEnabled")
        payload: dict[str, Any] = {
            "name": normalized_name,
            "identifier": safe_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "bundle_mode": bundle_mode,
            "cloudEnabled": bool(cloud_enabled) if cloud_enabled is not None else None,
            "presets": selection,
        }
        try:
            destination = self._bundle_root / f"{safe_name}.yaml"
            with destination.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        except OSError as exc:
            _LOGGER.error("Nie udało się zapisać bundla presetów: %s", exc)
            return {"success": False, "message": str(exc)}
        self._set_last_bundle_path(str(destination))
        return {"success": True, "path": str(destination), "bundle": payload}

    # ------------------------------------------------------------------
    @Slot(bool, result="QVariantMap")
    def setCloudRuntimeEnabled(self, enabled: bool) -> dict[str, Any]:
        config = self._load_runtime_config()
        cloud_section = dict(config.get("cloud") or {})
        cloud_section["enabled_signed"] = bool(enabled)
        config["cloud"] = cloud_section
        try:
            self._save_runtime_config(config)
        except OSError as exc:
            _LOGGER.error("Nie udało się zapisać config/runtime.yaml: %s", exc)
            return {"success": False, "message": str(exc)}
        self._set_cloud_runtime_enabled(bool(enabled))
        return {"success": True, "enabled": self._cloud_runtime_enabled}

    # ------------------------------------------------------------------
    def _read_cloud_runtime_flag(self) -> bool:
        config = self._load_runtime_config()
        cloud_section = config.get("cloud") if isinstance(config, Mapping) else None
        if isinstance(cloud_section, Mapping):
            if "enabled_signed" in cloud_section:
                return bool(cloud_section.get("enabled_signed"))
            if "enabled" in cloud_section:
                return bool(cloud_section.get("enabled"))
        return False

    # ------------------------------------------------------------------
    def _load_runtime_config(self) -> dict[str, Any]:
        try:
            if self._runtime_path.exists():
                with self._runtime_path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                    if isinstance(data, Mapping):
                        return dict(data)
        except Exception:
            _LOGGER.debug("Nie udało się wczytać runtime.yaml", exc_info=True)
        return {}

    # ------------------------------------------------------------------
    def _save_runtime_config(self, payload: Mapping[str, Any]) -> None:
        self._runtime_path.parent.mkdir(parents=True, exist_ok=True)
        with self._runtime_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=True)


def _install_result_payload(result: MarketplaceInstallResult) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "presetId": result.preset_id,
        "version": result.version,
        "issues": list(result.issues),
        "warnings": list(result.warnings),
        "signatureVerified": result.signature_verified,
        "fingerprintVerified": result.fingerprint_verified,
        "licensePayload": result.license_payload,
    }


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\-_.]+", "-", value.strip().lower()).strip("-._")
    return normalized


def _normalize_bundle_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, Mapping):
            continue
        preset_id = str(entry.get("presetId") or "").strip()
        if not preset_id:
            continue
        label = str(entry.get("label") or preset_id)
        mode = str(entry.get("mode") or "auto")
        order_value = entry.get("order")
        try:
            order = int(order_value)
        except (TypeError, ValueError):
            order = len(sanitized) + 1
        sanitized.append({
            "presetId": preset_id,
            "label": label,
            "mode": mode,
            "order": order,
        })
    sanitized.sort(key=lambda item: item.get("order", 0))
    for index, entry in enumerate(sanitized, start=1):
        entry["order"] = index
    return sanitized

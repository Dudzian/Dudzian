"""Serwis runtime dostarczający dane dziennika decyzji do QML."""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QObject, Property, Signal, Slot

from bot_core.config import load_core_config
from bot_core.portfolio import resolve_decision_log_config
from bot_core.runtime.journal import TradingDecisionJournal
from .demo_data import load_demo_decisions

try:  # pragma: no cover - moduł może nie być dostępny w wersjach light
    from bot_core.ai import ModelRepository
except Exception:  # pragma: no cover - fallback dla dystrybucji bez komponentu AI
    ModelRepository = None  # type: ignore[assignment]

try:  # pragma: no cover - harmonogram retrainingu jest opcjonalny
    from bot_core.runtime.ai_retrain import CronSchedule
except Exception:  # pragma: no cover - fallback dla środowisk bez retrainingu
    CronSchedule = None  # type: ignore[assignment]

try:  # pragma: no cover - funkcja ładowania runtime może nie być dostępna w starszych gałęziach
    from bot_core.config.loader import load_runtime_app_config
except Exception:  # pragma: no cover - fallback gdy brak unified loadera
    load_runtime_app_config = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - adnotacje tylko w czasie statycznym
    from bot_core.config.models import RuntimeAppConfig

_LOGGER = logging.getLogger(__name__)

DecisionRecord = Mapping[str, str]
DecisionLoader = Callable[[int], Iterable[DecisionRecord]]


@dataclass(slots=True)
class _RuntimeDecisionEntry:
    """Struktura pośrednia używana w konwersji rekordów."""

    event: str
    timestamp: str
    environment: str
    portfolio: str
    risk_profile: str
    schedule: str | None
    strategy: str | None
    symbol: str | None
    side: str | None
    status: str | None
    quantity: str | None
    price: str | None
    market_regime: Mapping[str, object]
    decision: Mapping[str, object]
    ai: Mapping[str, object]
    extras: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "event": self.event,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "portfolio": self.portfolio,
            "riskProfile": self.risk_profile,
            "schedule": self.schedule,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "quantity": self.quantity,
            "price": self.price,
            "marketRegime": dict(self.market_regime),
            "decision": dict(self.decision),
            "ai": dict(self.ai),
            "metadata": dict(self.extras),
        }


def _default_loader(limit: int) -> Iterable[DecisionRecord]:
    """Zapewnia dane demonstracyjne przy pierwszym uruchomieniu."""

    entries = list(load_demo_decisions())
    if not entries:
        return []
    if limit > 0:
        entries = entries[-limit:]
    # Zwracamy w kolejności od najnowszych do najstarszych, aby zachować spójność z dziennikiem
    return reversed(entries)


def _load_from_journal(journal: TradingDecisionJournal, limit: int) -> Iterable[DecisionRecord]:
    exported = list(journal.export())
    if limit > 0:
        exported = exported[-limit:]
    return reversed(exported)


def _normalize_bool(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return value


def _camelize(prefix: str, key: str) -> str:
    suffix = key[len(prefix) :].lstrip("_")
    if not suffix:
        return ""
    parts = [part for part in suffix.split("_") if part]
    if not parts:
        return ""
    head, *tail = parts
    return head + "".join(segment.capitalize() for segment in tail)


_BASE_FIELD_MAP: Mapping[str, str] = {
    "event": "event",
    "timestamp": "timestamp",
    "environment": "environment",
    "portfolio": "portfolio",
    "risk_profile": "risk_profile",
    "schedule": "schedule",
    "strategy": "strategy",
    "symbol": "symbol",
    "side": "side",
    "status": "status",
    "quantity": "quantity",
    "price": "price",
}


def _parse_entry(record: DecisionRecord) -> _RuntimeDecisionEntry:
    base: MutableMapping[str, str | None] = {key: None for key in _BASE_FIELD_MAP.values()}
    decision_payload: MutableMapping[str, object] = {}
    ai_payload: MutableMapping[str, object] = {}
    regime_payload: MutableMapping[str, object] = {}
    extras: MutableMapping[str, object] = {}

    confidence = record.get("confidence")
    latency = record.get("latency_ms")
    if confidence is not None:
        decision_payload["confidence"] = confidence
    if latency is not None:
        decision_payload["latencyMs"] = latency

    for key, value in record.items():
        if key in {"confidence", "latency_ms"}:
            continue
        mapped = _BASE_FIELD_MAP.get(key)
        if mapped is not None:
            base[mapped] = value
            continue
        if key.startswith("decision_"):
            normalized = _camelize("decision_", key)
            if not normalized:
                continue
            payload_value: object = value
            if normalized == "shouldTrade":
                payload_value = _normalize_bool(value)
            decision_payload[normalized] = payload_value
            continue
        if key.startswith("ai_"):
            normalized = _camelize("ai_", key)
            if not normalized:
                continue
            ai_payload[normalized] = value
            continue
        if key.startswith("market_regime"):
            normalized = _camelize("market_regime", key)
            if not normalized:
                continue
            regime_payload[normalized] = value
            continue
        if key == "risk_profile":
            # już zmapowane do base
            continue
        extras[key] = value

    event = str(base["event"] or "")
    timestamp = str(base["timestamp"] or "")
    environment = str(base["environment"] or "")
    portfolio = str(base["portfolio"] or "")
    risk_profile = str(base["risk_profile"] or "")

    schedule = base.get("schedule")
    strategy = base.get("strategy")
    symbol = base.get("symbol")
    side = base.get("side")
    status = base.get("status")
    quantity = base.get("quantity")
    price = base.get("price")

    return _RuntimeDecisionEntry(
        event=event,
        timestamp=timestamp,
        environment=environment,
        portfolio=portfolio,
        risk_profile=risk_profile,
        schedule=schedule,
        strategy=strategy,
        symbol=symbol,
        side=side,
        status=status,
        quantity=quantity,
        price=price,
        market_regime=regime_payload,
        decision=decision_payload,
        ai=ai_payload,
        extras=extras,
    )


class RuntimeService(QObject):
    """Zapewnia QML dostęp do najnowszych decyzji autotradera."""

    decisionsChanged = Signal()
    errorMessageChanged = Signal()
    liveSourceChanged = Signal()
    retrainNextRunChanged = Signal()
    adaptiveStrategySummaryChanged = Signal()

    def __init__(
        self,
        *,
        journal: TradingDecisionJournal | None = None,
        decision_loader: DecisionLoader | None = None,
        parent: QObject | None = None,
        default_limit: int = 20,
        core_config_path: str | os.PathLike[str] | None = None,
        runtime_config_path: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(parent)
        if decision_loader is not None:
            self._loader: DecisionLoader = decision_loader
        elif journal is not None:
            self._loader = lambda limit: _load_from_journal(journal, limit)
        else:
            self._loader = _default_loader
        self._default_limit = max(1, int(default_limit))
        self._decisions: list[dict[str, object]] = []
        self._error_message = ""
        self._core_config_path = Path(core_config_path).expanduser() if core_config_path else None
        self._cached_core_config = None
        self._active_profile: str | None = None
        self._active_log_path: Path | None = None
        self._runtime_config_path = Path(runtime_config_path).expanduser() if runtime_config_path else None
        self._runtime_config_cache: "RuntimeAppConfig | None" = None
        self._retrain_next_run: str = ""
        self._adaptive_summary: str = ""
        try:
            self._update_runtime_metadata(invalidate_cache=False)
        except Exception:  # pragma: no cover - defensywna inicjalizacja
            _LOGGER.debug("Nie udało się zainicjalizować metadanych runtime", exc_info=True)

    # ------------------------------------------------------------------
    @Property("QVariantList", notify=decisionsChanged)
    def decisions(self) -> list[dict[str, object]]:  # type: ignore[override]
        return list(self._decisions)

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return self._error_message

    @Property(str, notify=retrainNextRunChanged)
    def retrainNextRun(self) -> str:  # type: ignore[override]
        return self._retrain_next_run

    @Property(str, notify=adaptiveStrategySummaryChanged)
    def adaptiveStrategySummary(self) -> str:  # type: ignore[override]
        return self._adaptive_summary

    # ------------------------------------------------------------------
    @Slot(int, result="QVariantList")
    def loadRecentDecisions(self, limit: int = 0) -> list[dict[str, object]]:  # type: ignore[override]
        """Pobiera najnowsze decyzje z dziennika."""

        size = int(limit)
        if size <= 0:
            size = self._default_limit
        try:
            raw_entries = list(self._loader(size))
        except Exception as exc:  # pragma: no cover - diagnostyka
            self._error_message = str(exc)
            self.errorMessageChanged.emit()
            return []

        self._error_message = ""
        self.errorMessageChanged.emit()

        parsed: list[dict[str, object]] = []
        for record in raw_entries:
            entry = _parse_entry(record)
            payload = entry.to_payload()
            parsed.append(payload)
        self._decisions = parsed
        self.decisionsChanged.emit()
        self._update_runtime_metadata(invalidate_cache=False)
        return list(self._decisions)

    @Slot()
    def refreshRuntimeMetadata(self) -> None:  # type: ignore[override]
        """Wymusza ponowne wczytanie metadanych retrainingu i presetów adaptacyjnych."""

        self._update_runtime_metadata(invalidate_cache=True)

    # ------------------------------------------------------------------
    @Property(str, notify=liveSourceChanged)
    def activeDecisionLogPath(self) -> str:  # type: ignore[override]
        if self._active_log_path is None:
            return ""
        return str(self._active_log_path)

    @Slot(str, result=bool)
    def attachToLiveDecisionLog(self, profile: str = "") -> bool:  # type: ignore[override]
        """Przełącza loader na rzeczywisty decision log skonfigurowany w core.yaml."""

        try:
            loader, log_path = self._build_live_loader(profile.strip() or None)
        except Exception as exc:  # pragma: no cover - diagnostyka
            _LOGGER.debug("attachToLiveDecisionLog failed", exc_info=True)
            self._error_message = str(exc)
            self.errorMessageChanged.emit()
            return False

        self._loader = loader
        self._active_profile = profile.strip() or None
        self._active_log_path = log_path
        self._error_message = ""
        self.errorMessageChanged.emit()
        self.liveSourceChanged.emit()
        self.loadRecentDecisions(self._default_limit)
        return True

    # ------------------------------------------------------------------
    def _build_live_loader(
        self, profile: str | None
    ) -> tuple[DecisionLoader, Path]:
        core_config = self._load_core_config()
        configured_path, _kwargs = resolve_decision_log_config(core_config)
        if configured_path is None:
            raise FileNotFoundError(
                "Decision log portfela nie jest skonfigurowany w pliku core.yaml"
            )

        log_path = Path(configured_path)
        if not log_path.is_absolute():
            config_path = self._resolve_core_config_path()
            if config_path is not None:
                log_path = (config_path.parent / log_path).resolve()
            else:
                log_path = log_path.expanduser().resolve()

        if not log_path.exists():
            raise FileNotFoundError(
                f"Decision log '{log_path}' nie istnieje – uruchom autotradera, aby utworzyć plik"
            )
        if not log_path.is_file():
            raise IsADirectoryError(
                f"Decision log '{log_path}' wskazuje na katalog – oczekiwany plik JSONL"
            )

        loader = self._build_jsonl_loader(log_path)
        return loader, log_path

    def _build_jsonl_loader(self, log_path: Path) -> DecisionLoader:
        def _loader(limit: int) -> Iterable[DecisionRecord]:
            entries: list[DecisionRecord] = []
            try:
                with log_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        payload = line.strip()
                        if not payload:
                            continue
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            _LOGGER.warning(
                                "Pominięto uszkodzony wpis decision logu %s", log_path, exc_info=True
                            )
                            continue
                        if isinstance(data, Mapping):
                            entries.append(data)
            except FileNotFoundError:
                raise
            except OSError as exc:
                raise RuntimeError(
                    f"Nie udało się odczytać decision logu '{log_path}': {exc}"
                ) from exc

            if limit > 0:
                entries = entries[-limit:]
            return entries

        return _loader

    def _load_core_config(self):
        if self._cached_core_config is not None:
            return self._cached_core_config
        config_path = self._resolve_core_config_path()
        if config_path is None:
            raise FileNotFoundError(
                "Nie znaleziono ścieżki do core.yaml – ustaw zmienną BOT_CORE_UI_CORE_CONFIG_PATH"
            )
        self._cached_core_config = load_core_config(config_path)
        return self._cached_core_config

    def _resolve_core_config_path(self) -> Path | None:
        if self._core_config_path is not None:
            return self._core_config_path

        candidates = (
            os.environ.get("BOT_CORE_UI_CORE_CONFIG_PATH"),
            os.environ.get("BOT_CORE_CORE_CONFIG"),
            os.environ.get("BOT_CORE_CONFIG"),
            os.environ.get("DUDZIAN_CORE_CONFIG"),
        )
        for candidate in candidates:
            if candidate:
                path = Path(candidate).expanduser()
                self._core_config_path = path
                return path

        default = Path("config/core.yaml")
        self._core_config_path = default
        return default

    # ------------------------------------------------------------------ runtime metadata helpers --
    def _update_runtime_metadata(self, *, invalidate_cache: bool) -> None:
        if invalidate_cache:
            self._runtime_config_cache = None
        next_run = self._compute_next_retrain()
        if next_run != self._retrain_next_run:
            self._retrain_next_run = next_run
            self.retrainNextRunChanged.emit()
        summary = self._build_adaptive_summary()
        if summary != self._adaptive_summary:
            self._adaptive_summary = summary
            self.adaptiveStrategySummaryChanged.emit()

    def _load_runtime_config(self) -> "RuntimeAppConfig":
        if load_runtime_app_config is None:
            raise RuntimeError("Ładowanie konfiguracji runtime nie jest dostępne w tej dystrybucji")
        if self._runtime_config_cache is not None:
            return self._runtime_config_cache
        config_path = self._resolve_runtime_config_path()
        if config_path is None:
            raise FileNotFoundError(
                "Nie znaleziono pliku runtime.yaml – ustaw zmienną BOT_CORE_UI_RUNTIME_CONFIG_PATH"
            )
        self._runtime_config_cache = load_runtime_app_config(config_path)
        return self._runtime_config_cache

    def _resolve_runtime_config_path(self) -> Path | None:
        if self._runtime_config_path is not None:
            return self._runtime_config_path

        candidates = (
            os.environ.get("BOT_CORE_UI_RUNTIME_CONFIG_PATH"),
            os.environ.get("BOT_CORE_RUNTIME_CONFIG_PATH"),
            os.environ.get("BOT_CORE_RUNTIME_CONFIG"),
            os.environ.get("DUDZIAN_RUNTIME_CONFIG"),
        )
        for candidate in candidates:
            if candidate:
                path = Path(candidate).expanduser()
                self._runtime_config_path = path
                return path

        default = Path("config/runtime.yaml")
        self._runtime_config_path = default
        return default

    def _compute_next_retrain(self) -> str:
        if CronSchedule is None:
            return ""
        try:
            runtime_config = self._load_runtime_config()
        except FileNotFoundError:
            return ""
        except Exception:  # pragma: no cover - diagnostyka środowiska
            _LOGGER.debug("Nie udało się wczytać konfiguracji runtime", exc_info=True)
            return ""

        schedule: str | None = None
        retrain_cfg = getattr(runtime_config.ai, "retrain", None)
        if retrain_cfg and getattr(retrain_cfg, "enabled", False):
            schedule = getattr(retrain_cfg, "schedule", None) or getattr(runtime_config.ai, "retrain_schedule", None)
        else:
            schedule = getattr(runtime_config.ai, "retrain_schedule", None)
        if not schedule:
            return ""
        try:
            cron = CronSchedule(schedule)
            next_run = cron.next_after(datetime.now(timezone.utc))
        except Exception:  # pragma: no cover - niepoprawna składnia lub błąd obliczeń
            _LOGGER.debug("Nie udało się obliczyć najbliższego retrainingu", exc_info=True)
            return ""
        return next_run.astimezone().isoformat(timespec="minutes")

    def _build_adaptive_summary(self) -> str:
        if ModelRepository is None:
            return ""
        try:
            runtime_config = self._load_runtime_config()
        except FileNotFoundError:
            return ""
        except Exception:  # pragma: no cover - diagnostyka środowiska
            _LOGGER.debug("Nie udało się wczytać konfiguracji runtime dla presetów adaptacyjnych", exc_info=True)
            return ""

        registry_path = getattr(runtime_config.ai, "model_registry_path", None)
        if not registry_path:
            return ""
        try:
            repository = ModelRepository(Path(registry_path))  # type: ignore[abstract]
        except Exception:  # pragma: no cover - repozytorium może być nieosiągalne
            _LOGGER.debug("Nie udało się zainicjalizować ModelRepository", exc_info=True)
            return ""
        try:
            artifact = repository.load("adaptive_strategy_policy.json")
        except FileNotFoundError:
            return ""
        except Exception:  # pragma: no cover - uszkodzony plik lub brak dostępu
            _LOGGER.debug("Nie udało się wczytać stanu adaptive learnera", exc_info=True)
            return ""

        state = getattr(artifact, "model_state", None)
        policies = state.get("policies") if isinstance(state, Mapping) else None
        if not isinstance(policies, Mapping) or not policies:
            return ""

        fragments: list[str] = []
        for regime_key, payload in policies.items():
            if not isinstance(payload, Mapping):
                continue
            strategies = payload.get("strategies")
            if not isinstance(strategies, Iterable):
                continue
            best_name: str | None = None
            best_score: float | None = None
            best_plays = 0
            for entry in strategies:
                if not isinstance(entry, Mapping):
                    continue
                name = str(entry.get("name") or "").strip()
                if not name:
                    continue
                plays = int(entry.get("plays", 0) or 0)
                total_reward = float(entry.get("total_reward", 0.0) or 0.0)
                mean_reward = total_reward / plays if plays > 0 else float(entry.get("last_reward", 0.0) or 0.0)
                if best_score is None or mean_reward > best_score:
                    best_name = name
                    best_score = mean_reward
                    best_plays = plays
            if best_name is None or best_score is None:
                continue
            regime_label = str(regime_key).replace("_", " ")
            fragments.append(f"{regime_label}: {best_name} (μ={best_score:.2f}, n={best_plays})")

        if not fragments:
            return ""

        updated_at = ""
        try:
            updated_at = str(artifact.metadata.get("updated_at", "")).strip()
        except Exception:
            updated_at = ""
        summary = "; ".join(fragments)
        if updated_at:
            summary = f"{summary} — aktualizacja {updated_at}"
        return summary


__all__ = ["RuntimeService"]

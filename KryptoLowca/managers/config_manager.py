# managers/config_manager.py
# -*- coding: utf-8 -*-
"""Konsekwentny manager presetów z walidacją Pydantic.

Funkcje kluczowe:
- zapis/odczyt presetów do katalogu ``presets`` (YAML lub JSON),
- walidacja i normalizacja ustawień GUI bota przed zapisem,
- raportowanie błędów walidacji użytkownikowi (ValueError),
- metody pomocnicze przydatne w GUI: listowanie, usuwanie, ścieżki plików.

Walidacja wykorzystuje ``pydantic`` inspirowaną komercyjnymi botami –
nieprawidłowe wartości (np. frakcja kapitału > 1) są blokowane, a dane
starszych presetów są automatycznie uzupełniane domyślnymi wartościami.
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# YAML opcjonalnie
try:
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover - środowiska bez PyYAML
    yaml = None  # type: ignore
    _HAS_YAML = False

_SANITIZE = re.compile(r"[^a-zA-Z0-9_\-\.]")
_TIMEFRAME_PATTERN = re.compile(r"^[1-9][0-9]*(m|h|d|w)$", re.IGNORECASE)


def _sanitize_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Nazwa presetu nie może być pusta.")
    name = name.replace(" ", "_")
    name = _SANITIZE.sub("", name)
    if name in {".", ".."}:
        raise ValueError("Nieprawidłowa nazwa presetu.")
    return name


class _SectionModel(BaseModel):
    """Bazowa klasa modeli sekcji – pozwala na dodatkowe pola."""

    model_config = ConfigDict(extra="allow")


class AISettings(_SectionModel):
    enable: bool = True
    seq_len: int = Field(default=256, ge=16, le=10_000)
    epochs: int = Field(default=10, ge=1, le=10_000)
    batch: int = Field(default=32, ge=1, le=10_000)
    retrain_min: int = Field(default=60, ge=1, le=100_000)
    train_window: int = Field(default=500, ge=32, le=500_000)
    valid_window: int = Field(default=100, ge=16, le=200_000)
    ai_threshold_bps: float = Field(default=5.0, ge=0.0, le=10_000.0)
    train_all: bool = True


class RiskSettings(_SectionModel):
    max_daily_loss_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    soft_halt_losses: int = Field(default=3, ge=0)
    trade_cooldown_on_error: int = Field(default=30, ge=0, le=3600)
    risk_per_trade: float = Field(default=0.01, ge=0.0, le=1.0)
    portfolio_risk: float = Field(default=0.20, ge=0.0, le=1.0)
    one_trade_per_bar: bool = True
    cooldown_s: int = Field(default=0, ge=0, le=86_400)
    min_move_pct: float = Field(default=0.0, ge=0.0, le=1.0)


class DcaTrailingSettings(_SectionModel):
    use_trailing: bool = True
    atr_period: int = Field(default=14, ge=1, le=500)
    trail_atr_mult: float = Field(default=2.0, ge=0.0, le=20.0)
    take_atr_mult: float = Field(default=3.0, ge=0.0, le=50.0)
    dca_enabled: bool = False
    dca_max_adds: int = Field(default=0, ge=0, le=50)
    dca_step_atr: float = Field(default=2.0, ge=0.0, le=50.0)


class SlippageSettings(_SectionModel):
    use_orderbook_vwap: bool = True
    fallback_bps: float = Field(default=5.0, ge=0.0, le=10_000.0)


class AdvancedSettings(_SectionModel):
    rsi_period: int = Field(default=14, ge=1, le=200)
    ema_fast: int = Field(default=12, ge=1, le=500)
    ema_slow: int = Field(default=26, ge=1, le=1_000)
    atr_period: int = Field(default=14, ge=1, le=500)
    rsi_buy: float = Field(default=35.0, ge=0.0, le=100.0)
    rsi_sell: float = Field(default=65.0, ge=0.0, le=100.0)


class PaperSettings(_SectionModel):
    capital: float = Field(default=10_000.0, ge=0.0, le=1_000_000_000.0)


class ConfigPreset(BaseModel):
    """Model najwyższego poziomu opisujący pojedynczy preset."""

    network: Literal["Testnet", "Live"] = "Testnet"
    mode: Literal["Spot", "Futures"] = "Spot"
    timeframe: str = "1m"
    fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    ai: AISettings = Field(default_factory=AISettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    dca_trailing: DcaTrailingSettings = Field(default_factory=DcaTrailingSettings)
    slippage: SlippageSettings = Field(default_factory=SlippageSettings)
    advanced: AdvancedSettings = Field(default_factory=AdvancedSettings)
    paper: PaperSettings = Field(default_factory=PaperSettings)
    selected_symbols: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")

    @field_validator("timeframe")
    @classmethod
    def _validate_timeframe(cls, value: Any) -> str:
        tf = str(value or "").strip()
        if not tf:
            raise ValueError("Pole 'timeframe' nie może być puste.")
        if not _TIMEFRAME_PATTERN.match(tf):
            raise ValueError("Timeframe musi mieć format np. '1m', '4h', '1d'.")
        return tf.lower()

    @field_validator("fraction")
    @classmethod
    def _validate_fraction(cls, value: Any) -> float:
        try:
            fraction = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Frakcja kapitału musi być liczbą.") from exc
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("Frakcja kapitału musi być w zakresie 0.0 - 1.0.")
        return round(fraction, 4)

    @field_validator("selected_symbols", mode="before")
    @classmethod
    def _clean_symbols(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        cleaned: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            sym = item.strip().upper()
            if sym:
                cleaned.append(sym)
        # usuwamy duplikaty zachowując kolejność
        deduped: List[str] = []
        seen = set()
        for sym in cleaned:
            if sym not in seen:
                seen.add(sym)
                deduped.append(sym)
        return deduped

    @model_validator(mode="after")
    def _post_validation(self) -> "ConfigPreset":
        if self.mode == "Futures" and self.network == "Live":
            logger.warning(
                "Preset używa trybu Futures na Live – upewnij się, że konto ma odpowiednie uprawnienia (zalecane testy na testnet)."
            )
        if self.fraction == 0.0:
            logger.warning(
                "Preset zapisany z frakcją 0.0 – bot nie będzie otwierał nowych pozycji, traktuj to jako tryb tylko-do-oglądania."
            )
        if self.paper.capital < 100.0:
            logger.warning(
                "Preset posiada bardzo niski kapitał symulacyjny (%.2f). Rozważ >= 100 USD dla sensownych testów.",
                self.paper.capital,
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Zwróć słownik bez pól None – gotowy do serializacji."""
        return self.model_dump(mode="python", exclude_none=True)


class ConfigManager:
    """Manager presetów wykorzystywany przez ``trading_gui.py``."""

    def __init__(self, presets_dir: Path | str) -> None:
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ConfigManager: katalog presetów = %s", self.presets_dir)
        self._demo_required = True
        self._default_template = ConfigPreset().to_dict()

    # --- walidacja ---
    def validate_preset(self, data: Dict[str, Any]) -> ConfigPreset:
        """Zwróć obiekt ``ConfigPreset`` po walidacji danych wejściowych."""

        try:
            preset = ConfigPreset.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - trudne do pełnego pokrycia
            errors = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            raise ValueError(f"Preset validation failed: {errors}") from exc
        return preset

    def require_demo_mode(self, required: bool = True) -> None:
        """Włącz/wyłącz wymóg zapisu presetów w trybie demo."""

        self._demo_required = bool(required)

    def demo_mode_required(self) -> bool:
        """Zwraca informację, czy polityka bezpieczeństwa wymaga trybu demo."""

        return self._demo_required

    @staticmethod
    def _merge_nested(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        result = deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = ConfigManager._merge_nested(result[key], value)
            else:
                result[key] = value
        return result

    def audit_preset(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Przeprowadź szybki audyt bezpieczeństwa i ryzyka dla presetu."""

        preset = self.validate_preset(data)
        issues: List[str] = []
        warnings: List[str] = []

        is_demo = preset.network.lower() == "testnet"
        if self._demo_required and not is_demo:
            issues.append(
                "Preset wymaga trybu Testnet zgodnie z polityką bezpieczeństwa. Zmień pole 'network' na 'Testnet'."
            )
        if preset.network.lower() == "live":
            warnings.append(
                "Przed uruchomieniem na Live wykonaj pełne testy na koncie demo i upewnij się, że klucze API mają ograniczone uprawnienia."
            )
        if preset.fraction > 0.5:
            warnings.append("Frakcja kapitału przekracza 50% – rozważ niższą wartość dla redukcji ryzyka.")
        if preset.risk.max_daily_loss_pct > 0.2:
            warnings.append(
                "Parametr max_daily_loss_pct przekracza 20% – może naruszać zasady zarządzania ryzykiem."
            )
        if preset.paper.capital < 500:
            warnings.append("Kapitał paper trading < 500 – wyniki backtestu mogą być mało reprezentatywne.")

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "network": preset.network,
            "issues": issues,
            "warnings": warnings,
            "demo_required": self._demo_required,
            "is_demo": is_demo,
        }

    # --- ścieżki ---
    def _path_yaml(self, safe_name: str) -> Path:
        return self.presets_dir / f"{safe_name}.yaml"

    def _path_json(self, safe_name: str) -> Path:
        return self.presets_dir / f"{safe_name}.json"

    def _pick_existing_path(self, safe_name: str) -> Optional[Path]:
        yml = self._path_yaml(safe_name)
        jsn = self._path_json(safe_name)
        if yml.exists():
            return yml
        if jsn.exists():
            return jsn
        return None

    # --- API używane w trading_gui.py ---
    def save_preset(self, name: str, data: Dict[str, Any]) -> Path:
        safe = _sanitize_name(name)
        if not isinstance(data, dict):
            raise ValueError("Preset musi być słownikiem (dict).")

        preset = self.validate_preset(data)
        if self._demo_required and preset.network.lower() != "testnet":
            raise ValueError(
                "Polityka bezpieczeństwa wymaga tworzenia presetów w trybie Testnet. "
                "Zmień pole 'network' na 'Testnet' lub wyłącz ten wymóg metodą require_demo_mode(False)."
            )
        payload = preset.to_dict()

        if _HAS_YAML:
            path = self._path_yaml(safe)
            try:
                with path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)  # type: ignore[arg-type]
                logger.info("Zapisano preset YAML: %s", path)
                return path
            except Exception as e:  # pragma: no cover - zależne od dysku/środowiska
                logger.error("Błąd zapisu YAML (%s): %s – próba JSON", path, e)

        path = self._path_json(safe)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Zapisano preset JSON: %s", path)
        return path

    def create_preset(
        self,
        name: str,
        *,
        base: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        ensure_demo: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Stwórz preset na bazie szablonu i nadpisów, zwracając raport audytu."""

        payload = deepcopy(self._default_template)
        if base:
            base_payload = self.load_preset(base)
            payload = ConfigManager._merge_nested(payload, base_payload)
        if overrides:
            payload = ConfigManager._merge_nested(payload, overrides)

        preset = self.validate_preset(payload)
        require_demo = self._demo_required if ensure_demo is None else bool(ensure_demo)
        if require_demo and preset.network.lower() != "testnet":
            preset = preset.copy(update={"network": "Testnet"})

        audit = self.audit_preset(preset.to_dict())
        if require_demo and audit["issues"]:
            # Próba zapisu bez trybu demo – blokujemy zanim powstanie plik.
            raise ValueError(audit["issues"][0])

        path = self.save_preset(name, preset.to_dict())
        audit.update({
            "name": name,
            "path": str(path),
        })
        audit["preset"] = preset.to_dict()
        return audit

    def load_preset(self, name: str) -> Dict[str, Any]:
        safe = _sanitize_name(name)
        path = self._pick_existing_path(safe)
        if path is None:
            raise FileNotFoundError(f"Preset '{name}' nie istnieje w {self.presets_dir}")

        if path.suffix.lower() in {".yml", ".yaml"}:
            if not _HAS_YAML:
                raise RuntimeError(
                    "Preset jest w YAML, ale PyYAML nie jest zainstalowany. Zainstaluj: pip install pyyaml"
                )
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)  # type: ignore[assignment]
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Preset '{name}' ma niepoprawną strukturę (oczekiwano dict).")

        preset = self.validate_preset(data)
        payload = preset.to_dict()
        logger.info("Załadowano preset %s (znormalizowany)", path)
        return payload

    def preset_wizard(self) -> "PresetWizard":
        """Utwórz kreator presetów ułatwiający stopniowe budowanie konfiguracji."""

        return PresetWizard(self)

    # --- dodatki przydatne w GUI ---
    def list_presets(self) -> List[str]:
        names: set[str] = set()
        for p in self.presets_dir.glob("*.yaml"):
            names.add(p.stem)
        for p in self.presets_dir.glob("*.yml"):
            names.add(p.stem)
        for p in self.presets_dir.glob("*.json"):
            names.add(p.stem)
        return sorted(names)

    def delete_preset(self, name: str) -> bool:
        safe = _sanitize_name(name)
        ok = False
        for p in (self._path_yaml(safe), self._path_json(safe)):
            try:
                if p.exists():
                    p.unlink()
                    ok = True
            except Exception as e:  # pragma: no cover - błędy IO
                logger.error("Nie udało się usunąć presetu %s: %s", p, e)
        return ok

    def path_for(self, name: str) -> Path:
        safe = _sanitize_name(name)
        p = self._pick_existing_path(safe)
        return p if p is not None else self._path_yaml(safe)


class PresetWizard:
    """Prosty kreator presetów wspierający profile ryzyka i kontrolę demo."""

    def __init__(self, manager: ConfigManager) -> None:
        self._manager = manager
        self._base: Optional[str] = None
        self._overrides: Dict[str, Any] = {}
        self._ensure_demo: Optional[bool] = None

    def from_template(self, name: str) -> "PresetWizard":
        self._base = name
        return self

    def with_symbols(self, symbols: Iterable[str]) -> "PresetWizard":
        cleaned: List[str] = []
        for sym in symbols:
            if not isinstance(sym, str):
                continue
            normalised = sym.strip().upper()
            if normalised and normalised not in cleaned:
                cleaned.append(normalised)
        if cleaned:
            self._overrides["selected_symbols"] = cleaned
        return self

    def with_risk_profile(self, profile: str) -> "PresetWizard":
        presets = {
            "conservative": {
                "fraction": 0.1,
                "risk": {"max_daily_loss_pct": 0.04, "risk_per_trade": 0.01},
            },
            "balanced": {
                "fraction": 0.2,
                "risk": {"max_daily_loss_pct": 0.08, "risk_per_trade": 0.02},
            },
            "aggressive": {
                "fraction": 0.35,
                "risk": {"max_daily_loss_pct": 0.15, "risk_per_trade": 0.04},
            },
        }
        key = profile.strip().lower()
        if key not in presets:
            raise ValueError(f"Nieznany profil ryzyka: {profile}")
        self._overrides = ConfigManager._merge_nested(self._overrides, presets[key])
        return self

    def ensure_demo(self, required: bool = True) -> "PresetWizard":
        self._ensure_demo = required
        return self

    def override(self, **kwargs: Any) -> "PresetWizard":
        if kwargs:
            self._overrides = ConfigManager._merge_nested(self._overrides, kwargs)
        return self

    def build(self, name: str) -> Dict[str, Any]:
        audit = self._manager.create_preset(
            name,
            base=self._base,
            overrides=self._overrides or None,
            ensure_demo=self._ensure_demo,
        )
        # reset stanu aby kreator można było użyć ponownie
        self._base = None
        self._overrides = {}
        self._ensure_demo = None
        return audit


__all__ = [
    "ConfigManager",
    "ConfigPreset",
    "AISettings",
    "RiskSettings",
    "DcaTrailingSettings",
    "SlippageSettings",
    "AdvancedSettings",
    "PaperSettings",
    "PresetWizard",
]

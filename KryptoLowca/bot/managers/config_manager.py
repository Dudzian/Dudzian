# managers/config_manager.py
# -*- coding: utf-8 -*-
"""
Lekki ConfigManager zgodny z trading_gui.py:
- operuje na katalogu presetów (PRESETS_DIR z GUI),
- metody: save_preset(name, dict), load_preset(name),
- zapis w YAML (jeśli dostępny PyYAML), fallback do JSON,
- bez importów ExchangeAdapter itp.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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
except Exception:
    yaml = None  # type: ignore
    _HAS_YAML = False

_SANITIZE = re.compile(r"[^a-zA-Z0-9_\-\.]")

def _sanitize_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Nazwa presetu nie może być pusta.")
    name = name.replace(" ", "_")
    name = _SANITIZE.sub("", name)
    if name in {".", ".."}:
        raise ValueError("Nieprawidłowa nazwa presetu.")
    return name


class ConfigManager:
    """
    Minimalny manager presetów używany przez trading_gui.py

    Przykład:
        cfg = ConfigManager(Path('presets'))
        cfg.save_preset('moja_konfig', {'exchange': 'binance', 'testnet': True})
        data = cfg.load_preset('moja_konfig')
    """

    def __init__(self, presets_dir: Path | str) -> None:
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ConfigManager: katalog presetów = %s", self.presets_dir)

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

        if _HAS_YAML:
            path = self._path_yaml(safe)
            try:
                with path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)  # type: ignore
                logger.info("Zapisano preset YAML: %s", path)
                return path
            except Exception as e:
                logger.error("Błąd zapisu YAML (%s): %s – próba JSON", path, e)

        path = self._path_json(safe)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Zapisano preset JSON: %s", path)
        return path

    def load_preset(self, name: str) -> Dict[str, Any]:
        safe = _sanitize_name(name)
        path = self._pick_existing_path(safe)
        if path is None:
            raise FileNotFoundError(f"Preset '{name}' nie istnieje w {self.presets_dir}")

        if path.suffix.lower() in {".yml", ".yaml"}:
            if not _HAS_YAML:
                raise RuntimeError("Preset jest w YAML, ale PyYAML nie jest zainstalowany. Zainstaluj: pip install pyyaml")
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)  # type: ignore
            if not isinstance(data, dict):
                raise ValueError(f"Preset '{name}' ma niepoprawną strukturę (oczekiwano dict).")
            return data  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Preset '{name}' ma niepoprawną strukturę (oczekiwano dict).")
        return data  # type: ignore

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
            except Exception as e:
                logger.error("Nie udało się usunąć presetu %s: %s", p, e)
        return ok

    def path_for(self, name: str) -> Path:
        safe = _sanitize_name(name)
        p = self._pick_existing_path(safe)
        return p if p is not None else self._path_yaml(safe)

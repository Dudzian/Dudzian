"""Prosty edytor CLI presetĂ„â€šĂ˘â‚¬ĹľÄ‚ËĂ˘â€šÂ¬ÄąË‡Ä‚â€žĂ„â€¦Ä‚ËĂ˘â€šÂ¬ÄąË‡w marketplace.

Pozwala zastosowaÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€ąĂ˘â‚¬Ë‡ preset, wprowadziÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€ąĂ˘â‚¬Ë‡ modyfikacje i zapisaÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€ąĂ˘â‚¬Ë‡ konfiguracjÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄľÄ‚â€ąĂ‚Â
z wykorzystaniem szyfrowania sekcji API. ModuĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…Ă‹â€ˇ stanowi stub moĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ˘â‚¬ĹľÄ‚â€ąÄąÄ„liwy do
rozszerzenia o GUI w przyszĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…Ă‹â€ˇoĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…ÄąĹźci.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml
from KryptoLowca.config_manager import ConfigManager, ConfigError, ValidationError


def _load_key(args: argparse.Namespace) -> bytes:
    if args.encryption_key and args.encryption_key_file:
        raise SystemExit("Podaj klucz w formie tekstu lub pliku, nie obu jednoczeĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…ÄąĹźnie.")
    if args.encryption_key:
        return args.encryption_key.encode()
    if args.encryption_key_file:
        path = Path(args.encryption_key_file)
        try:
            return path.read_text(encoding="utf-8").strip().encode()
        except OSError as exc:  # pragma: no cover - informacja o bĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…Ă‹â€ˇÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄľÄ‚â€ąĂ‚Âdzie IO
            raise SystemExit(f"Nie moĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ˘â‚¬ĹľÄ‚â€ąÄąÄ„na odczytaÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€ąĂ˘â‚¬Ë‡ pliku z kluczem: {exc}")
    raise SystemExit("Wymagany jest klucz szyfrujÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦cy (podaj --encryption-key lub --encryption-key-file).")


def _parse_overrides(items: Iterable[str]) -> Dict[str, Dict[str, object]]:
    overrides: Dict[str, Dict[str, object]] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"NieprawidĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…Ă‹â€ˇowy format nadpisania: '{item}'. UĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ˘â‚¬ĹľÄ‚â€ąÄąÄ„yj section.key=value.")
        path, raw_value = item.split("=", 1)
        if "." not in path:
            raise SystemExit(f"NieprawidĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…Ă‹â€ˇowy format Ă„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…ÄąĹźcieĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ˘â‚¬ĹľÄ‚â€ąÄąÄ„ki: '{path}'. UĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ˘â‚¬ĹľÄ‚â€ąÄąÄ„yj section.key.")
        section, key = path.split(".", 1)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        section = section.strip()
        key = key.strip()
        overrides.setdefault(section, {})[key] = value
    return overrides


def _summarise_overrides(overrides: Dict[str, Dict[str, object]]) -> str:
    flattened: Iterable[Tuple[str, str, object]] = (
        (section, key, value)
        for section, values in overrides.items()
        for key, value in values.items()
    )
    return ", ".join(f"{section}.{key}={value}" for section, key, value in flattened)


async def _run_async(args: argparse.Namespace) -> int:
    key = _load_key(args)
    manager = ConfigManager(Path(args.config_path), encryption_key=key)  # type: ignore[misc,call-arg]
    if args.marketplace_dir:
        manager.set_marketplace_directory(Path(args.marketplace_dir))  # type: ignore[attr-defined]

    await manager.load_config()  # type: ignore[attr-defined]
    try:
        config = manager.apply_marketplace_preset(
            args.preset_id,
            actor=args.actor,
            user_confirmed=args.confirm_live,
            note=args.note,
        )
    except (ConfigError, ValidationError) as exc:
        print(f"BĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…Ă‹â€ˇÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦d zastosowania presetu: {exc}")
        return 1

    overrides = _parse_overrides(args.set or [])
    for section, values in overrides.items():
        section_payload = config.get(section)
        if not isinstance(section_payload, dict):
            section_payload = {}
            config[section] = section_payload
        section_payload.update(values)

    override_note = _summarise_overrides(overrides) if overrides else None
    combined_note = args.note
    if override_note:
        combined_note = f"{args.note}; overrides: {override_note}" if args.note else f"overrides: {override_note}"

    await manager.save_config(
        config,
        actor=args.actor,
        preset_id=args.preset_id,
        note=combined_note,
        source="editor",
    )  # type: ignore[attr-defined]

    print(
        "Zapisano preset '{preset}' (wersja: {version}) do pliku {path}".format(
            preset=args.preset_id,
            version=config.get("strategy", {}).get("preset", "custom"),
            path=manager.config_path,  # type: ignore[attr-defined]
        )
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Edytor presetĂ„â€šĂ˘â‚¬ĹľÄ‚ËĂ˘â€šÂ¬ÄąË‡Ä‚â€žĂ„â€¦Ä‚ËĂ˘â€šÂ¬ÄąË‡w marketplace (CLI)")
    parser.add_argument("--config-path", required=True, help="Ă„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ä‚â€žĂ„â€¦Ä‚â€ąĂ˘â‚¬Ë‡cieĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ˘â‚¬ĹľÄ‚â€ąÄąÄ„ka do pliku konfiguracji YAML")
    parser.add_argument("--preset-id", required=True, help="Identyfikator presetu z marketplace")
    parser.add_argument("--marketplace-dir", help="Katalog z plikami presetĂ„â€šĂ˘â‚¬ĹľÄ‚ËĂ˘â€šÂ¬ÄąË‡Ä‚â€žĂ„â€¦Ä‚ËĂ˘â€šÂ¬ÄąË‡w marketplace (opcjonalnie)")
    parser.add_argument("--encryption-key", help="Klucz Fernet w formacie base64")
    parser.add_argument("--encryption-key-file", help="Plik zawierajÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦cy klucz Fernet")
    parser.add_argument("--set", action="append", help="Nadpisania sekcji, np. trade.max_open_positions=3")
    parser.add_argument("--actor", required=True, help="Adres e-mail lub identyfikator uĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ˘â‚¬ĹľÄ‚â€ąÄąÄ„ytkownika wykonujÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦cego zmianÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄľÄ‚â€ąĂ‚Â")
    parser.add_argument("--note", help="Dodatkowa notatka do zapisu audytowego")
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Potwierdza Ă„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…ÄąĹźwiadomÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ä‚â€šĂ‚Â¦ aktywacjÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąÄľĂ„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄľÄ‚â€ąĂ‚Â trybu LIVE (jeĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…ÄąĹźli preset go wymaga)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:  # pragma: no cover - obsĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…Ă‹â€ˇuga przerwania
        return 130


if __name__ == "__main__":  # pragma: no cover - wejĂ„â€šĂ˘â‚¬ĹľÄ‚â€žĂ˘â‚¬Â¦Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…ÄąĹźcie CLI
    raise SystemExit(main())

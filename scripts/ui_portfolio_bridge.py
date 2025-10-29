"""Mostek CLI pomiędzy UI a lokalnym stanem harmonogramu multi-portfelowego."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

STORE_DEFAULT = Path("var/portfolio_links.json")


def _load_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"portfolios": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - diagnostyczne logowanie
        raise SystemExit(f"Plik {path} zawiera niepoprawny JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        return {"portfolios": {}}
    portfolios = payload.get("portfolios")
    if isinstance(portfolios, Mapping):
        return {"portfolios": {key: dict(value) for key, value in portfolios.items() if isinstance(value, Mapping)}}
    if isinstance(portfolios, list):
        normalized: dict[str, Any] = {}
        for entry in portfolios:
            if not isinstance(entry, Mapping):
                continue
            portfolio_id = str(entry.get("portfolio_id") or entry.get("id") or "").strip()
            if not portfolio_id:
                continue
            normalized[portfolio_id] = dict(entry)
        return {"portfolios": normalized}
    return {"portfolios": {}}


def _write_store(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(serialized, encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _list_command(args: argparse.Namespace) -> None:
    store = _load_store(args.store)
    entries = list(store.get("portfolios", {}).values())
    entries.sort(key=lambda item: str(item.get("portfolio_id", "")))
    document = {
        "portfolios": entries,
        "generated_at": _timestamp(),
    }
    json.dump(document, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _load_payload(path: str | None) -> Mapping[str, Any]:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        data = sys.stdin.read()
        if not data.strip():
            raise SystemExit("Oczekiwano JSON na standardowym wejściu")
        payload = json.loads(data)
    if not isinstance(payload, Mapping):
        raise SystemExit("Payload musi być obiektem JSON")
    return dict(payload)


def _apply_command(args: argparse.Namespace) -> None:
    store = _load_store(args.store)
    payload = _load_payload(args.payload)
    portfolio_id = str(payload.get("portfolio_id") or payload.get("id") or "").strip()
    if not portfolio_id:
        raise SystemExit("Pole 'portfolio_id' jest wymagane")
    entry = dict(payload)
    entry["portfolio_id"] = portfolio_id
    entry["updated_at"] = _timestamp()
    store.setdefault("portfolios", {})[portfolio_id] = entry
    _write_store(args.store, store)


def _remove_command(args: argparse.Namespace) -> None:
    store = _load_store(args.store)
    portfolio_id = args.portfolio_id.strip()
    if not portfolio_id:
        raise SystemExit("Wymagany identyfikator portfela")
    if portfolio_id in store.get("portfolios", {}):
        store["portfolios"].pop(portfolio_id, None)
        _write_store(args.store, store)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        type=Path,
        default=STORE_DEFAULT,
        help="Ścieżka do pliku z konfiguracją portfeli",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Wyświetla zapisane portfele")
    list_parser.set_defaults(func=_list_command)

    apply_parser = subparsers.add_parser("apply", help="Aktualizuje konfigurację portfela na podstawie JSON")
    apply_parser.add_argument("--payload", help="Ścieżka do pliku JSON z definicją portfela")
    apply_parser.set_defaults(func=_apply_command)

    remove_parser = subparsers.add_parser("remove", help="Usuwa portfel z konfiguracji")
    remove_parser.add_argument("portfolio_id", help="Identyfikator portfela do usunięcia")
    remove_parser.set_defaults(func=_remove_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.store = args.store if isinstance(args.store, Path) else Path(args.store)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())

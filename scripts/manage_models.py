"""CLI do zarządzania rejestrem modeli (publish/list/rollback)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import timezone
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.runtime.state_manager import RuntimeStateManager
from core.ml.model_registry import ModelRegistry, ModelRegistryError

LOGGER = logging.getLogger("manage_models")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zarządza rejestrem modeli ML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish", help="Publikuje artefakt modelu")
    publish.add_argument("--artifact", type=Path, required=True, help="Ścieżka do pliku modelu")
    publish.add_argument("--backend", required=True, help="Nazwa backendu modelu")
    publish.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dodatkowe metadane datasetu",
    )
    publish.add_argument(
        "--state-dir",
        type=Path,
        default=Path("var/runtime"),
        help="Katalog z plikiem state.json RuntimeStateManager",
    )
    publish.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("var/models"),
        help="Katalog rejestru modeli",
    )

    list_parser = subparsers.add_parser("list", help="Listuje zarejestrowane modele")
    list_parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("var/models"),
        help="Katalog rejestru modeli",
    )
    list_parser.add_argument(
        "--output",
        choices=("json", "table"),
        default="table",
        help="Format wypisu (domyślnie tabela tekstowa)",
    )

    rollback = subparsers.add_parser("rollback", help="Aktywuje wskazany model")
    rollback.add_argument("--model-id", required=True, help="Identyfikator modelu do aktywacji")
    rollback.add_argument(
        "--state-dir",
        type=Path,
        default=Path("var/runtime"),
        help="Katalog z plikiem state.json RuntimeStateManager",
    )
    rollback.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("var/models"),
        help="Katalog rejestru modeli",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (domyślnie INFO)",
    )
    return parser


def _parse_metadata(entries: Sequence[str]) -> Mapping[str, str]:
    metadata: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ModelRegistryError(f"Niepoprawny format metadanych: '{entry}' (oczekiwano KEY=VALUE)")
        key, value = entry.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _build_registry(registry_dir: Path, state_dir: Path | None = None) -> ModelRegistry:
    state_manager = None
    if state_dir is not None:
        state_manager = RuntimeStateManager(root=state_dir)
    return ModelRegistry(root=registry_dir, state_manager=state_manager)


def _command_publish(args: argparse.Namespace) -> int:
    registry = _build_registry(args.registry_dir, args.state_dir)
    metadata = _parse_metadata(args.metadata)
    result = registry.publish_model(args.artifact, backend=args.backend, dataset_metadata=metadata)
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0


def _command_list(args: argparse.Namespace) -> int:
    registry = _build_registry(args.registry_dir, None)
    models = registry.list_models()
    if args.output == "json":
        payload = [model.to_dict() for model in models]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not models:
        print("Brak modeli w rejestrze")
        return 0

    header = f"{'MODEL ID':<24} {'BACKEND':<12} {'HASH':<16} CREATED AT"
    print(header)
    print("-" * len(header))
    for model in models:
        created = model.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        print(f"{model.model_id:<24} {model.backend:<12} {model.sha256[:16]:<16} {created}")
    return 0


def _command_rollback(args: argparse.Namespace) -> int:
    registry = _build_registry(args.registry_dir, args.state_dir)
    metadata = registry.rollback(args.model_id)
    print(json.dumps(metadata.to_dict(), ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    try:
        if args.command == "publish":
            return _command_publish(args)
        if args.command == "list":
            return _command_list(args)
        if args.command == "rollback":
            return _command_rollback(args)
    except ModelRegistryError as exc:
        LOGGER.error("Błąd rejestru modeli: %s", exc)
        return 2

    return 1


if __name__ == "__main__":  # pragma: no cover - wywołanie z CLI
    raise SystemExit(main())


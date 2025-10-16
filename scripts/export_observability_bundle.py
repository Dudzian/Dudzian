"""Eksportuje paczkę obserwowalności Stage6 wraz z podpisanym manifestem."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.observability import (  # noqa: E402 - import po modyfikacji sys.path
    AlertOverrideManager,
    load_overrides_document,
)
from bot_core.observability.bundle import (  # noqa: E402 - import po modyfikacji sys.path
    AssetSource,
    ObservabilityBundleBuilder,
)


def _parse_metadata(values: list[str] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not values:
        return metadata
    for item in values:
        if "=" not in item:
            raise ValueError("Metadane muszą mieć format klucz=wartość")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Klucz metadanych nie może być pusty")
        metadata[key] = value.strip()
    return metadata


def _parse_sources(values: list[str] | None) -> list[AssetSource]:
    sources: list[AssetSource] = []
    if values:
        for item in values:
            if "=" not in item:
                raise ValueError("Źródło musi mieć format kategoria=ścieżka")
            category, raw_path = item.split("=", 1)
            category = category.strip()
            if not category:
                raise ValueError("Kategoria źródła nie może być pusta")
            path = Path(raw_path.strip())
            sources.append(AssetSource(category=category, root=path))
    else:
        sources = [
            AssetSource(
                category="dashboards",
                root=REPO_ROOT / "deploy" / "grafana" / "provisioning" / "dashboards",
            ),
            AssetSource(
                category="alerts",
                root=REPO_ROOT / "deploy" / "prometheus",
            ),
        ]
    return sources


def _load_hmac_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    inline = args.hmac_key
    file_path = args.hmac_key_file
    env_name = args.hmac_key_env
    key_id = args.hmac_key_id

    if inline and file_path:
        raise ValueError("Podaj klucz HMAC jako wartość lub plik, nie oba jednocześnie")

    if inline:
        key = inline.encode("utf-8")
    elif file_path:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Plik klucza HMAC nie istnieje: {path}")
        if os.name != "nt":
            mode = path.stat().st_mode
            if mode & 0o077:
                raise ValueError("Plik klucza HMAC powinien mieć uprawnienia maks. 600")
        key = path.read_bytes()
    elif env_name:
        value = os.getenv(env_name)
        if not value:
            raise ValueError(f"Zmienna środowiskowa {env_name} nie zawiera klucza HMAC")
        key = value.encode("utf-8")
    else:
        return None, None

    if len(key) < 16:
        raise ValueError("Klucz HMAC powinien mieć co najmniej 16 bajtów")
    return key, key_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Buduje paczkę obserwowalności Stage6 z manifestem i opcjonalnym podpisem.",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Źródło w formacie kategoria=ścieżka (domyślnie Stage6 dashboardy i alerty)",
    )
    parser.add_argument(
        "--include",
        action="append",
        help="Wzorce plików do uwzględnienia (glob, można podać wielokrotnie)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="Wzorce plików do pominięcia (glob, można podać wielokrotnie)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "var" / "observability"),
        help="Katalog docelowy dla paczki",
    )
    parser.add_argument(
        "--bundle-name",
        default="stage6-observability",
        help="Prefiks nazwy paczki (domyślnie stage6-observability)",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        help="Metadane w formacie klucz=wartość (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--overrides",
        help="Ścieżka do pliku override alertów (JSON Stage6)",
    )
    parser.add_argument("--hmac-key", help="Wartość klucza HMAC do podpisu manifestu")
    parser.add_argument("--hmac-key-file", help="Plik zawierający klucz HMAC (UTF-8)")
    parser.add_argument(
        "--hmac-key-env",
        help="Nazwa zmiennej środowiskowej z kluczem HMAC",
    )
    parser.add_argument(
        "--hmac-key-id",
        help="Opcjonalny identyfikator klucza HMAC zapisywany w podpisie",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        sources = _parse_sources(args.source)
        include = args.include or ["stage6*", "**/stage6*"]
        metadata = _parse_metadata(args.metadata)
        if args.overrides:
            overrides_path = Path(args.overrides)
            overrides_data = json.loads(overrides_path.read_text(encoding="utf-8"))
            overrides = load_overrides_document(overrides_data)
            overrides_manager = AlertOverrideManager(overrides)
            overrides_manager.prune_expired()
            overrides_payload = overrides_manager.to_payload()
            metadata["alert_overrides"] = {
                "path": overrides_path.as_posix(),
                "summary": overrides_payload.get("summary"),
                "annotations": overrides_payload.get("annotations"),
            }
        signing_key, key_id = _load_hmac_key(args)
        builder = ObservabilityBundleBuilder(
            sources,
            include=include,
            exclude=args.exclude or None,
        )
        artifacts = builder.build(
            bundle_name=args.bundle_name,
            output_dir=Path(args.output_dir),
            metadata=metadata,
            signing_key=signing_key,
            signing_key_id=key_id,
        )
    except Exception as exc:  # noqa: BLE001 - komunikat dla operatora
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    summary = {
        "bundle": artifacts.bundle_path.as_posix(),
        "manifest": artifacts.manifest_path.as_posix(),
        "files": artifacts.manifest.get("file_count"),
        "size_bytes": artifacts.manifest.get("total_size_bytes"),
    }
    if artifacts.signature_path:
        summary["signature"] = artifacts.signature_path.as_posix()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(run())


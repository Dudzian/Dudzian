#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage5 TCO – merged CLI (HEAD + main)

Subcommands:
  - summary : agreguje koszty z pojedynczego JSON (pozycje kosztowe) i generuje JSON/CSV + podpis HMAC
  - analyze : analizuje JSONL z fillami, generuje raporty przez TCOAnalyzer/TCOReportWriter + opcjonalne podpisy
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional, Sequence

# Domyślne ścieżki Stage5 wykorzystywane do trybu summary
DEFAULT_STAGE5_TCO_INPUT = Path("data/stage5/tco.json")
DEFAULT_CORE_CONFIG_PATH = Path("config/core.yaml")


def _load_yaml_module() -> ModuleType | None:
    spec = importlib.util.find_spec("yaml")
    if spec is None:
        return None
    return importlib.import_module("yaml")  # type: ignore[return-value]


_YAML_MODULE = _load_yaml_module()

# ────────────────────────────────────────────────────────────────────────────────
# Importy wariantu HEAD (summary)
# ────────────────────────────────────────────────────────────────────────────────
from bot_core.reporting.tco import (
    TcoUsageMetrics,
    aggregate_costs,
    load_cost_items,
    write_summary_csv,
    write_summary_json,
    write_summary_signature,
)

# ────────────────────────────────────────────────────────────────────────────────
# Importy wariantu main (analyze)
# ────────────────────────────────────────────────────────────────────────────────
from bot_core.tco import TCOAnalyzer, TCOReportWriter, TradeCostEvent


# ==============================================================================
# Wspólne utilsy
# ==============================================================================

def _now_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize_timestamp(raw: str | None) -> str:
    if not raw or not raw.strip():
        return _now_utc_ts()
    candidate = raw.strip()
    return candidate.replace(":", "").replace("-", "").replace(" ", "_")


_TIMESTAMP_PATTERN = re.compile(r"^\d{8}(T\d{6}Z?)?$")


def _looks_like_timestamp(candidate: str | None) -> bool:
    if not candidate:
        return False
    sanitized = _sanitize_timestamp(candidate)
    normalized = sanitized.replace("_", "")
    return _TIMESTAMP_PATTERN.fullmatch(normalized) is not None


# ==============================================================================
# Domyślne ścieżki i konfiguracja Stage5
# ==============================================================================

def _extract_stage5_tco_input(node: object) -> Optional[str]:
    """Spróbuj odnaleźć ścieżkę Stage5 TCO w zagnieżdżonych strukturach YAML."""

    if isinstance(node, dict):
        # Potencjalne ścieżki konfiguracyjne (stage5.tco.support_input itp.)
        preferred_paths = [
            ("stage5", "tco", "support_input"),
            ("stage5", "tco", "input"),
            ("stage5", "support", "tco_input"),
            ("stage5_support", "tco", "input"),
        ]
        for keys in preferred_paths:
            cursor: object = node
            for key in keys:
                if not isinstance(cursor, dict) or key not in cursor:
                    break
                cursor = cursor[key]
            else:
                if isinstance(cursor, str):
                    return cursor

        for value in node.values():
            candidate = _extract_stage5_tco_input(value)
            if candidate:
                return candidate
    elif isinstance(node, list):
        for item in node:
            candidate = _extract_stage5_tco_input(item)
            if candidate:
                return candidate
    elif isinstance(node, str):
        lowered = node.lower()
        if "stage5" in lowered and "tco" in lowered and lowered.endswith(".json") and "summary" not in lowered:
            return node
    return None


def _resolve_summary_input_from_env() -> tuple[Optional[Path], list[str]]:
    warnings: list[str] = []
    raw = os.environ.get("STAGE5_TCO_INPUT")
    if not raw or not raw.strip():
        return None, warnings

    expanded = os.path.expandvars(raw.strip())
    candidate = Path(expanded).expanduser()
    return candidate, warnings


def _resolve_summary_input_from_config() -> tuple[Optional[Path], list[str]]:
    warnings: list[str] = []
    config_path = DEFAULT_CORE_CONFIG_PATH

    if _YAML_MODULE is None:
        warnings.append(
            "Ostrzeżenie: biblioteka PyYAML nie jest dostępna – pomijam automatyczny odczyt config/core.yaml."
        )
        return None, warnings

    if not config_path.exists():
        return None, warnings

    try:
        contents = config_path.read_text(encoding="utf-8")
        data = _YAML_MODULE.safe_load(contents)  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover - błędy I/O są rzadkie
        warnings.append(f"Ostrzeżenie: nie udało się wczytać {config_path}: {exc}")
        return None, warnings

    candidate = _extract_stage5_tco_input(data)
    if candidate:
        candidate_path = Path(candidate).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = (config_path.parent / candidate_path).resolve(strict=False)
        return candidate_path, warnings

    return None, warnings


def _resolve_default_summary_input() -> tuple[Optional[Path], list[str], str]:
    resolved_path, warnings = _resolve_summary_input_from_env()
    if resolved_path:
        return resolved_path, warnings, "env"

    config_path, config_warnings = _resolve_summary_input_from_config()
    warnings.extend(config_warnings)
    if config_path:
        return config_path, warnings, "config"

    return DEFAULT_STAGE5_TCO_INPUT.expanduser(), warnings, "fallback"


# ==============================================================================
# Subcommand: summary (HEAD)
# ==============================================================================

def _build_summary_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "summary",
        help="Agregacja pozycji kosztowych i podpisany raport TCO (wariant HEAD)",
        description="Generuje raport TCO (JSON + CSV) z pojedynczego pliku JSON i opcjonalnie podpisuje HMAC."
    )
    p.add_argument(
        "--input",
        required=False,
        help=(
            "Plik JSON z listą pozycji kosztowych. Domyślnie użyje ścieżki Stage5/TCO z config/core.yaml lub data/stage5/tco.json"
        ),
    )
    p.add_argument(
        "--artifact-root",
        default="var/audit/tco",
        help="Katalog docelowy artefaktów TCO (domyślnie var/audit/tco)",
    )
    p.add_argument("--json-name", default="tco_summary.json", help="Nazwa pliku podsumowania JSON")
    p.add_argument("--csv-name", default="tco_breakdown.csv", help="Nazwa pliku CSV z rozbiciem pozycji")
    p.add_argument(
        "--signature-name",
        default="tco_summary.signature.json",
        help="Nazwa pliku podpisu HMAC podsumowania",
    )
    p.add_argument(
        "--output",
        help="Alias runbookowy: katalog docelowy lub ścieżka CSV (ustawia też nazwy JSON/podpisu)",
    )
    p.add_argument("--monthly-trades", type=float, help="Średnia miesięczna liczba transakcji")
    p.add_argument("--monthly-volume", type=float, help="Średni miesięczny wolumen obrotu (np. USD)")
    p.add_argument("--tag", help="Opcjonalny tag raportu (trafi do metadanych)")
    p.add_argument("--timestamp", help="Znacznik czasu katalogu artefaktów (domyślnie bieżący UTC)")
    # podpis (HEAD dopuszcza inline lub plik)
    p.add_argument("--signing-key", help="Wartość klucza HMAC (ciąg znaków) do podpisu raportu")
    p.add_argument("--signing-key-file", help="Plik zawierający klucz HMAC do podpisu raportu")
    p.add_argument("--signing-key-id", default="tco", help="Identyfikator klucza podpisującego (domyślnie 'tco')")
    p.add_argument("--print-summary", action="store_true", help="Wypisz podsumowanie JSON na stdout")
    p.set_defaults(_handler=_handle_summary)
    return p


def _load_signing_key_inline_or_file(*, value: str | None, path: str | None) -> bytes | None:
    if value and path:
        raise ValueError("Użyj tylko jednej z opcji: --signing-key lub --signing-key-file")
    if value:
        return value.encode("utf-8")
    if path:
        return Path(path).expanduser().read_bytes().strip()
    return None


def _handle_summary(args: argparse.Namespace) -> int:
    if getattr(args, "output", None):
        output_path = Path(args.output).expanduser()
        if output_path.suffix.lower() == ".csv":
            base_name = output_path.stem or "tco_summary"
            parent_dir = output_path.parent
            timestamp_hint = parent_dir.name if args.timestamp is None else None
            if timestamp_hint and _looks_like_timestamp(timestamp_hint):
                args.timestamp = timestamp_hint
                parent_dir = parent_dir.parent
            parent_dir = parent_dir if str(parent_dir) else Path(".")
            args.artifact_root = str(parent_dir)
            args.csv_name = output_path.name
            args.json_name = f"{base_name}.json"
            args.signature_name = f"{base_name}.signature.json"
        else:
            args.artifact_root = str(output_path)

    # walidacje
    if args.monthly_trades is not None and args.monthly_trades < 0:
        print("Parametr --monthly-trades musi być nieujemny", file=sys.stderr)
        return 2
    if args.monthly_volume is not None and args.monthly_volume < 0:
        print("Parametr --monthly-volume musi być nieujemny", file=sys.stderr)
        return 2

    input_argument = getattr(args, "input", None)
    if input_argument:
        input_path = Path(input_argument).expanduser()
    else:
        default_input, default_warnings, resolved_source = _resolve_default_summary_input()
        for message in default_warnings:
            print(message, file=sys.stderr)
        if default_input is None:
            print(
                "Ostrzeżenie: nie podano --input, a nie znaleziono domyślnej ścieżki Stage5/TCO. Użyj --input, aby wskazać plik kosztów.",
                file=sys.stderr,
            )
            return 0

        input_path = default_input
        if not input_path.exists():
            if resolved_source == "config":
                location_hint = str(DEFAULT_CORE_CONFIG_PATH)
            elif resolved_source == "env":
                location_hint = "zmienna środowiskowa STAGE5_TCO_INPUT"
            else:
                location_hint = str(DEFAULT_STAGE5_TCO_INPUT)
            print(
                f"Ostrzeżenie: nie podano --input, a plik Stage5/TCO {input_path} (źródło: {location_hint}) nie istnieje. "
                "Pomiń ostrzeżenie, jeśli raport nie jest wymagany, lub użyj --input.",
                file=sys.stderr,
            )
            return 0

        print(f"Używam domyślnej ścieżki Stage5/TCO: {input_path}", file=sys.stderr)

    try:
        items = load_cost_items(input_path)
    except Exception as exc:
        print(f"Nie udało się wczytać pozycji kosztowych: {exc}", file=sys.stderr)
        return 2

    summary = aggregate_costs(items)
    usage = TcoUsageMetrics(
        monthly_trades=args.monthly_trades,
        monthly_volume=args.monthly_volume,
    )

    artifact_root = Path(args.artifact_root).expanduser()
    run_dir = artifact_root / _sanitize_timestamp(args.timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc)
    metadata = {
        "input_path": str(input_path.resolve()),
        "items_count": len(summary.items),
    }
    if args.tag:
        metadata["tag"] = args.tag

    json_path = run_dir / args.json_name
    csv_path = run_dir / args.csv_name
    payload = write_summary_json(
        summary,
        json_path,
        generated_at=generated_at,
        usage=usage,
        metadata=metadata,
    )
    write_summary_csv(summary, csv_path)

    # podpis HMAC (opcjonalnie)
    signature_path: Optional[Path] = None
    try:
        signing_key = _load_signing_key_inline_or_file(value=args.signing_key, path=args.signing_key_file)
    except FileNotFoundError as exc:
        print(f"Błąd: nie znaleziono pliku z kluczem HMAC: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 2

    if signing_key is not None:
        signature_path = run_dir / args.signature_name
        write_summary_signature(payload, signature_path, key=signing_key, key_id=args.signing_key_id)

    # wyjście
    if args.print_summary:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")

    if signature_path:
        print(f"Raport zapisano w {json_path} (podpis: {signature_path})")
    else:
        print(f"Raport zapisano w {json_path}")
    return 0


# ==============================================================================
# Subcommand: analyze (main)
# ==============================================================================

def _build_analyze_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "analyze",
        help="Analiza filli JSONL i podpisany pakiet raportów TCO (wariant main)",
        description="Wczytuje wiele JSONL z transakcjami (fills), buduje raport TCO i opcjonalnie podpisuje artefakty."
    )
    p.add_argument(
        "--fills",
        dest="fills",
        action="append",
        required=True,
        help="Ścieżka do pliku JSONL z transakcjami (można podać wielokrotnie)",
    )
    p.add_argument("--output-dir", default="var/audit/tco", help="Katalog docelowy raportu (domyślnie var/audit/tco)")
    p.add_argument("--basename", default=None, help="Opcjonalna nazwa bazowa plików (bez rozszerzeń)")
    p.add_argument(
        "--output",
        help="Alias runbookowy: katalog docelowy lub ścieżka CSV (ustawia też basename)",
    )
    p.add_argument(
        "--signing-key-path",
        dest="signing_key_path",
        help="Ścieżka do klucza HMAC używanego do podpisów CSV/PDF/JSON",
    )
    p.add_argument(
        "--signing-key-id",
        dest="signing_key_id",
        default=None,
        help="Identyfikator klucza podpisującego umieszczany w dokumencie podpisu",
    )
    p.add_argument(
        "--cost-limit-bps",
        type=float,
        default=None,
        help="Opcjonalny limit kosztów w bps do generowania alertów",
    )
    p.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Dodatkowe metadane w formacie klucz=wartość",
    )
    p.set_defaults(_handler=_handle_analyze)
    return p


def _load_events(path: Path) -> list[TradeCostEvent]:
    contents = path.read_text(encoding="utf-8")
    events: list[TradeCostEvent] = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Nieprawidłowy JSON w {path}:{line_number}") from exc
        events.append(TradeCostEvent.from_mapping(payload))
    return events


def _parse_metadata(items: Iterable[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Wpis metadanych musi mieć postać klucz=wartość: {item}")
        key, value = item.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _read_signing_key_path(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        raise ValueError("Ścieżka do klucza nie może być katalogiem")
    if path.is_symlink():
        raise ValueError("Ścieżka do klucza nie może być symlinkiem")
    data = path.read_bytes()
    if len(data) < 32:
        raise ValueError("Klucz HMAC musi mieć co najmniej 32 bajty")
    return data


def _handle_analyze(args: argparse.Namespace) -> int:
    if getattr(args, "output", None):
        output_path = Path(args.output).expanduser()
        if output_path.suffix.lower() == ".csv":
            args.output_dir = str(output_path.parent or Path("."))
            args.basename = output_path.stem or args.basename or "tco_report"
        else:
            args.output_dir = str(output_path)

    fill_paths = [Path(item).expanduser() for item in args.fills]
    events: list[TradeCostEvent] = []
    for candidate in fill_paths:
        events.extend(_load_events(candidate))

    metadata = _parse_metadata(args.metadata)
    metadata.setdefault("source", "run_tco_analysis")
    metadata.setdefault("inputs", [str(path) for path in fill_paths])

    cost_limit = Decimal(str(args.cost_limit_bps)) if args.cost_limit_bps is not None else None
    analyzer = TCOAnalyzer(cost_limit_bps=cost_limit)
    report = analyzer.analyze(events, metadata=metadata)

    writer = TCOReportWriter(report)
    output_root = Path(args.output_dir).expanduser()
    if args.basename is None:
        timestamp_dir = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
        target_dir = output_root / timestamp_dir
    else:
        target_dir = output_root
    artifacts = writer.write_outputs(target_dir, basename=args.basename)

    if args.signing_key_path:
        signing_key = _read_signing_key_path(Path(args.signing_key_path).expanduser())
        writer.sign_artifacts(artifacts, signing_key=signing_key, key_id=args.signing_key_id)
    else:
        print("Ostrzeżenie: brak klucza podpisującego – raport zostanie zapisany bez podpisów", file=sys.stderr)

    print(f"Wygenerowano raport TCO w katalogu {target_dir}")
    print(f"Liczba transakcji: {report.metadata.get('events_count', 0)}")
    if report.alerts:
        print("Alerty:")
        for alert in report.alerts:
            print(f" - {alert}")
    return 0


# ==============================================================================
# Główny parser i wejście
# ==============================================================================

_SUMMARY_FLAGS = {
    "--input",
    "--artifact-root",
    "--json-name",
    "--csv-name",
    "--signature-name",
    "--monthly-trades",
    "--monthly-volume",
    "--signing-key",
    "--signing-key-file",
    "--signing-key-id",
    "--tag",
    "--timestamp",
    "--print-summary",
    "--output",
}

_ANALYZE_FLAGS = {
    "--fills",
    "--output-dir",
    "--basename",
    "--signing-key-path",
    "--cost-limit-bps",
    "--metadata",
}


def _detect_default_command(argv: Sequence[str]) -> Optional[str]:
    if any(token in {"-h", "--help"} for token in argv):
        return None
    for token in argv:
        if token.startswith("-"):
            continue
        if token in {"summary", "analyze"}:
            return None
        break

    flags = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
    if flags & _ANALYZE_FLAGS:
        return "analyze"
    if flags & _SUMMARY_FLAGS or argv:
        return "summary"
    return "summary"


def _inject_default_command(argv: Sequence[str], default_cmd: Optional[str]) -> Sequence[str]:
    if not argv or default_cmd is None:
        return argv
    for token in argv:
        if token.startswith("-"):
            continue
        if token in {"summary", "analyze"}:
            return argv
        break
    return (default_cmd, *argv)


def _build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    effective_argv: Sequence[str] = tuple(argv or ())
    default_cmd = _detect_default_command(effective_argv)

    parser = argparse.ArgumentParser(
        description=(
            "Stage5 TCO – CLI łączące warianty summary (HEAD) i analyze (main)."
            " Zgodność wsteczna: brak subkomendy = tryb summary."
        )
    )
    sub = parser.add_subparsers(dest="_cmd", metavar="{summary|analyze}")
    sub.required = default_cmd is None
    summary_parser = _build_summary_parser(sub)
    analyze_parser = _build_analyze_parser(sub)
    if not effective_argv:
        mirrored: set[str] = set()
        for action in (*summary_parser._actions, *analyze_parser._actions):
            appended = False
            for option in getattr(action, "option_strings", []) or []:
                if option in {"-h", "--help"}:
                    continue
                if option not in mirrored:
                    if not appended:
                        parser._actions.append(action)
                        appended = True
                    mirrored.add(option)
    setattr(parser, "_default_cmd", default_cmd)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        effective_argv: Sequence[str] = tuple(sys.argv[1:])
    else:
        effective_argv = tuple(argv)
    parser = _build_parser(effective_argv)
    normalized_argv = _inject_default_command(effective_argv, getattr(parser, "_default_cmd", None))
    args = parser.parse_args(normalized_argv)
    return args._handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Narzędzia do odnajdywania raportów Market Intel w skryptach Stage6."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def resolve_market_intel_path(
    expanded: Path | None,
    raw_value: str | Path | None,
    *,
    environment: str,
    governor: str,
    fallback_directories: Sequence[Path] | None = None,
    log_context: str = "stage6.market_intel",
) -> Path:
    """Zwraca ścieżkę do raportu Market Intel na podstawie pliku, katalogu lub wzorca.

    Funkcja potrafi rozwiązać zarówno bezpośrednie wskazanie pliku, katalog z raportami,
    jak i wzorzec wykorzystujący symboliczny `<timestamp>`. Gdy raport nie istnieje,
    zgłaszany jest `FileNotFoundError` wraz z podpowiedzią komendy budującej metryki.
    """

    default_pattern = Path("var/market_intel") / f"market_intel_{governor}_<timestamp>.json"

    if expanded is None:
        expanded = default_pattern
        raw_value = raw_value or default_pattern.as_posix()

    expanded = expanded.expanduser()
    if expanded.is_file():
        return expanded

    search_candidates: list[Path] = []
    seen_roots: set[tuple[str, str]] = set()

    def _append_candidates(root: Path, pattern: str) -> None:
        key = (str(root), pattern)
        if key in seen_roots:
            return
        seen_roots.add(key)
        if not root.exists() or not root.is_dir():
            return
        matches = sorted(root.glob(pattern), key=_safe_mtime, reverse=True)
        search_candidates.extend(match for match in matches if match.is_file())

    if expanded.is_dir():
        pattern_name = f"market_intel_{governor}_*.json"
        _append_candidates(expanded, pattern_name)
        fallback_pattern = pattern_name
    else:
        pattern_hint = str(raw_value or expanded)
        placeholder_pattern = None
        if "<timestamp>" in pattern_hint:
            placeholder_pattern = pattern_hint.replace("<timestamp>", "*")

        if "*" in pattern_hint or placeholder_pattern:
            pattern_path = Path(placeholder_pattern or pattern_hint).expanduser()
            _append_candidates(pattern_path.parent, pattern_path.name)
            fallback_pattern = pattern_path.name
        else:
            fallback_pattern = f"market_intel_{governor}_*.json"

    if fallback_directories:
        for directory in fallback_directories:
            if directory:
                _append_candidates(directory.expanduser(), fallback_pattern)

    for candidate in search_candidates:
        if candidate.is_file():
            print(
                f"[{log_context}] Wybrano raport Market Intel z wzorca "
                f"{raw_value or expanded} -> {candidate}",
                file=sys.stderr,
            )
            return candidate

    fallback_file = Path("var/audit/stage6/market_intel.json").expanduser()
    if fallback_file.exists():
        print(
            f"[{log_context}] Brak raportu Market Intel {raw_value or expanded} – "
            f"używam awaryjnej kopii {fallback_file}.",
            file=sys.stderr,
        )
        return fallback_file

    default_pattern_display = default_pattern.as_posix()
    suggestion_output = f"var/market_intel/market_intel_{governor}_$(date -u +%Y%m%dT%H%M%SZ).json"
    suggestion = (
        "python scripts/build_market_intel_metrics.py "
        f"--environment {environment} --governor {governor} "
        f"--output \"{suggestion_output}\""
    )

    details = str(raw_value or expanded)
    raise FileNotFoundError(
        (
            "Raport Market Intel nie istnieje ani nie znaleziono go w podanym "
            f"wzorcu/katalogu ({details}). Uruchom {suggestion}. "
            f"Domyślny wzorzec nazwy pliku to {default_pattern_display} (UTC)."
        )
    )

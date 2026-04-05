from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ACQUISITION_DOC = REPO_ROOT / "docs/architecture/acquisition_package_stage6.md"
README = REPO_ROOT / "README.md"


REQUIRED_SECTIONS = (
    "## 1) High-level architecture (aktualny runtime)",
    "## 2) Module boundaries (granice modułów)",
    "## 3) Kluczowy przepływ end-to-end: config → strategy → execution → risk → observability",
    "## 4) Kluczowe bounded contexts",
    "## 5) Najważniejsze ryzyka + mitigacje (z kodu)",
    "## 6) Opis operacyjny: co jest prod-minded vs. wymaga uwagi",
    "## 7) Co już zrefaktorowane vs. dalsza praca",
)

REQUIRED_ANCHORS = (
    "bot_core.runtime.bootstrap",
    "PipelineConfigLoader",
    "StrategyBootstrapper",
    "ExecutionBootstrapper",
    "RiskBootstrapper",
    "LiveExecutionRouter",
    "build_alert_channels",
)

SYMBOL_LOCATIONS: tuple[tuple[str, str], ...] = (
    ("bot_core/runtime/observability.py", "def build_alert_channels("),
    ("bot_core/runtime/bootstrap.py", "def build_alert_channels("),
    ("bot_core/runtime/pipeline_config_loader.py", "class PipelineConfigLoader"),
    ("bot_core/runtime/strategy_bootstrapper.py", "class StrategyBootstrapper"),
    ("bot_core/runtime/execution_bootstrapper.py", "class ExecutionBootstrapper"),
    ("bot_core/runtime/risk_bootstrapper.py", "class RiskBootstrapper"),
    ("bot_core/execution/live_router.py", "class LiveExecutionRouter"),
)


def test_acquisition_document_exists() -> None:
    assert ACQUISITION_DOC.exists(), "Brak dokumentu acquisition package Stage6"


def test_readme_links_acquisition_document() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/architecture/acquisition_package_stage6.md" in text


def test_acquisition_document_contains_required_sections_and_anchors() -> None:
    text = ACQUISITION_DOC.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, f"Brak wymaganej sekcji: {section}"
    for anchor in REQUIRED_ANCHORS:
        assert anchor in text, f"Brak wymaganego anchora/symbolu: {anchor}"


def test_referenced_symbols_exist_in_repository() -> None:
    for rel_path, symbol in SYMBOL_LOCATIONS:
        target = REPO_ROOT / rel_path
        assert target.exists(), f"Brak pliku referencyjnego: {rel_path}"
        body = target.read_text(encoding="utf-8")
        assert symbol in body, f"Brak symbolu '{symbol}' w pliku {rel_path}"

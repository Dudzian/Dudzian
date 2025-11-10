from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def marketing_docs() -> dict[str, Path]:
    return {
        "whitepaper": REPO_ROOT / "docs/marketing/stage6_stress_lab_whitepaper.md",
        "case_studies": REPO_ROOT / "docs/marketing/stage6_stress_lab_case_studies.md",
        "benchmark": REPO_ROOT / "docs/benchmark/cryptohopper_comparison.md",
    }


def test_marketing_documents_exist(marketing_docs: dict[str, Path]) -> None:
    missing = [name for name, path in marketing_docs.items() if not path.exists()]
    assert not missing, f"Brak wymaganych dokumentów marketingowych: {missing}"


def _extract_footnote_paths(text: str) -> set[str]:
    pattern = re.compile(r"\[\^[^\]]+\]: `([^`]+)`")
    return {match.group(1) for match in pattern.finditer(text)}


def test_marketing_footnotes_point_to_existing_files(marketing_docs: dict[str, Path]) -> None:
    for path in marketing_docs.values():
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for target in _extract_footnote_paths(text):
            target_path = (REPO_ROOT / target).resolve()
            assert target_path.exists(), f"Dokument {path} odwołuje się do nieistniejącej ścieżki: {target}"


def test_benchmark_references_latest_marketing(marketing_docs: dict[str, Path]) -> None:
    benchmark_path = marketing_docs["benchmark"]
    text = benchmark_path.read_text(encoding="utf-8")
    assert "Stress Lab i materiały marketingowe" in text
    assert "../marketing/stage6_stress_lab_whitepaper.md" in text
    assert "../marketing/stage6_stress_lab_case_studies.md" in text


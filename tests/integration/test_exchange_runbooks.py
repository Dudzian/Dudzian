from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "exchange, expected_fragments",
    [
        (
            "kraken",
            [
                "# Kraken – Runbook go-live",
                "## Runbook go-live",
                "## Checklist licencyjna",
                "paper_exchange_metrics",
                "tests/exchanges/test_kraken_signing.py",
                "execution.paper_profiles.kraken_paper",
                "execution.trading_profiles.kraken_desktop",
            ],
        ),
        (
            "okx",
            [
                "# OKX – Runbook go-live",
                "## Checklist specyficzna",
                "paper_exchange_metrics.okx_desktop_paper",
                "tests/integration/exchanges/test_okx.py",
                "tests/exchanges/test_okx_signing.py",
                "execution.paper_profiles.okx_paper",
            ],
        ),
        (
            "bybit",
            [
                "# Bybit – Runbook go-live",
                "## Checklist licencyjna",
                "io_queue.exchanges.bybit_spot",
                "tests/integration/exchanges/test_bybit.py",
                "tests/exchanges/test_bybit_signing.py",
                "execution.paper_profiles.bybit_paper",
            ],
        ),
    ],
)
def test_runbook_documents(exchange: str, expected_fragments: list[str]) -> None:
    path = Path("docs/deployment") / f"{exchange}_go_live.md"
    assert path.exists(), f"Brak runbooka dla {exchange}"
    content = path.read_text(encoding="utf-8")
    for fragment in expected_fragments:
        assert fragment in content, f"Nie znaleziono fragmentu '{fragment}' w runbooku {exchange}"
    assert "pytest tests/integration/exchanges/test_" + exchange in content

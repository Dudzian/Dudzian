from __future__ import annotations

from pathlib import Path


_REQUIRED_NODE_IDS = (
    "tests/integration/exchanges/test_binance.py::test_binance_spot_rate_limit_and_retry",
    "tests/integration/exchanges/test_kraken.py::test_kraken_spot_rate_limit_and_retry",
    "tests/integration/exchanges/test_okx.py::test_okx_spot_rate_limit",
    "tests/integration/test_exchange_manager_failover.py::test_exchange_manager_failover_to_ccxt_backend",
    "tests/integration/test_exchange_manager_failover.py::test_exchange_manager_stays_on_ccxt_after_rate_limit",
    "tests/exchanges/test_ccxt_private_backend_contracts.py::test_create_order_maps_response",
    "tests/exchanges/test_ccxt_private_backend_contracts.py::test_fetch_open_orders_maps_fields",
    "tests/integration/test_exchange_runbooks.py",
)

_REQUIRED_VERDICTS = (
    "PASS",
    "FAIL",
    "NOT EXECUTED",
    "SKIPPED",
    "BLOCKED",
    "ENVIRONMENTAL LIMITATION",
)


def test_sandbox_proof_pack_contains_contract_sections_and_node_ids() -> None:
    path = Path("docs/deployment/sandbox_proof_pack.md")
    assert path.exists(), "Brak dokumentu sandbox proof pack"
    content = path.read_text(encoding="utf-8")

    required_sections = (
        "## 1) Runbook uruchomień sandboxowych",
        "### 1.2 Jawna klasyfikacja coverage/verdict",
        "### 1.3 Evidence base (zweryfikowane node IDs)",
        "### 1.6 Obecny stan coverage (uczciwy snapshot, aktualizuj per run)",
        "## 3) Raport/template wyników",
        "## 4) Lekkie integration scaffolding",
    )
    for section in required_sections:
        assert section in content, f"Brak sekcji '{section}'"

    for node_id in _REQUIRED_NODE_IDS:
        assert node_id in content, f"Brak node ID '{node_id}' w runbooku"

    for verdict in _REQUIRED_VERDICTS:
        assert verdict in content, f"Runbook nie zawiera klasyfikacji '{verdict}'"


def test_sandbox_report_template_enforces_adapter_area_verdict_evidence() -> None:
    template_path = Path("reports/templates/sandbox_proof_report_template.md")
    assert template_path.exists(), "Brak template raportu sandbox proof"
    content = template_path.read_text(encoding="utf-8")

    assert "## 2. Verdict taxonomy (must use one)" in content
    assert "## 4. Adapter x area verdict matrix (required)" in content
    assert "Test / node reference" in content
    assert "Artifact path" in content
    assert "Rationale" in content

    for verdict in _REQUIRED_VERDICTS:
        assert verdict in content, f"Template nie zawiera statusu '{verdict}'"

    for adapter in ("Binance", "Kraken", "OKX"):
        for area in ("Stability", "Rate limits", "Failover", "Partial fills", "Recovery"):
            needle = f"| {adapter} | {area} |"
            assert needle in content, f"Brak wiersza '{needle}' w macierzy verdictów"

    assert "## 7. Gaps / blocked evidence" in content
    assert "NOT EXECUTED/SKIPPED/BLOCKED/ENVIRONMENTAL LIMITATION" in content

"""Sonda używana przez emisję nie importuje producentów dowodów platformy."""

import ast
from pathlib import Path

from deployment.windows_stage8_evidence_contract import WINDOWS_STAGE8_ITEMS
from tests.architecture.test_stage9_root_proof_signed_immutable_attempt_contract import (
    _source_imports,
)

ROOT = Path(__file__).resolve().parents[2]


def test_stage8_shared_identifiers_have_no_executable_dependencies():
    path = "deployment/windows_stage8_evidence_contract.py"
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    assert not _source_imports(tree, path)
    assert WINDOWS_STAGE8_ITEMS == (
        "WINDOWS_POSTGRESQL_SUBSTRATE",
        "WINDOWS_LOCAL_PRINCIPAL_AUTHENTICATION",
    )


def test_stage8_probe_imports_data_contract_without_evidence_producers():
    path = "deployment/windows_stage8_postgresql_probe.py"
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    local_imports = {
        name
        for name in _source_imports(tree, path)
        if name.startswith("deployment.") and (ROOT / (name.replace(".", "/") + ".py")).is_file()
    }
    assert local_imports == {
        "deployment.host_identity",
        "deployment.windows_stage8_evidence_contract",
    }


def test_evidence_producer_preserves_shared_stage8_identifiers():
    from deployment.platform_evidence import WINDOWS_STAGE8_ITEMS as producer_items

    assert producer_items is WINDOWS_STAGE8_ITEMS

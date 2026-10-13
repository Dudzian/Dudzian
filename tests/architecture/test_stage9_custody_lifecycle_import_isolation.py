"""Concrete lifecycle imports preserve the existing local transport review gate."""

import ast
from pathlib import Path

import pytest

from tests.architecture.test_stage9_root_proof_signed_immutable_attempt_contract import (
    _TRANSPORT_IMPORTS,
    _assert_local_issuance_execution_ownership,
    _source_imports,
)

ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE = "bot_core/postgresql_custody_lifecycle.py"
NEW_MODULES = (
    LIFECYCLE,
    "bot_core/local_signing_lifecycle_evidence.py",
    "bot_core/postgresql_custody_lifecycle_schema.py",
    "bot_core/postgresql_custody_lifecycle_qualification.py",
)


@pytest.mark.parametrize("path", NEW_MODULES)
def test_lifecycle_modules_name_concrete_dependencies_without_root_facade(path):
    imports = _source_imports(ast.parse((ROOT / path).read_text(encoding="utf-8")), path)
    assert "bot_core" not in imports
    if path == LIFECYCLE:
        assert "bot_core.local_signing_custody" in imports


def _sources(import_statement, transport, *, selected_transport=False):
    return {
        "bot_core/__init__.py": "from bot_core.observability.server import MetricsHTTPServer\n",
        "bot_core/observability/server.py": f"import {transport}\n",
        LIFECYCLE: import_statement + "\n",
        "bot_core/local_signing_custody.py": (
            "from bot_core.cha_attempt_signatures import RetainedIssuanceRequest\n"
            + (f"import {transport}\n" if selected_transport else "")
        ),
    }


@pytest.mark.parametrize("transport", sorted(_TRANSPORT_IMPORTS))
def test_selected_lifecycle_import_avoids_independent_facade_transport(transport):
    _assert_local_issuance_execution_ownership(
        _sources("import bot_core.local_signing_custody as custody", transport)
    )


@pytest.mark.parametrize("transport", sorted(_TRANSPORT_IMPORTS))
def test_lifecycle_root_facade_regression_remains_rejected(transport):
    with pytest.raises(
        AssertionError, match="issuer transport bypass in bot_core.observability.server"
    ):
        _assert_local_issuance_execution_ownership(
            _sources("from bot_core import local_signing_custody as custody", transport)
        )


@pytest.mark.parametrize("transport", sorted(_TRANSPORT_IMPORTS))
def test_selected_custody_dependency_cannot_supply_issuer_transport(transport):
    with pytest.raises(
        AssertionError, match="issuer transport bypass in bot_core.local_signing_custody"
    ):
        _assert_local_issuance_execution_ownership(
            _sources(
                "import bot_core.local_signing_custody as custody",
                transport,
                selected_transport=True,
            )
        )

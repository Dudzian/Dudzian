"""History reproposal stays inside the isolated atomic authority boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.architecture.test_stage9_atomic_root_proof_commit_boundary import (
    PUBLIC_ENTRY_POINTS,
    _repository_dependencies,
)

ROOT = Path(__file__).resolve().parents[2]
INTERNAL_MODULES = frozenset(
    "bot_core.postgresql_history_successor_reproposal" + suffix
    for suffix in ("", "_schema", "_qualification", "_evidence")
)
PUBLIC_READERS = (
    "bot_core.postgresql_root_proof_preparation_v2",
    "bot_core.postgresql_root_proof_preparation_readonly",
    "bot_core.postgresql_root_proof_issuance_authority_v2",
)


@pytest.mark.parametrize("entry_point", PUBLIC_ENTRY_POINTS + PUBLIC_READERS)
def test_public_imports_cannot_reach_history_reproposal_administration(entry_point):
    pending, visited = [entry_point], set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        assert module not in INTERNAL_MODULES, (entry_point, module)
        pending.extend(_repository_dependencies(module) - visited)


def test_reproposal_runtime_cannot_mint_or_sign_another_root_proof():
    tree = ast.parse(
        (ROOT / "bot_core/postgresql_history_successor_reproposal.py").read_text(encoding="utf-8")
    )
    forbidden = {
        "mint_uuid7",
        "reserve",
        "reserve_intent",
        "retain_prepared",
        "sign_root_proof",
        "sign_root_proof_issuance_attempt",
        "reserve_root_proof_issuance_attempt",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            assert name not in forbidden, f"reproposal invokes {name}"


def test_reproposal_does_not_weaken_reserved_staged_uniqueness():
    from tests.architecture.test_stage9_postgresql_v2_root_proof_preparation import (
        test_legacy_unique_constraints_still_reject_two_attempts_of_one_entitlement,
    )

    test_legacy_unique_constraints_still_reject_two_attempts_of_one_entitlement()

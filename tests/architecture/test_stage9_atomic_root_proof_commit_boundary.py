"""Production entry points cannot reach the internal atomic write authority."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from bot_core.postgresql_atomic_root_proof_preaccount_sql import reviewed_preaccount_helper

ROOT = Path(__file__).parents[2]
INTERNAL_WRITE_AUTHORITY = "bot_core.postgresql_atomic_root_proof_commit"
PUBLIC_ENTRY_POINTS = (
    "bot_core.root_proof_issuer_runtime",
    "bot_core.root_proof_issuer_orchestration",
    "bot_core.root_proof_issuer_recovery",
)


def _repository_dependencies(module: str) -> set[str]:
    """Read actual import statements, including relative package imports."""
    path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    if not path.is_file():
        path = ROOT.joinpath(*module.split("."), "__init__.py")
        if not path.is_file():
            return set()
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dependencies = {".".join(package[:count]) for count in range(1, len(package) + 1)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dependencies.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                prefix = ".".join(base + (node.module or "").split(".")).rstrip(".")
            else:
                prefix = node.module or ""
            dependencies.add(prefix)
            dependencies.update(f"{prefix}.{alias.name}" for alias in node.names)
    return {
        dependency
        for dependency in dependencies
        if dependency == "bot_core" or dependency.startswith("bot_core.")
    }


@pytest.mark.parametrize("entry_point", PUBLIC_ENTRY_POINTS)
def test_public_entry_point_import_graph_excludes_atomic_write_authority(entry_point: str):
    pending = [entry_point]
    visited = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        assert module != INTERNAL_WRITE_AUTHORITY, (
            f"public issuer entry point {entry_point} reaches internal write authority"
        )
        pending.extend(_repository_dependencies(module) - visited)


def test_stage9_status_remains_blocked_after_atomic_persistence_implementation():
    status = json.loads((ROOT / "deployment/stage9_current_status.json").read_text())
    assert status["semantic_root_proof_issuer_runtime"] == "NOT_IMPLEMENTED / BLOCKED"
    assert status["root_proof"] == "NOT_ISSUED"
    assert status["account_genesis"] == "INITIAL_BINDING_ONLY"
    assert status["prepared"] == "NOT_STARTED"
    assert status["root_proof_admission"] == "NOT_STARTED"
    assert status["production_provisioning_ready"] is False
    assert status["windows_production_ready"] == "NOT_READY"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "public;SELECT true"),
        ("peer_schema", "peer.schema"),
        ("owner", "owner;SET ROLE postgres"),
        ("writer", None),
        ("semantic_role", "ROOT_PROOF_ISSUER_SIGNING"),
        ("semantic_role", "ROOT_PROOF_REQUESTER';SELECT true"),
        ("trust_domain", ""),
        ("peer_schema", "requester"),
        ("writer", "requester_owner"),
    ],
)
def test_preaccount_helper_rejects_unqualified_fixed_authority_names(field, value):
    configuration = {
        "schema": "requester",
        "peer_schema": "claimant",
        "owner": "requester_owner",
        "writer": "issuer_writer",
        "semantic_role": "ROOT_PROOF_REQUESTER",
        "trust_domain": "trust",
    }
    configuration[field] = value
    with pytest.raises(ValueError):
        reviewed_preaccount_helper(**configuration)

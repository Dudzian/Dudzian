from __future__ import annotations

from bot_core.marketplace import (
    MarketplaceIndex,
    PresetDocument,
    PresetSignatureVerification,
    build_marketplace_preset,
)


def _document(metadata: dict[str, object]) -> PresetDocument:
    payload = {
        "name": metadata.get("name", "demo"),
        "metadata": metadata,
    }
    return PresetDocument(
        payload=payload,
        signature=None,
        verification=PresetSignatureVerification(True, ()),
        fmt="json",
        path=None,
        issues=tuple(),
    )


def test_build_marketplace_preset_parses_dependencies_and_updates() -> None:
    metadata = {
        "id": "alpha",
        "version": "2.0.0",
        "dependencies": [
            "core/base",
            {"preset_id": "risk-kit", "constraints": [">=1.0", "<2.0"], "optional": True, "notes": "Risk module"},
        ],
        "updates": {
            "channels": [
                {"name": "stable", "version": "2.0.0", "released_at": "2025-01-10T00:00:00Z"},
                {"name": "beta", "version": "2.1.0-beta1", "severity": "preview"},
            ],
            "replaces": ["alpha-archive"],
            "requires_approval": True,
            "default_channel": "stable",
        },
    }
    preset = build_marketplace_preset(_document(metadata))

    assert preset.preset_id == "alpha"
    assert preset.version == "2.0.0"
    assert len(preset.dependencies) == 2
    assert preset.dependencies[0].preset_id == "core/base"
    assert preset.dependencies[1].constraints == (">=1.0", "<2.0")
    assert preset.update_channels[0].name == "stable"
    assert preset.update_channels[0].version == "2.0.0"
    assert preset.update_directive.replaces == ("alpha-archive",)
    assert preset.update_directive.requires_approval is True


def test_marketplace_index_plans_dependencies_and_upgrades() -> None:
    alpha_doc = _document(
        {
            "id": "alpha",
            "version": "1.1.0",
            "dependencies": [
                {"preset_id": "beta", "version": ">=1.0.0"},
            ],
            "updates": {"channels": [{"name": "stable", "version": "1.1.0"}]},
        }
    )
    beta_doc = _document(
        {
            "id": "beta",
            "version": "1.2.0",
            "dependencies": [],
            "updates": {"channels": [{"name": "stable", "version": "1.2.0"}]},
        }
    )

    index = MarketplaceIndex.from_documents([alpha_doc, beta_doc])
    plan = index.plan_installation(["alpha"], installed_versions={"beta": "1.0.0"})

    assert plan.install_order == ("beta", "alpha")
    assert plan.missing == {}
    assert plan.issues == ()
    assert plan.upgrades[0].preset_id == "beta"
    assert plan.upgrades[0].from_version == "1.0.0"
    assert plan.upgrades[0].to_version == "1.2.0"


def test_marketplace_index_reports_constraint_violation() -> None:
    alpha_doc = _document(
        {
            "id": "alpha",
            "version": "1.0.0",
            "dependencies": [
                {"preset_id": "beta", "constraints": [">=2.0.0"]},
            ],
        }
    )
    beta_doc = _document({"id": "beta", "version": "1.5.0"})

    index = MarketplaceIndex.from_documents([alpha_doc, beta_doc])
    plan = index.plan_installation(["alpha"], installed_versions={"beta": "1.5.0"})

    assert any(issue.startswith("version-constraint") for issue in plan.issues)


def test_marketplace_plan_to_payload_serializes_dependencies() -> None:
    parent = _document({"id": "parent", "version": "2.0.0", "dependencies": ["child"]})
    child = _document({"id": "child", "version": "1.0.0"})

    index = MarketplaceIndex.from_documents([parent, child])
    plan = index.plan_installation(["parent"], installed_versions={})
    payload = plan.to_payload()

    assert payload["installOrder"] == ["child", "parent"]
    assert payload["requiredDependencies"]["parent"][0]["presetId"] == "child"
    assert payload["issues"] == []
    assert payload["upgrades"] == []

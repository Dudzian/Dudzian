"""Testy integracyjne pipeline'u jakości i procesu publikacji pluginów."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from bot_core.plugins import (
    PluginAuthor,
    PluginRegistryError,
    PluginReviewBoard,
    PluginSignature,
    PluginSigner,
    PluginVerifier,
    RegisteredPlugin,
    ReviewStatus,
    SignedStrategyPlugin,
    StrategyPluginManifest,
    StrategyPluginRegistry,
)
from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG
from bot_core.strategies.marketplace import MarketplaceCatalog, load_catalog
from bot_core.strategies.quality_pipeline import StrategyQualityPipeline


def _manifest(identifier: str = "acme.adaptive") -> StrategyPluginManifest:
    return StrategyPluginManifest(
        identifier=identifier,
        version="1.0.0",
        title="Acme Adaptive Pack",
        description="Pakiet strategii adaptacyjnego market makingu",
        author=PluginAuthor(name="Acme Labs", email="ops@acme.test"),
        strategies=("adaptive_market_making",),
        capabilities=("adaptive_mm",),
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        metadata={"category": "market_making"},
    )


def test_plugin_signing_roundtrip_registers_package() -> None:
    signer = PluginSigner(b"super-secret", key_id="unit-test")
    manifest = _manifest()
    package = signer.build_package(manifest, review_notes=["QA: ok"])

    verifier = PluginVerifier(b"super-secret")
    board = PluginReviewBoard(verifier)
    registry = StrategyPluginRegistry(board)

    entry = registry.register(package)

    assert isinstance(entry, RegisteredPlugin)
    assert entry.manifest.identifier == manifest.identifier
    assert entry.signature.key_id == "unit-test"
    assert entry.review.status is ReviewStatus.ACCEPTED
    assert any(
        finding.message.startswith("note:") for finding in entry.review.findings
    ), "Oczekiwano przeniesienia notatek review do wyników"


def test_plugin_registry_rejects_invalid_manifest() -> None:
    manifest = StrategyPluginManifest(
        identifier="broken.plugin",
        version="1.0.0",
        title="Broken",
        description="Missing strategies",
        author=PluginAuthor(name="Broken"),
        strategies=(),
        capabilities=(),
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    signature = PluginSignature(algorithm="HMAC-SHA256", key_id=None, value="invalid")
    package = SignedStrategyPlugin(manifest=manifest, signature=signature)

    board = PluginReviewBoard()
    registry = StrategyPluginRegistry(board)

    with pytest.raises(PluginRegistryError):
        registry.register(package)


def test_plugin_review_reports_invalid_signature() -> None:
    manifest = _manifest(identifier="acme.invalid")
    signer = PluginSigner(b"super-secret", key_id="unit-test")
    package = signer.build_package(manifest)

    verifier = PluginVerifier(b"another-secret")
    board = PluginReviewBoard(verifier)

    result = board.evaluate(package)

    assert result.status is ReviewStatus.REJECTED
    assert any("Podpis manifestu" in finding.message for finding in result.findings)


def test_marketplace_catalog_exposes_new_presets() -> None:
    catalog = load_catalog()
    assert isinstance(catalog, MarketplaceCatalog)

    adaptive = catalog.find("adaptive_mm_enterprise")
    assert adaptive is not None
    assert adaptive.author.name == "Marketplace Labs"
    assert "market-making" in adaptive.tags
    assert adaptive.artifact_path.exists()

    triangular = catalog.find("triangular_arbitrage_pro")
    assert triangular is not None
    assert set(triangular.required_exchanges) == {"BINANCE", "KRAKEN", "OKX"}


def test_quality_pipeline_generates_reports(tmp_path: Path) -> None:
    sandbox_source = Path("data/simulations/strategy_quality_sandbox.yaml")
    sandbox_copy = tmp_path / "strategy_quality_sandbox.yaml"
    sandbox_copy.write_text(sandbox_source.read_text(encoding="utf-8"), encoding="utf-8")

    report_dir = tmp_path / "reports"
    pipeline = StrategyQualityPipeline(
        DEFAULT_STRATEGY_CATALOG,
        simulations_dir=tmp_path,
        report_dir=report_dir,
    )
    reports = pipeline.run()

    scenario_names = {report.scenario.name for report in reports}
    assert scenario_names == {"adaptive_mm_liquidity_cycle", "triangular_arbitrage_multi_venue"}
    assert all(report.passed for report in reports)

    summary_path = report_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total"] == 2
    assert payload["passed"] == 2
    assert payload["failed"] == 0

    for name in scenario_names:
        report_path = report_dir / f"{name}.json"
        assert report_path.exists()
        detail = json.loads(report_path.read_text(encoding="utf-8"))
        assert detail["scenario"]["engine"] in ("adaptive_market_making", "triangular_arbitrage")


def test_quality_pipeline_rejects_unknown_engine(tmp_path: Path) -> None:
    payload = {
        "scenarios": [
            {
                "name": "unknown_engine_case",
                "engine": "non_existing_engine",
                "parameters": {},
            }
        ]
    }
    sandbox = tmp_path / "invalid.yaml"
    sandbox.write_text(yaml.safe_dump(payload), encoding="utf-8")

    pipeline = StrategyQualityPipeline(simulations_dir=tmp_path, report_dir=tmp_path / "reports")

    with pytest.raises(ValueError):
        pipeline.run()

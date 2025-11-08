from datetime import datetime, timezone

import pytest

from bot_core.plugins import (
    PluginAuthor,
    PluginLoadError,
    PluginSigner,
    PluginVerifier,
    StrategyPluginLoader,
    StrategyPluginManifest,
    StrategyPluginRegistry,
    PluginReviewBoard,
)
from bot_core.trading.strategies.plugins import StrategyCatalog


def _manifest(entrypoint: str) -> StrategyPluginManifest:
    return StrategyPluginManifest(
        identifier="acme.plugins.adaptive",
        version="1.2.3",
        title="Acme Adaptive Pack",
        description="Pakiet strategii od Acme do testów integracyjnych.",
        author=PluginAuthor(name="Acme Labs", email="qa@acme.test"),
        strategies=("acme_adaptive_mm",),
        capabilities=("adaptive_mm",),
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        metadata={
            "entry_points": {
                "strategies": {
                    "acme_adaptive_mm": entrypoint,
                }
            }
        },
    )


def test_loader_installs_reviewed_plugin_using_class_entrypoint() -> None:
    catalog = StrategyCatalog()
    board = PluginReviewBoard(PluginVerifier(b"secret"))
    registry = StrategyPluginRegistry(board)
    loader = StrategyPluginLoader(catalog=catalog, registry=registry)

    manifest = _manifest("tests.fixtures.plugins.dummy_strategy:DummyAdaptiveMMPlugin")
    signer = PluginSigner(b"secret", key_id="qa")
    package = signer.build_package(manifest)

    installed = loader.install(package)

    assert installed == ("acme_adaptive_mm",)
    assert "acme_adaptive_mm" in catalog.available()
    plugin = catalog.create("acme_adaptive_mm")
    assert plugin is not None
    assert plugin.description.startswith("Neutralny sygnał")


def test_loader_supports_factory_entrypoint() -> None:
    catalog = StrategyCatalog()
    loader = StrategyPluginLoader(catalog=catalog)

    manifest = _manifest("tests.fixtures.plugins.dummy_strategy:build_dummy_plugin")
    package = PluginSigner(b"secret").build_package(manifest)

    installed = loader.install(package, strict_review=False)

    assert installed == ("acme_adaptive_mm",)
    plugin = catalog.create("acme_adaptive_mm")
    assert plugin is not None
    assert plugin.metadata()["capability"] == "adaptive_mm"


def test_loader_requires_entrypoints_definition() -> None:
    catalog = StrategyCatalog()
    loader = StrategyPluginLoader(catalog=catalog)

    manifest = StrategyPluginManifest(
        identifier="acme.plugins.broken",
        version="0.1.0",
        title="Broken",
        description="Brak entrypointów",
        author=PluginAuthor(name="Acme"),
        strategies=("acme_adaptive_mm",),
        capabilities=("adaptive_mm",),
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        metadata={},
    )
    package = PluginSigner(b"secret").build_package(manifest)

    with pytest.raises(PluginLoadError):
        loader.install(package)


def test_loader_rejects_packages_needing_changes() -> None:
    catalog = StrategyCatalog()
    board = PluginReviewBoard(PluginVerifier(b"secret"))
    registry = StrategyPluginRegistry(board)
    loader = StrategyPluginLoader(catalog=catalog, registry=registry)

    manifest = StrategyPluginManifest(
        identifier="acme.plugins.pending",
        version="0.9.0",
        title="Pending",
        description="Manifest bez capabilities powodujący ostrzeżenia.",
        author=PluginAuthor(name="Acme"),
        strategies=("acme_adaptive_mm",),
        capabilities=(),
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        metadata={
            "entry_points": {
                "strategies": {
                    "acme_adaptive_mm": "tests.fixtures.plugins.dummy_strategy:DummyAdaptiveMMPlugin"
                }
            }
        },
    )
    package = PluginSigner(b"secret").build_package(manifest)

    with pytest.raises(PluginLoadError):
        loader.install(package)

    assert "acme_adaptive_mm" not in catalog.available()

"""Testy CLI wspierającego dostawców pluginów."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot_core.plugins import (
    PluginAuthor,
    PluginSigner,
    StrategyPluginManifest,
    dump_manifest,
    dump_package,
    load_manifest,
    load_package,
)
from bot_core.plugins.cli import main as cli_main


def _manifest() -> StrategyPluginManifest:
    return StrategyPluginManifest(
        identifier="vendor.alpha",
        version="1.2.3",
        title="Alpha Pack",
        description="Zestaw strategii alfa",
        author=PluginAuthor(name="Vendor", email="ops@vendor.test"),
        strategies=("adaptive_market_making", "triangular_arbitrage"),
        capabilities=("market-making",),
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        metadata={"category": "pro"},
    )


def test_manifest_io_roundtrip(tmp_path) -> None:
    manifest = _manifest()
    path = tmp_path / "manifest.yaml"
    dump_manifest(manifest, path)

    loaded = load_manifest(path)

    assert loaded.identifier == manifest.identifier
    assert loaded.capabilities == manifest.capabilities
    assert loaded.created_at == manifest.created_at


def test_package_io_roundtrip(tmp_path) -> None:
    manifest = _manifest()
    package_path = tmp_path / "package.json"

    signer = PluginSigner(b"secret", key_id="test")
    package = signer.build_package(manifest, review_notes=["QA ok"])
    dump_package(package, package_path)

    loaded = load_package(package_path)
    assert loaded.signature.key_id == "test"
    assert "QA ok" in loaded.review_notes


def test_cli_sign_and_verify(tmp_path, capsys) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest().to_json(), encoding="utf-8")

    package_path = tmp_path / "package.json"

    with pytest.raises(SystemExit) as sign_exit:
        cli_main([
            "sign",
            "--manifest",
            str(manifest_path),
            "--key",
            "top-secret",
            "--key-id",
            "cli-test",
            "--output",
            str(package_path),
            "--note",
            "QA:passed",
        ])
    assert sign_exit.value.code == 0

    with pytest.raises(SystemExit) as verify_exit:
        cli_main([
            "verify",
            "--package",
            str(package_path),
            "--key",
            "top-secret",
        ])
    assert verify_exit.value.code == 0
    output = capsys.readouterr().out
    assert "OK" in output.splitlines()[-1]


def test_cli_verify_failure(tmp_path, capsys) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest().to_json(), encoding="utf-8")
    package = tmp_path / "package.json"

    signer = PluginSigner(b"secret")
    dump_package(signer.build_package(_manifest()), package)

    with pytest.raises(SystemExit) as exc:
        cli_main([
            "verify",
            "--package",
            str(package),
            "--key",
            "invalid",
        ])
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "INVALID" in output.splitlines()[-1]


def test_cli_describe_outputs_metadata(tmp_path, capsys) -> None:
    manifest = _manifest()

    signer = PluginSigner(b"secret")
    package = signer.build_package(manifest, review_notes=["QA: ok"])
    package_path = tmp_path / "package.json"
    package_path.write_text(package.to_json(), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli_main([
            "describe",
            "--package",
            str(package_path),
        ])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "id" in output
    assert "QA: ok" in output


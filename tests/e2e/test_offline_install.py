from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from bot_core.backtest.simulation import MatchingConfig
from bot_core.runtime.paper_trading import PaperTradingAdapter
from bot_core.security.license_store import LicenseStore
from scripts.build.desktop_distribution import build_distribution, DesktopBuildError


def _create_sample_tree(root: Path, entries: dict[str, str]) -> None:
    for relative, content in entries.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def test_build_distribution_and_paper_trade(tmp_path: Path) -> None:
    runtime_src = tmp_path / "runtime_src"
    ui_src = tmp_path / "ui_src"
    extras_src = tmp_path / "extras_src"
    runtime_src.mkdir()
    ui_src.mkdir()
    extras_src.mkdir()

    _create_sample_tree(runtime_src, {"bin/start.sh": "#!/bin/sh\necho start"})
    _create_sample_tree(ui_src, {"Main.qml": "import QtQuick 2.15"})
    _create_sample_tree(extras_src, {"config/settings.json": json.dumps({"paper": True})})

    license_json = tmp_path / "license.json"
    license_payload = {
        "license_id": "TEST-LIC-001",
        "profile": "desktop.pro",
        "issuer": "unit-test",
        "schema": "core.oem.license",
        "schema_version": 1,
        "issued_at": "2024-01-01T00:00:00Z",
        "expires_at": "2026-01-01T00:00:00Z",
    }
    license_json.write_text(json.dumps(license_payload, ensure_ascii=False), encoding="utf-8")

    archive = build_distribution(
        version="1.0.0",
        platform="linux",
        runtime_dir=runtime_src,
        ui_dir=ui_src,
        includes=["resources_extra=" + str(extras_src)],
        license_json=license_json,
        license_fingerprint="FAKE-FP-123",
        output_dir=tmp_path,
        signing_key="hex:313233343536",  # "123456" w hex
    )

    install_root = tmp_path / "install"
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(install_root)

    bundle_dir = next(install_root.iterdir())
    license_store_path = bundle_dir / "resources" / "license" / "license_store.json"
    store = LicenseStore(path=license_store_path, fingerprint_override="FAKE-FP-123")
    document = store.load()
    assert "TEST-LIC-001" in document.data["licenses"]

    adapter = PaperTradingAdapter(initial_balance=1_000.0, matching=MatchingConfig(latency_bars=0))
    adapter.submit_order(symbol="BTC/USDT", side="buy", size=0.5)
    adapter.update_market_data(
        "BTC/USDT",
        "1m",
        {"ohlcv": {"open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0, "volume": 5.0}},
    )
    snapshot = adapter.portfolio_snapshot("BTC/USDT")
    assert snapshot["position"] != 0.0


def test_missing_runtime_directory(tmp_path: Path) -> None:
    ui_src = tmp_path / "ui_src"
    ui_src.mkdir()
    license_json = tmp_path / "license.json"
    license_json.write_text(json.dumps({"license_id": "L"}), encoding="utf-8")

    with pytest.raises(DesktopBuildError):
        build_distribution(
            version="1.0.0",
            platform="linux",
            runtime_dir=tmp_path / "missing",
            ui_dir=ui_src,
            includes=[],
            license_json=license_json,
            license_fingerprint="FP",
            output_dir=tmp_path,
        )

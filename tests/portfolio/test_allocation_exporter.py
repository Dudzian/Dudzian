from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bot_core.portfolio.allocation_exporter import (
    PortfolioAllocationExportError,
    export_allocations_for_governor_config,
    export_allocations_from_core_config,
)


class _Asset(SimpleNamespace):
    pass


class _Governor(SimpleNamespace):
    pass


def test_export_allocations_for_governor_config_writes_yaml(tmp_path: Path) -> None:
    governor = _Governor(
        name="stage6_core",
        portfolio_id="stage6_core",
        assets=[
            _Asset(symbol="ETH_USDT", target_weight=0.4),
            _Asset(symbol="BTC_USDT", target_weight=0.6),
        ],
    )

    output = tmp_path / "allocations.yaml"
    document = export_allocations_for_governor_config(governor, output)

    assert document.path == output
    assert output.exists()
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload == {"BTC_USDT": 0.6, "ETH_USDT": 0.4}


def test_export_allocations_for_governor_config_rejects_missing_symbol(tmp_path: Path) -> None:
    governor = _Governor(
        assets=[
            _Asset(symbol="BTC_USDT", target_weight=0.5),
            _Asset(symbol=None, target_weight=0.5),
        ],
    )

    with pytest.raises(PortfolioAllocationExportError):
        export_allocations_for_governor_config(governor, tmp_path / "allocations.yaml")


def test_export_allocations_from_core_config_integration(tmp_path: Path) -> None:
    core_dir = tmp_path / "config"
    core_dir.mkdir()
    core_yaml = core_dir / "core.yaml"
    core_yaml.write_text(
        """
environments:
  binance_paper:
    exchange: binance
    environment: paper
    keychain_key: local
    data_cache_path: var/cache/binance_paper
    risk_profile: balanced
    alert_channels: []
    adapter_settings: {}
    adapter_factories: {}
portfolio_governors:
  stage6_core:
    portfolio_id: stage6_core
    assets:
      - symbol: BTC_USDT
        target_weight: 0.55
      - symbol: ETH_USDT
        target_weight: 0.45
""",
        encoding="utf-8",
    )

    output_path = tmp_path / "var/audit/portfolio/allocations_stage6.yaml"
    document = export_allocations_from_core_config(
        core_yaml,
        "stage6_core",
        output_path=output_path,
        environment="binance_paper",
    )

    assert document.path == output_path
    assert output_path.exists()
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload == {"BTC_USDT": 0.55, "ETH_USDT": 0.45}


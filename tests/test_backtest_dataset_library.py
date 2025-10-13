from pathlib import Path
from textwrap import dedent

import pytest

from bot_core.data.backtest_library import (
    BacktestDatasetLibrary,
    DataQualityValidator,
    DatasetNotFoundError,
)


@pytest.fixture(scope="module")
def library() -> BacktestDatasetLibrary:
    manifest = Path("data/backtests/normalized/manifest.yaml")
    return BacktestDatasetLibrary(manifest)


def test_manifest_lists_expected_datasets(library: BacktestDatasetLibrary) -> None:
    dataset_names = library.list_dataset_names()
    assert set(dataset_names) == {
        "mean_reversion",
        "volatility_target",
        "cross_exchange_arbitrage",
    }


def test_descriptor_contains_metadata(library: BacktestDatasetLibrary) -> None:
    descriptor = library.describe("mean_reversion")
    assert descriptor.interval == "1m"
    assert descriptor.timezone == "UTC"
    assert descriptor.strategies == ("mean_reversion",)
    assert "open" in descriptor.schema
    assert descriptor.file_path.exists()


def test_loading_typed_rows_respects_schema(library: BacktestDatasetLibrary) -> None:
    rows = library.load_typed_rows("volatility_target")
    assert len(rows) == 6
    assert isinstance(rows[0]["timestamp"], int)
    assert rows[0]["target_volatility"] == pytest.approx(0.2)


def test_interval_conversion(library: BacktestDatasetLibrary) -> None:
    assert library.interval_to_seconds("1m") == 60
    assert library.interval_to_seconds("2m") == 120
    with pytest.raises(ValueError):
        library.interval_to_seconds("15x")


def test_unknown_dataset_raises_error(library: BacktestDatasetLibrary) -> None:
    with pytest.raises(DatasetNotFoundError):
        library.describe("unknown")


def test_validator_reports_success(library: BacktestDatasetLibrary) -> None:
    validator = DataQualityValidator(library)
    report = validator.validate("cross_exchange_arbitrage")
    assert report.is_passing
    assert report.row_count == 6
    assert report.start_timestamp < report.end_timestamp


def test_validator_detects_schema_and_value_issues(tmp_path: Path) -> None:
    dataset = tmp_path / "faulty.csv"
    dataset.write_text(
        """timestamp,instrument,exchange,open,high,low,close,volume,z_score\n"
        "1706652000,BTC-USD,DEMEX,42000.0,42050.0,41980.0,42020.0,-5.0,4.5\n""",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        dedent(
            """
            version: 1
            interval_units:
              m: 60
            datasets:
              faulty:
                file: faulty.csv
                interval: 1m
                timezone: UTC
                strategies: [mean_reversion]
                risk_profiles: [balanced]
                schema:
                  timestamp: int
                  instrument: str
                  exchange: str
                  open: float
                  high: float
                  low: float
                  close: float
                  volume: float
                  z_score: float
                checks:
                  price_columns: [open, high, low, close]
                  volume_columns: [volume]
                  bounded_columns:
                    z_score:
                      min: -3.0
                      max: 3.0
            """
        ),
        encoding="utf-8",
    )
    library = BacktestDatasetLibrary(manifest)
    validator = DataQualityValidator(library)
    report = validator.validate("faulty")
    assert not report.is_passing
    issue_codes = {issue.code for issue in report.issues}
    assert issue_codes == {"volume_negative", "bound_max_violation"}

"""Biblioteka znormalizowanych danych backtestowych dla strategii Etapu 4."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import csv
import logging

import pandas as pd
import yaml

from bot_core.observability.pandas_warnings import capture_pandas_warnings


_LOGGER = logging.getLogger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    """Wyjątek zgłaszany, gdy żądany zestaw backtestowy nie jest dostępny."""


@dataclass(slots=True)
class DatasetDescriptor:
    """Opis pojedynczego zestawu danych backtestowych."""

    name: str
    file_path: Path
    interval: str
    timezone: str
    strategies: tuple[str, ...]
    risk_profiles: tuple[str, ...]
    schema: Mapping[str, str]
    checks: Mapping[str, object]

    def iterate_rows(self) -> Iterator[MutableMapping[str, str]]:
        """Zwraca iterator po wierszach CSV jako słowniki wartości tekstowych."""

        with self.file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return iter(())
            expected_order = list(self.schema.keys())
            if reader.fieldnames != expected_order:
                raise ValueError(
                    f"Dataset {self.name} header mismatch: {reader.fieldnames} != {expected_order}"
                )
            for row in reader:
                yield row

    @property
    def reference_results(self) -> Mapping[str, object]:
        """Zwraca sekcję referencyjnych wyników, jeśli istnieje."""

        raw = self.checks.get("reference_results") if isinstance(self.checks, Mapping) else None
        if isinstance(raw, Mapping):
            return raw
        return {}


@dataclass(slots=True)
class Manifest:
    """Manifest biblioteki backtestowej."""

    version: int
    interval_units: Mapping[str, int]
    datasets: Mapping[str, DatasetDescriptor]

    def dataset(self, name: str) -> DatasetDescriptor:
        try:
            return self.datasets[name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise DatasetNotFoundError(name) from exc

    def list_datasets(self) -> Sequence[DatasetDescriptor]:
        return list(self.datasets.values())


class BacktestDatasetLibrary:
    """Loader biblioteki znormalizowanych danych backtestowych."""

    def __init__(self, manifest_path: Path):
        self._manifest_path = manifest_path
        self._manifest = self._load_manifest(manifest_path)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def _load_manifest(self, manifest_path: Path) -> Manifest:
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw_manifest = yaml.safe_load(handle)
        version = int(raw_manifest.get("version", 1))
        interval_units = {
            unit: int(seconds) for unit, seconds in raw_manifest.get("interval_units", {}).items()
        }
        datasets_section: Mapping[str, Mapping[str, object]] = raw_manifest.get("datasets", {})
        descriptors: Dict[str, DatasetDescriptor] = {}
        for dataset_name, config in datasets_section.items():
            file_name = config["file"]
            descriptor = DatasetDescriptor(
                name=dataset_name,
                file_path=manifest_path.parent / file_name,
                interval=str(config["interval"]),
                timezone=str(config.get("timezone", "UTC")),
                strategies=tuple(config.get("strategies", [])),
                risk_profiles=tuple(config.get("risk_profiles", [])),
                schema=dict(config.get("schema", {})),
                checks=dict(config.get("checks", {})),
            )
            descriptors[dataset_name] = descriptor
        return Manifest(version=version, interval_units=interval_units, datasets=descriptors)

    def interval_to_seconds(self, interval: str) -> int:
        """Konwertuje zapis interwału (np. `1m`, `5m`) na sekundy."""

        if not interval:
            raise ValueError("Interval string must be provided")
        multiplier = ""
        unit = ""
        for char in interval:
            if char.isdigit():
                multiplier += char
            else:
                unit += char
        if not multiplier or not unit:
            raise ValueError(f"Invalid interval format: {interval}")
        units = self.manifest.interval_units
        if unit not in units:
            raise ValueError(f"Unsupported interval unit '{unit}' for value {interval}")
        return int(multiplier) * units[unit]

    def list_dataset_names(self) -> Sequence[str]:
        return [descriptor.name for descriptor in self.manifest.list_datasets()]

    def describe(self, name: str) -> DatasetDescriptor:
        return self.manifest.dataset(name)

    def load_typed_rows(self, name: str) -> List[Mapping[str, object]]:
        """Ładuje wiersze i konwertuje je do typów określonych w schemacie."""

        descriptor = self.describe(name)
        typed_rows: List[Mapping[str, object]] = []
        schema = descriptor.schema
        for index, row in enumerate(descriptor.iterate_rows(), start=1):
            typed_row: Dict[str, object] = {}
            for column, expected_type in schema.items():
                if column not in row:
                    raise ValueError(
                        f"Dataset {name} row {index} missing column '{column}'"
                    )
                value = row[column]
                if value in {"", None}:
                    raise ValueError(
                        f"Dataset {name} row {index} column '{column}' contains empty value"
                    )
                typed_row[column] = self._cast_value(value, expected_type, name, column, index)
            typed_rows.append(typed_row)
        return typed_rows

    def load_dataframe(
        self,
        name: str,
        *,
        index_column: str | None = None,
        datetime_columns: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        """Ładuje dataset jako :class:`pandas.DataFrame` wraz z konwersją typów."""

        rows = self.load_typed_rows(name)
        with capture_pandas_warnings(
            _LOGGER, component="data.backtest_library.load_dataframe"
        ):
            frame = pd.DataFrame(rows)
            datetime_columns = datetime_columns or {}
            for column, unit in datetime_columns.items():
                if column in frame.columns:
                    frame[column] = pd.to_datetime(
                        frame[column], unit=unit, errors="coerce"
                    )

            if index_column and index_column in frame.columns:
                frame = frame.set_index(index_column)

        return frame

    def _cast_value(
        self, value: str, expected_type: str, dataset: str, column: str, row_index: int
    ) -> object:
        expected_type = expected_type.lower()
        if expected_type == "int":
            return int(value)
        if expected_type == "float":
            return float(value)
        if expected_type == "str":
            return value
        raise ValueError(
            f"Dataset {dataset} row {row_index} column '{column}' has unsupported type '{expected_type}'"
        )


@dataclass(slots=True)
class DataQualityIssue:
    code: str
    message: str
    row: int | None = None


@dataclass(slots=True)
class DataQualityReport:
    descriptor: DatasetDescriptor
    issues: Sequence[DataQualityIssue]
    row_count: int
    start_timestamp: int | None
    end_timestamp: int | None

    @property
    def is_passing(self) -> bool:
        return not self.issues


class DataQualityValidator:
    """Procedury walidacji spójności znormalizowanych danych backtestowych."""

    def __init__(self, library: BacktestDatasetLibrary):
        self._library = library

    def validate(self, dataset_name: str) -> DataQualityReport:
        descriptor = self._library.describe(dataset_name)
        issues: List[DataQualityIssue] = []
        try:
            rows = self._library.load_typed_rows(dataset_name)
        except ValueError as exc:
            issues.append(DataQualityIssue(code="schema_error", message=str(exc), row=None))
            return DataQualityReport(descriptor, issues, 0, None, None)

        if not rows:
            issues.append(
                DataQualityIssue(
                    code="empty_dataset",
                    message="Dataset nie zawiera żadnych wierszy",
                    row=None,
                )
            )
            return DataQualityReport(descriptor, issues, 0, None, None)

        interval_seconds = self._library.interval_to_seconds(descriptor.interval)
        previous_timestamp: int | None = None
        for index, row in enumerate(rows, start=1):
            timestamp = int(row["timestamp"])
            if previous_timestamp is not None:
                delta = timestamp - previous_timestamp
                if delta <= 0:
                    issues.append(
                        DataQualityIssue(
                            code="timestamp_order",
                            message="Znaczniki czasu muszą być ściśle rosnące",
                            row=index,
                        )
                    )
                elif delta != interval_seconds:
                    issues.append(
                        DataQualityIssue(
                            code="timestamp_interval",
                            message=(
                                f"Odstęp czasowy {delta}s niezgodny z zadeklarowanym interwałem"
                            ),
                            row=index,
                        )
                    )
            previous_timestamp = timestamp
            self._validate_checks(descriptor, row, index, issues)

        start_timestamp = int(rows[0]["timestamp"])
        end_timestamp = int(rows[-1]["timestamp"])
        return DataQualityReport(descriptor, issues, len(rows), start_timestamp, end_timestamp)

    def _validate_checks(
        self,
        descriptor: DatasetDescriptor,
        row: Mapping[str, object],
        index: int,
        issues: List[DataQualityIssue],
    ) -> None:
        checks = descriptor.checks
        price_columns: Iterable[str] = checks.get("price_columns", [])  # type: ignore[assignment]
        for column in price_columns:
            value = float(row[column])
            if value <= 0:
                issues.append(
                    DataQualityIssue(
                        code="price_non_positive",
                        message=f"Kolumna '{column}' musi być dodatnia",
                        row=index,
                    )
                )
        volume_columns: Iterable[str] = checks.get("volume_columns", [])  # type: ignore[assignment]
        for column in volume_columns:
            value = float(row[column])
            if value < 0:
                issues.append(
                    DataQualityIssue(
                        code="volume_negative",
                        message=f"Kolumna '{column}' nie może być ujemna",
                        row=index,
                    )
                )
        bounded_columns: Mapping[str, Mapping[str, float]] = checks.get("bounded_columns", {})  # type: ignore[assignment]
        for column, bounds in bounded_columns.items():
            value = float(row[column])
            lower = bounds.get("min")
            upper = bounds.get("max")
            if lower is not None and value < float(lower) - 1e-9:
                issues.append(
                    DataQualityIssue(
                        code="bound_min_violation",
                        message=f"Kolumna '{column}' poniżej minimum {lower}",
                        row=index,
                    )
                )
            if upper is not None and value > float(upper) + 1e-9:
                issues.append(
                    DataQualityIssue(
                        code="bound_max_violation",
                        message=f"Kolumna '{column}' powyżej maksimum {upper}",
                        row=index,
                    )
                )
        for bid, ask in checks.get("bid_ask_pairs", []):  # type: ignore[assignment]
            bid_value = float(row[bid])
            ask_value = float(row[ask])
            if bid_value > ask_value + 1e-9:
                issues.append(
                    DataQualityIssue(
                        code="bid_ask_inverted",
                        message=f"Para bid/ask ({bid}, {ask}) jest odwrócona",
                        row=index,
                    )
                )
        if {"high", "low", "open", "close"}.issubset(descriptor.schema.keys()):
            high = float(row["high"])
            low = float(row["low"])
            open_ = float(row["open"])
            close = float(row["close"])
            if low > high:
                issues.append(
                    DataQualityIssue(
                        code="ohlc_low_gt_high",
                        message="Wartość low nie może być większa od high",
                        row=index,
                    )
                )
            if not (low - 1e-9 <= open_ <= high + 1e-9):
                issues.append(
                    DataQualityIssue(
                        code="ohlc_open_outside_range",
                        message="Open poza zakresem high/low",
                        row=index,
                    )
                )
            if not (low - 1e-9 <= close <= high + 1e-9):
                issues.append(
                    DataQualityIssue(
                        code="ohlc_close_outside_range",
                        message="Close poza zakresem high/low",
                        row=index,
                    )
                )


__all__ = [
    "BacktestDatasetLibrary",
    "DataQualityIssue",
    "DataQualityReport",
    "DataQualityValidator",
    "DatasetDescriptor",
    "DatasetNotFoundError",
    "Manifest",
]

"""Porównanie wyników strategii przed i po migracji do warstwy async."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

Number = int | float


@dataclass(slots=True)
class MetricDeviation:
    """Reprezentuje odchyłkę metryki pomiędzy snapshotami."""

    strategy: str
    metric: str
    legacy_value: Number | None
    async_value: Number | None
    absolute_delta: float
    relative_delta: float | None


@dataclass(slots=True)
class SnapshotComparison:
    """Wynik porównania snapshotów."""

    deviations: tuple[MetricDeviation, ...]
    missing_in_async: tuple[str, ...]
    missing_in_legacy: tuple[str, ...]

    @property
    def is_within_tolerance(self) -> bool:
        return not self.deviations and not self.missing_in_async and not self.missing_in_legacy


def _load_snapshot(path: Path) -> Mapping[str, Mapping[str, Number]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    strategies = data.get("strategies")
    result: dict[str, dict[str, Number]] = {}
    if isinstance(strategies, Mapping):
        for name, metrics in strategies.items():
            if not isinstance(metrics, Mapping):
                continue
            numeric_metrics: dict[str, Number] = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_metrics[str(key)] = value
            if numeric_metrics:
                result[str(name)] = numeric_metrics
    return result


def _aggregate_directory(directory: Path) -> Mapping[str, Mapping[str, Number]]:
    if not directory.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu snapshotów: {directory}")
    aggregated: MutableMapping[str, dict[str, Number]] = {}
    json_files = sorted(p for p in directory.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"Katalog {directory} nie zawiera plików JSON ze snapshotami.")
    for file_path in json_files:
        snapshot = _load_snapshot(file_path)
        for strategy, metrics in snapshot.items():
            aggregated.setdefault(strategy, {}).update(metrics)
    return aggregated


def compare_snapshots(
    legacy_directory: Path,
    async_directory: Path,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> SnapshotComparison:
    legacy_snapshots = _aggregate_directory(legacy_directory)
    async_snapshots = _aggregate_directory(async_directory)

    deviations: list[MetricDeviation] = []
    missing_in_async: list[str] = []
    missing_in_legacy: list[str] = []

    for strategy, legacy_metrics in legacy_snapshots.items():
        async_metrics = async_snapshots.get(strategy)
        if async_metrics is None:
            missing_in_async.append(strategy)
            continue
        for metric, legacy_value in legacy_metrics.items():
            async_value = async_metrics.get(metric)
            if async_value is None:
                missing_in_async.append(f"{strategy}:{metric}")
                continue
            delta = float(async_value) - float(legacy_value)
            abs_delta = abs(delta)
            rel_base = max(abs(float(legacy_value)), 1e-12)
            rel_delta = abs_delta / rel_base if rel_base > 0 else None
            allowed = max(absolute_tolerance, relative_tolerance * rel_base)
            if abs_delta > allowed:
                deviations.append(
                    MetricDeviation(
                        strategy=strategy,
                        metric=metric,
                        legacy_value=float(legacy_value),
                        async_value=float(async_value),
                        absolute_delta=abs_delta,
                        relative_delta=rel_delta,
                    )
                )

    for strategy in async_snapshots.keys() - legacy_snapshots.keys():
        missing_in_legacy.append(strategy)

    deviations.sort(key=lambda item: (item.strategy, item.metric))
    missing_in_async = sorted(set(missing_in_async))
    missing_in_legacy = sorted(set(missing_in_legacy))

    return SnapshotComparison(
        deviations=tuple(deviations),
        missing_in_async=tuple(missing_in_async),
        missing_in_legacy=tuple(missing_in_legacy),
    )


def _format_deviation(deviation: MetricDeviation) -> str:
    rel = (
        f" ({deviation.relative_delta * 100:.2f}% różnicy)"
        if deviation.relative_delta is not None
        else ""
    )
    return (
        f"- Strategia '{deviation.strategy}', metryka '{deviation.metric}': "
        f"legacy={deviation.legacy_value:.6f}, async={deviation.async_value:.6f}, "
        f"Δ={deviation.absolute_delta:.6f}{rel}"
    )


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-dir",
        default="data/snapshots/legacy",
        help="Katalog ze snapshotami przed migracją.",
    )
    parser.add_argument(
        "--async-dir",
        default="data/snapshots/async",
        help="Katalog ze snapshotami po migracji.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.05,
        help="Maksymalna względna różnica KPI (np. 0.05 = 5%).",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=1e-4,
        help="Minimalna tolerancja bezwzględna dla bardzo małych wartości.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    comparison = compare_snapshots(
        Path(args.legacy_dir),
        Path(args.async_dir),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )

    if comparison.is_within_tolerance:
        print("Wszystkie metryki mieszczą się w zadanych tolerancjach.")
        return 0

    if comparison.missing_in_async:
        print("Brakujące elementy po migracji:")
        for item in comparison.missing_in_async:
            print(f"  - {item}")
    if comparison.missing_in_legacy:
        print("Nowe elementy po migracji (brak w legacy):")
        for item in comparison.missing_in_legacy:
            print(f"  - {item}")
    if comparison.deviations:
        print("Metryki przekraczające tolerancję:")
        for deviation in comparison.deviations:
            print(_format_deviation(deviation))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

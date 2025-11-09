"""Narzędzie do porównywania championów między instalacjami."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from bot_core.reporting import model_quality


@dataclass(slots=True)
class ChampionSnapshot:
    model: str
    exists: bool
    metadata: Mapping[str, Any]
    status: str | None
    metrics: Mapping[str, Any]
    parameters: Mapping[str, Any]
    numeric_metrics: Mapping[str, float]


def _flatten_numeric(source: Mapping[str, Any], *, prefix: str | None = None) -> Mapping[str, float]:
    result: MutableMapping[str, float] = {}
    for key, value in source.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            nested = _flatten_numeric(value, prefix=name)
            result.update(nested)
            continue
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        result[name] = numeric
    return dict(sorted(result.items()))


def _normalize_mapping(payload: object) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    normalized: MutableMapping[str, Any] = {}
    for key in sorted(payload.keys(), key=str):
        normalized[str(key)] = payload[key]
    return normalized


def _build_snapshot(model: str, overview: Mapping[str, Any] | None) -> ChampionSnapshot:
    if overview is None:
        return ChampionSnapshot(
            model=model,
            exists=False,
            metadata={},
            status=None,
            metrics={},
            parameters={},
            numeric_metrics={},
        )

    champion = overview.get("champion")
    metadata_raw = overview.get("champion_metadata")

    metrics = _normalize_mapping(
        champion.get("metrics") if isinstance(champion, Mapping) else {}
    )
    parameters = _normalize_mapping(
        champion.get("parameters") if isinstance(champion, Mapping) else {}
    )
    status = None
    if isinstance(champion, Mapping):
        raw_status = champion.get("status")
        status = str(raw_status) if raw_status is not None else None

    metadata = _normalize_mapping(metadata_raw)
    numeric_metrics = _flatten_numeric(metrics)

    return ChampionSnapshot(
        model=model,
        exists=True,
        metadata=metadata,
        status=status,
        metrics=metrics,
        parameters=parameters,
        numeric_metrics=numeric_metrics,
    )


def _load_models(lhs_root: Path, rhs_root: Path, requested: Sequence[str] | None) -> Sequence[str]:
    if requested:
        return tuple(dict.fromkeys(requested))
    lhs_models = model_quality.list_tracked_models(base_dir=lhs_root)
    rhs_models = model_quality.list_tracked_models(base_dir=rhs_root)
    if not lhs_models and not rhs_models:
        return ()
    all_models = sorted(set(lhs_models) | set(rhs_models))
    return tuple(all_models)


def _load_snapshot(root: Path, model: str) -> ChampionSnapshot:
    overview = model_quality.load_champion_overview(model, base_dir=root)
    return _build_snapshot(model, overview)


def _diff_parameters(lhs: ChampionSnapshot, rhs: ChampionSnapshot) -> Mapping[str, Mapping[str, Any]]:
    keys = sorted(set(lhs.parameters.keys()) | set(rhs.parameters.keys()))
    diff: MutableMapping[str, Mapping[str, Any]] = {}
    for key in keys:
        lhs_value = lhs.parameters.get(key)
        rhs_value = rhs.parameters.get(key)
        if lhs_value == rhs_value:
            continue
        diff[key] = {"from": lhs_value, "to": rhs_value}
    return diff


def _diff_metrics(lhs: ChampionSnapshot, rhs: ChampionSnapshot) -> Mapping[str, Mapping[str, Any]]:
    keys = sorted(set(lhs.numeric_metrics.keys()) | set(rhs.numeric_metrics.keys()))
    diff: MutableMapping[str, Mapping[str, Any]] = {}
    for key in keys:
        lhs_value = lhs.numeric_metrics.get(key)
        rhs_value = rhs.numeric_metrics.get(key)
        if lhs_value is None or rhs_value is None:
            diff[key] = {"from": lhs_value, "to": rhs_value}
            continue
        if abs(rhs_value - lhs_value) < 1e-12:
            continue
        diff[key] = {
            "from": lhs_value,
            "to": rhs_value,
            "delta": rhs_value - lhs_value,
        }
    return diff


def _build_comparison(lhs: ChampionSnapshot, rhs: ChampionSnapshot) -> Mapping[str, Any]:
    status_change = None
    if lhs.status != rhs.status:
        status_change = {"from": lhs.status, "to": rhs.status}

    metadata_changed = lhs.metadata != rhs.metadata
    parameters_diff = _diff_parameters(lhs, rhs)
    metrics_diff = _diff_metrics(lhs, rhs)

    return {
        "model": lhs.model,
        "lhs": {
            "exists": lhs.exists,
            "metadata": lhs.metadata,
            "status": lhs.status,
            "metrics": lhs.metrics,
            "parameters": lhs.parameters,
        },
        "rhs": {
            "exists": rhs.exists,
            "metadata": rhs.metadata,
            "status": rhs.status,
            "metrics": rhs.metrics,
            "parameters": rhs.parameters,
        },
        "differences": {
            "metadata_changed": metadata_changed,
            "status_change": status_change,
            "metrics_delta": metrics_diff,
            "parameter_changes": parameters_diff,
        },
    }


def generate_diff(
    *,
    lhs_root: Path | str,
    rhs_root: Path | str,
    models: Sequence[str] | None = None,
    output_dir: Path | str,
    tag: str | None = None,
) -> Path:
    lhs_root = Path(lhs_root).expanduser()
    rhs_root = Path(rhs_root).expanduser()
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = _load_models(lhs_root, rhs_root, models)

    comparisons = []
    for model in model_names:
        lhs_snapshot = _load_snapshot(lhs_root, model)
        rhs_snapshot = _load_snapshot(rhs_root, model)
        comparisons.append(_build_comparison(lhs_snapshot, rhs_snapshot))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lhs_root": str(lhs_root),
        "rhs_root": str(rhs_root),
        "models": model_names,
        "comparisons": comparisons,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = f"_{tag}" if tag else ""
    output_path = output_root / f"champion_diff_{timestamp}{suffix}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Porównuje championów pomiędzy dwoma katalogami jakości modeli."
    )
    parser.add_argument(
        "--lhs",
        type=Path,
        default=Path("deploy/packaging/samples/var/models/quality"),
        help="Katalog bazowy championów referencyjnych (domyślnie próbki dystrybucyjne).",
    )
    parser.add_argument(
        "--rhs",
        type=Path,
        default=Path("var/models/quality"),
        help="Katalog bazowy championów środowiska docelowego.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Nazwa modelu do porównania (można powtarzać wielokrotnie).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("audit/champion_diffs"),
        help="Katalog zapisu raportów audytowych.",
    )
    parser.add_argument(
        "--tag",
        help="Dodatkowy sufiks pliku wynikowego (np. nazwa środowiska).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_path = generate_diff(
        lhs_root=args.lhs,
        rhs_root=args.rhs,
        models=args.models,
        output_dir=args.output_dir,
        tag=args.tag,
    )
    print(f"Raport zapisano w: {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())

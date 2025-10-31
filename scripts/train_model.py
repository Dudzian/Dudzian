"""CLI do trenowania modeli ML z obsługą fallbacku backendów."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector
from core.ml.training_pipeline import TrainingPipeline

LOGGER = logging.getLogger("train_model")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trenuje model ML korzystając z preferowanych backendów i obsługą fallbacku",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Plik JSON z kluczami 'features' (lista słowników) oraz 'targets' (lista wartości)",
    )
    parser.add_argument(
        "--preferred-backend",
        action="append",
        dest="preferred_backends",
        default=[],
        help="Preferowane backendy w kolejności priorytetu (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help="Ścieżka do pliku konfiguracyjnego backendów ML (backends.yml)",
    )
    parser.add_argument(
        "--fallback-log-dir",
        type=Path,
        default=Path("logs/ml/fallback"),
        help="Katalog zapisu logów fallbacku backendów",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (domyślnie INFO)",
    )
    return parser


def _load_dataset(path: Path | None) -> FeatureDataset:
    if path is None:
        return FeatureDataset(vectors=tuple(_synthetic_vectors()), metadata={"source": "synthetic"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    targets = payload.get("targets")
    if not isinstance(features, Sequence) or not isinstance(targets, Sequence):
        raise SystemExit("Niepoprawny format pliku datasetu – oczekiwano list features/targets")
    if len(features) != len(targets) or not features:
        raise SystemExit("Dataset musi zawierać co najmniej jedną próbkę z targetem")
    vectors: list[FeatureVector] = []
    symbol = str(payload.get("symbol", "SYNTH"))
    timestamp = float(payload.get("start_timestamp", 1_700_000_000.0))
    for idx, (feature_map, target) in enumerate(zip(features, targets)):
        if not isinstance(feature_map, Mapping):
            raise SystemExit("Każda próbka features musi być słownikiem")
        vectors.append(
            FeatureVector(
                timestamp=timestamp + idx * 60.0,
                symbol=symbol,
                features={str(k): float(v) for k, v in feature_map.items()},
                target_bps=float(target),
            )
        )
    metadata = {
        "source": str(path),
        "row_count": len(vectors),
    }
    return FeatureDataset(vectors=tuple(vectors), metadata=metadata)


def _synthetic_vectors() -> Iterable[FeatureVector]:
    base_timestamp = 1_700_000_000.0
    for idx in range(5):
        yield FeatureVector(
            timestamp=base_timestamp + idx * 60.0,
            symbol="SYNTH",
            features={
                "momentum": float(idx) * 0.2,
                "volatility": 0.3 + float(idx) * 0.1,
            },
            target_bps=0.015 * (idx - 1),
        )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    dataset = _load_dataset(args.dataset)
    preferred_backends = tuple(args.preferred_backends or ("lightgbm", "reference"))

    pipeline = TrainingPipeline(
        preferred_backends=preferred_backends,
        config_path=args.config_path,
        fallback_log_dir=args.fallback_log_dir,
    )

    result = pipeline.train(dataset)
    LOGGER.info("Wybrany backend: %s", result.backend)

    if result.fallback_chain:
        LOGGER.warning(
            "Aktywowano fallback backendów: %s",
            ", ".join(entry["backend"] for entry in result.fallback_chain),
        )
        for entry in result.fallback_chain:
            hint = entry.get("install_hint")
            if hint:
                LOGGER.warning(
                    "Backend %s niedostępny: %s. Instalacja: %s",
                    entry["backend"],
                    entry["message"],
                    hint,
                )
            else:
                LOGGER.warning(
                    "Backend %s niedostępny: %s",
                    entry["backend"],
                    entry["message"],
                )
        if result.log_path:
            LOGGER.info("Log fallbacku zapisany w %s", result.log_path)
    else:
        LOGGER.info("Trening zakończony bez konieczności fallbacku")

    print(
        json.dumps(
            {
                "backend": result.backend,
                "fallback_log": str(result.log_path) if result.log_path else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - wywołanie z CLI
    raise SystemExit(main())


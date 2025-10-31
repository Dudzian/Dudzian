"""Referencyjny backend ML działający w czystym Pythonie.

Moduł udostępnia klasę :class:`ReferenceRegressor`, która realizuje
prosty regresor liniowy przy użyciu równań normalnych rozwiązanych
metodą eliminacji Gaussa. Implementacja nie wymaga zewnętrznych
zależności – korzysta wyłącznie z biblioteki standardowej – dzięki
czemu stanowi bezpieczny fallback dla środowisk, w których pakiety typu
LightGBM lub XGBoost nie są dostępne.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_Number = float


class ModelNotTrainedError(RuntimeError):
    """Wyjątek zgłaszany, gdy model używany jest przed treningiem."""


def _resolve_feature_names(samples: Sequence[Mapping[str, float]]) -> tuple[str, ...]:
    names: set[str] = set()
    for sample in samples:
        for key in sample.keys():
            names.add(str(key))
    if not names:
        raise ValueError("Brak cech wejściowych – nie można przeprowadzić treningu")
    return tuple(sorted(names))


def _augment_row(values: Sequence[float]) -> list[float]:
    return [1.0, *map(float, values)]


def _build_matrix(
    samples: Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for sample in samples:
        row = [float(sample.get(name, 0.0)) for name in feature_names]
        matrix.append(row)
    return matrix


def _build_normal_equations(
    matrix: Sequence[Sequence[float]],
    targets: Sequence[float],
) -> tuple[list[list[float]], list[float]]:
    feature_count = len(matrix[0]) if matrix else 0
    size = feature_count + 1  # +1 na wyraz wolny (intercept)
    xtx = [[0.0 for _ in range(size)] for _ in range(size)]
    xty = [0.0 for _ in range(size)]

    for row, target in zip(matrix, targets):
        augmented = _augment_row(row)
        for i, value_i in enumerate(augmented):
            xty[i] += value_i * target
            for j, value_j in enumerate(augmented):
                xtx[i][j] += value_i * value_j
    return xtx, xty


def _gaussian_elimination(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Rozwiązuje układ liniowy przy pomocy eliminacji Gaussa z pivotowaniem."""

    size = len(vector)
    augmented = [row[:size] + [vector[idx]] for idx, row in enumerate(matrix[:size])]

    for col in range(size):
        pivot_row = max(range(col, size), key=lambda row_idx: abs(augmented[row_idx][col]))
        pivot_value = augmented[pivot_row][col]
        if abs(pivot_value) < 1e-12:
            raise ZeroDivisionError("Macierz układu jest osobliwa")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot_value = augmented[col][col]
        for idx in range(col, size + 1):
            augmented[col][idx] /= pivot_value

        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) < 1e-12:
                continue
            for idx in range(col, size + 1):
                augmented[row][idx] -= factor * augmented[col][idx]

    return [augmented[row][size] for row in range(size)]


@dataclass(slots=True)
class ReferenceRegressor:
    """Niewielki regresor liniowy przeznaczony do pracy offline."""

    feature_names: tuple[str, ...] = ()
    coefficients: tuple[float, ...] = ()
    intercept: float = 0.0
    trained_samples: int = 0

    def fit(
        self,
        samples: Sequence[Mapping[str, float]],
        targets: Sequence[float],
    ) -> None:
        if not samples:
            raise ValueError("Przekazano pusty zbiór danych – trening niemożliwy")
        if len(samples) != len(targets):
            raise ValueError("Liczba próbek i wartości docelowych musi być identyczna")

        feature_names = _resolve_feature_names(samples)
        matrix = _build_matrix(samples, feature_names)
        y = [float(value) for value in targets]

        try:
            xtx, xty = _build_normal_equations(matrix, y)
            solution = _gaussian_elimination(xtx, xty)
            intercept = float(solution[0])
            coefficients = tuple(float(value) for value in solution[1:])
        except ZeroDivisionError:
            intercept = sum(y) / len(y)
            coefficients = tuple(0.0 for _ in feature_names)

        self.feature_names = feature_names
        self.coefficients = coefficients
        self.intercept = intercept
        self.trained_samples = len(samples)

    # Typ Sequence[Mapping[str, float]] jest używany w pipeline Decision Engine,
    # dlatego zachowujemy identyczny podpis metod predykcji jak w modelach
    # zewnętrznych.
    def predict(self, sample: Mapping[str, float]) -> float:
        if not self.trained_samples:
            raise ModelNotTrainedError("Model nie został wytrenowany")
        return float(
            self.intercept
            + sum(
                coef * float(sample.get(name, 0.0))
                for name, coef in zip(self.feature_names, self.coefficients)
            )
        )

    def batch_predict(self, samples: Sequence[Mapping[str, float]]) -> list[float]:
        return [self.predict(sample) for sample in samples]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "intercept": float(self.intercept),
            "trained_samples": int(self.trained_samples),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        feature_names = data.get("feature_names")
        coefficients = data.get("coefficients")
        intercept = data.get("intercept")
        trained_samples = data.get("trained_samples", 0)

        if not isinstance(feature_names, Iterable) or not isinstance(coefficients, Iterable):
            raise ValueError("Uszkodzony plik modelu – brak współczynników")
        feature_names_tuple = tuple(str(name) for name in feature_names)
        coefficients_tuple = tuple(float(value) for value in coefficients)
        if len(feature_names_tuple) != len(coefficients_tuple):
            raise ValueError("Niespójna liczba cech i współczynników w pliku modelu")

        self.feature_names = feature_names_tuple
        self.coefficients = coefficients_tuple
        self.intercept = float(intercept or 0.0)
        self.trained_samples = int(trained_samples)


def build_reference_regressor() -> ReferenceRegressor:
    """Tworzy nową instancję referencyjnego regresora."""

    return ReferenceRegressor()


__all__ = ["ReferenceRegressor", "ModelNotTrainedError", "build_reference_regressor"]

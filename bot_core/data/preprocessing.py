"""Utility helpers for preparing sequential datasets for AI models."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["windowize"]


def windowize(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    seq_len: int,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a feature dataframe into overlapping windows.

    Parameters
    ----------
    df:
        Source dataframe containing all required columns.
    feature_cols:
        Columns that should be transformed into the feature tensor.
    seq_len:
        Number of time steps per sample.
    target_col:
        Column used to compute future returns.  The function generates
        percentage change labels between consecutive observations.
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if len(df) <= seq_len:
        raise ValueError("Not enough rows to generate windows")

    values = df.loc[:, feature_cols].to_numpy(dtype=float, copy=False)
    target = df.loc[:, target_col].to_numpy(dtype=float, copy=False)

    features: list[np.ndarray] = []
    labels: list[float] = []
    for idx in range(seq_len, len(df)):
        window = values[idx - seq_len : idx]
        features.append(window)
        prev = target[idx - 1] if target[idx - 1] != 0 else 1e-12
        labels.append((target[idx] / prev) - 1.0)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)

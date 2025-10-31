"""Pakiet na wygenerowane stuby gRPC (nie commitujemy wygenerowanych plików)."""
from __future__ import annotations

import importlib
import sys

__all__ = [
    "trading_pb2",
    "trading_pb2_grpc",
]


def _expose_module(module_name: str) -> None:
    full_name = f"{__name__}.{module_name}"
    try:
        module = importlib.import_module(full_name)
    except ModuleNotFoundError:
        return
    if module_name not in sys.modules:
        sys.modules[module_name] = module


for _module in __all__:
    _expose_module(_module)

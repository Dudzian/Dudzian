"""Static ABI entry point for the Ubuntu Noble Observer preflight builder."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, cast

_BASE_LIBPYTHON = Path("/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0")


def diagnose(request: dict[str, Any], expectations: dict[str, Any]) -> dict[str, Any]:
    abi = runpy.run_path(str(Path(__file__).with_name("stage9_v5d4_abi.py")))
    return cast(
        dict[str, Any],
        abi["diagnose"](request, expectations, observer_base_libpython=_BASE_LIBPYTHON),
    )

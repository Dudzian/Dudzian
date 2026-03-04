from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any


def run_cli_utf8(
    args: Sequence[str | os.PathLike[str]],
    *,
    env: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run subprocess with deterministic UTF-8 decoding for CLI tests."""

    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    merged_env["PYTHONUTF8"] = "1"
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")

    run_kwargs: dict[str, Any] = dict(kwargs)
    run_kwargs.setdefault("text", True)
    run_kwargs.setdefault("encoding", "utf-8")
    run_kwargs.setdefault("errors", "strict")
    run_kwargs["env"] = merged_env

    return subprocess.run(list(args), **run_kwargs)

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_betterleaks() -> str | None:
    configured = os.environ.get("BETTERLEAKS_BIN")
    candidates = [configured, shutil.which("betterleaks")]
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        candidates.append(
            str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "betterleaks" / "betterleaks.exe")
        )
    return next(
        (candidate for candidate in candidates if candidate and Path(candidate).is_file()), None
    )


def main(argv: list[str] | None = None) -> int:
    executable = _find_betterleaks()
    if executable is None:
        print(
            "Betterleaks is required for the staged secret scan. "
            "Install stable v1.9.x or set BETTERLEAKS_BIN.",
            file=sys.stderr,
        )
        return 2

    files = [str(Path(value)) for value in (argv or sys.argv[1:]) if Path(value).is_file()]
    if not files:
        return 0
    baseline = Path(".betterleaks-baseline.json")
    baseline_args = ["--baseline-path", str(baseline)] if baseline.is_file() else []
    completed = subprocess.run(
        [
            executable,
            "dir",
            *baseline_args,
            "--redact=100",
            "--no-banner",
            "--no-color",
            "--exit-code=1",
            *files,
        ],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

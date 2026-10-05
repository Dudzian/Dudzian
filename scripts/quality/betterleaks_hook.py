from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


# Keep comfortably below Windows' 32,767 character command-line limit and the
# usual POSIX ARG_MAX.  Splitting the targets would change Betterleaks' scan
# scope/fingerprints, so an oversized scope is rejected rather than weakened.
_MAX_COMMAND_BYTES = 30_000 if os.name == "nt" else 128_000


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


def _staged_paths(repository: Path) -> list[str] | None:
    completed = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z", "--"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        print("Unable to read staged paths; refusing to skip the secret scan.", file=sys.stderr)
        return None
    return [os.fsdecode(item) for item in completed.stdout.split(b"\0") if item]


def _scan_paths(repository: Path, supplied: list[str] | None) -> list[str] | None:
    paths = supplied if supplied else _staged_paths(repository)
    if paths is None:
        return None

    targets: list[str] = []
    for item in paths:
        candidate = Path(item)
        absolute = candidate if candidate.is_absolute() else repository / candidate
        try:
            relative = absolute.absolute().relative_to(repository.resolve())
        except ValueError:
            print(f"Secret-scan path is outside the repository: {item!r}", file=sys.stderr)
            return None
        try:
            mode = absolute.lstat().st_mode
        except OSError:
            continue
        if stat.S_ISREG(mode):
            targets.append(relative.as_posix())
    return targets


def main(argv: list[str] | None = None) -> int:
    repository = Path.cwd()
    targets = _scan_paths(repository, argv)
    if targets is None:
        return 2
    if not targets:
        return 0

    executable = _find_betterleaks()
    if executable is None:
        print(
            "Betterleaks is required for the staged secret scan. "
            "Install stable v1.9.x or set BETTERLEAKS_BIN.",
            file=sys.stderr,
        )
        return 2

    baseline = repository / ".betterleaks-baseline.json"
    baseline_args = ["--baseline-path", str(baseline)] if baseline.is_file() else []
    command = [
        executable,
        "dir",
        *baseline_args,
        "--redact=100",
        "--no-banner",
        "--no-color",
        "--exit-code=1",
        "--",
        *targets,
    ]
    command_bytes = sum(len(os.fsencode(argument)) + 1 for argument in command)
    if command_bytes > _MAX_COMMAND_BYTES:
        print(
            "Staged secret-scan scope exceeds the safe command-line limit; "
            "refusing to split or skip it.",
            file=sys.stderr,
        )
        return 2

    completed = subprocess.run(command, cwd=repository, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

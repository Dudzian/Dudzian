"""Private Stage-5 qualification child; it spawns only after the assignment gate."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

TIMEOUT = 20.0
# Win32 DETACHED_PROCESS, defined locally so static imports remain portable.
_DETACHED_PROCESS = 0x00000008


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--grandchild":
        time.sleep(3600)
        return 0
    gate = Path(argv[1])
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline and not gate.is_file():
        time.sleep(0.05)
    if not gate.is_file():
        return 2
    child = subprocess.Popen(
        [sys.executable, __file__, "--grandchild"],
        creationflags=_DETACHED_PROCESS,
    )
    temporary = gate.with_suffix(".tmp")
    temporary.write_text(str(child.pid), encoding="ascii")
    os.replace(temporary, gate)
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

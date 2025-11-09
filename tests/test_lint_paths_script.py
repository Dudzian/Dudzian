from __future__ import annotations

import io
from contextlib import redirect_stdout

from scripts import lint_paths


def test_lint_paths_script_runs_clean() -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = lint_paths.main()
    output = buffer.getvalue()
    assert result == 0, output
    assert "repository layout lint passed" in output.lower()

from __future__ import annotations

from pathlib import Path

import pytest

from tests.scripts import test_path_patch_safety as guard


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        (
            "import os\ndef test_x(monkeypatch):\n    monkeypatch.setattr(os, 'name', 'posix')\n",
            True,
        ),
        (
            "import os\n"
            "def test_x(monkeypatch):\n"
            "    monkeypatch.setattr(os, name='name', value='posix')\n",
            True,
        ),
        (
            "from pathlib import Path as P\n"
            "def test_x(monkeypatch):\n"
            "    monkeypatch.setattr(P, 'cwd', lambda: None)\n",
            True,
        ),
        (
            "import pathlib\n"
            "def test_x(monkeypatch):\n"
            "    monkeypatch.setattr(pathlib.Path, 'cwd', lambda: None)\n",
            True,
        ),
        (
            "import pathlib as pl\n"
            "def test_x(monkeypatch):\n"
            "    monkeypatch.setattr(pl.Path, 'cwd', lambda: None)\n",
            True,
        ),
        (
            "from pathlib import PosixPath as PP\n"
            "def test_x(monkeypatch):\n"
            "    monkeypatch.setattr(PP, 'cwd', lambda: None)\n",
            True,
        ),
        (
            "from pathlib import Path as P\n"
            "def test_x(monkeypatch):\n"
            "    monkeypatch.setattr(target=P, name='cwd', value=lambda: None)\n",
            True,
        ),
        (
            "from pathlib import Path as P\n"
            "def test_x(monkeypatch):\n"
            "    monkeypatch.setattr(P, name='cwd', value=lambda: None)\n",
            True,
        ),
        (
            "from unittest.mock import patch as p\ndef test_x():\n    p('os.name', 'posix')\n",
            True,
        ),
        (
            "def test_x(monkeypatch):\n    monkeypatch.setattr('os.name', 'posix')\n",
            True,
        ),
        (
            "def test_x(monkeypatch):\n    monkeypatch.setattr('pathlib.Path.cwd', lambda: None)\n",
            True,
        ),
        (
            "from unittest.mock import patch as p\n"
            "import os\n"
            "def test_x():\n"
            "    p.object(target=os, attribute='name', new='posix')\n",
            True,
        ),
        (
            "from pathlib import Path as P\n"
            "from unittest.mock import patch\n"
            "def test_x():\n"
            "    patch.object(P, 'cwd', lambda: None)\n",
            True,
        ),
        (
            "from unittest.mock import patch\n"
            "import pathlib\n"
            "def test_x():\n"
            "    patch.object(pathlib.Path, 'cwd', lambda: None)\n",
            True,
        ),
        (
            "from pathlib import PosixPath as PP\n"
            "from unittest.mock import patch\n"
            "def test_x():\n"
            "    patch.object(target=PP, attribute='cwd', new=lambda: None)\n",
            True,
        ),
        (
            "import os\n"
            "from unittest.mock import patch\n"
            "def test_x():\n"
            "    patch.object(os, attribute='name', new='posix')\n",
            True,
        ),
        (
            "from pathlib import Path as P\n"
            "from unittest.mock import patch\n"
            "def test_x():\n"
            "    patch.object(P, attribute='cwd', new=lambda: None)\n",
            True,
        ),
        (
            "from unittest.mock import patch\n"
            "def test_x():\n"
            "    patch(target='os.name', new='posix')\n",
            True,
        ),
        (
            "import some_module as installer\n"
            "def test_x(monkeypatch):\n"
            "    monkeypatch.setattr(target=installer.os, name='name', value='posix')\n",
            True,
        ),
        (
            "import some_module as installer\n"
            "from unittest.mock import patch\n"
            "def test_x():\n"
            "    patch.object(target=installer.os, attribute='name', new='posix')\n",
            True,
        ),
        (
            "def test_x(monkeypatch):\n"
            "    monkeypatch.setattr(target=42, name='value', value=100)\n",
            False,
        ),
    ],
)
def test_find_forbidden_monkeypatches_detects_expected_cases(
    tmp_path: Path, snippet: str, expected: bool
) -> None:
    test_file = tmp_path / "sample_test.py"
    test_file.write_text(snippet, encoding="utf-8")

    offenders = guard._find_forbidden_monkeypatches(test_file)
    assert bool(offenders) is expected

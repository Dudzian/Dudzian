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


@pytest.mark.parametrize("method", ["stat", "iterdir", "open"])
@pytest.mark.parametrize(
    "snippet",
    [
        "from pathlib import Path as P\n"
        "def test_x(monkeypatch):\n"
        "    monkeypatch.setattr(P, '{method}', lambda *args: None)\n",
        "import pathlib as pl\n"
        "def test_x(monkeypatch):\n"
        "    monkeypatch.setattr(target=pl.Path, name='{method}', value=lambda *args: None)\n",
        "def test_x(monkeypatch):\n"
        "    monkeypatch.setattr('pathlib.Path.{method}', lambda *args: None)\n",
        "from pathlib import Path as P\n"
        "def test_x(mocker):\n"
        "    mocker.patch.object(P, '{method}', lambda *args: None)\n",
        "from unittest.mock import patch as p\n"
        "def test_x():\n"
        "    p('pathlib.Path.{method}', lambda *args: None)\n",
    ],
)
def test_process_observation_path_operations_are_detected(tmp_path, method, snippet):
    test_file = tmp_path / "process_observation_test.py"
    test_file.write_text(snippet.format(method=method), encoding="utf-8")
    assert guard._find_forbidden_monkeypatches(
        test_file, forbidden_os_attrs=frozenset({"name", "stat", "readlink"})
    )


@pytest.mark.parametrize("method", ["stat", "readlink"])
@pytest.mark.parametrize(
    "snippet",
    [
        "import os as operating\n"
        "def test_x(monkeypatch):\n"
        "    monkeypatch.setattr(operating, '{method}', lambda *args: None)\n",
        "from deployment import stage9_stage41b_coexistence as q\n"
        "def test_x(monkeypatch):\n"
        "    monkeypatch.setattr(q.os, '{method}', lambda *args: None)\n",
        "def test_x(monkeypatch):\n    monkeypatch.setattr('os.{method}', lambda *args: None)\n",
        "from unittest.mock import patch as p\n"
        "import os\n"
        "def test_x():\n"
        "    p.object(target=os, attribute='{method}', new=lambda *args: None)\n",
        "def test_x(mocker):\n    mocker.patch('os.{method}', lambda *args: None)\n",
    ],
)
def test_process_observation_os_io_policy_remains_explicit(tmp_path, method, snippet):
    test_file = tmp_path / "process_observation_test.py"
    test_file.write_text(snippet.format(method=method), encoding="utf-8")
    # The stronger I/O policy applies only when the reviewed scope opts in.
    assert not guard._find_forbidden_monkeypatches(test_file)
    assert guard._find_forbidden_monkeypatches(
        test_file, forbidden_os_attrs=frozenset({"name", "stat", "readlink"})
    )


@pytest.mark.parametrize(
    "snippet",
    [
        "from deployment import stage9_stage41b_coexistence as q\n"
        "def test_x(monkeypatch):\n"
        "    monkeypatch.setattr(q, '_proc_readlink', lambda *args: None)\n",
        "def test_x(monkeypatch, local_path):\n"
        "    monkeypatch.setattr(local_path, 'stat', lambda *args: None)\n",
    ],
)
def test_process_observation_local_dependencies_remain_allowed(tmp_path, snippet):
    test_file = tmp_path / "process_observation_test.py"
    test_file.write_text(snippet, encoding="utf-8")
    assert not guard._find_forbidden_monkeypatches(
        test_file, forbidden_os_attrs=frozenset({"name", "stat", "readlink"})
    )

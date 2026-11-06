"""Rzeczywisty bash z YAML i walidator; wyłącznie wykonanie pytest jest modelem.

Raporty JUnit są syntetyczne, z dokładnym statycznym wykazem 83 przypadków.
Próby nie uruchamiają transportu, PostgreSQL, usług ani kwalifikacji dwóch UID.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from scripts.quality import check_s9_fv_01_junit as gate

_ROOT = Path(__file__).resolve().parents[2]
_VALIDATOR = "scripts/quality/check_s9_fv_01_junit.py"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Integracja bash dotyczy linuksowego joba jakości"
)

_PYTHON_SHIM = """\
import json
import os
import shutil
import sys
from pathlib import Path

arguments = sys.argv[1:]
if arguments[:2] == ["-m", "pytest"]:
    Path(os.environ["S9_P2_PYTEST_MARKER"]).write_text(json.dumps(arguments))
    report = next(item.split("=", 1)[1] for item in arguments if item.startswith("--junitxml="))
    if os.environ["S9_P2_WRITE_REPORT"] == "1":
        shutil.copyfile(os.environ["S9_P2_TEMPLATE"], report)
    raise SystemExit(int(os.environ["S9_P2_PYTEST_EXIT"]))
if arguments and arguments[0] == "scripts/quality/check_s9_fv_01_junit.py":
    Path(os.environ["S9_P2_VALIDATOR_MARKER"]).write_text(json.dumps(arguments))
    os.execv(sys.executable, [sys.executable, *arguments])
raise SystemExit("Nieoczekiwane wywołanie python w modelu kroku CI")
"""


@pytest.fixture
def migration_step():
    workflow = yaml.safe_load((_ROOT / ".github/workflows/quality-security.yml").read_text())
    steps = workflow["jobs"]["quality-ratchet"]["steps"]
    matches = [step for step in steps if step.get("name", "").startswith("S9-FV-01 ")]
    assert len(matches) == 1
    step = matches[0]
    assert step["shell"] == "bash"
    assert not step.get("continue-on-error", False)
    return step


def _template(path: Path, *, skip_required: bool) -> None:
    required = sorted(gate.load_required_cases())
    assert len(required) == 83
    skipped_identity = next(identity for identity in required if identity[0].endswith("_unix"))
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        {"tests": "83", "skipped": str(int(skip_required)), "failures": "0", "errors": "0"},
    )
    for classname, name in required:
        case = ET.SubElement(suite, "testcase", {"classname": classname, "name": name})
        if skip_required and (classname, name) == skipped_identity:
            ET.SubElement(case, "skipped", {"message": "Model: SO_PEERPIDFD niedostępny"})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


@pytest.mark.parametrize(
    "write_report,skip_required,pytest_exit,seed_old_report,expected_exit,expected_state",
    [
        (True, False, 0, False, 0, "PASS"),
        (True, True, 0, False, 1, "BLOCKED"),
        (True, False, 2, False, 1, "BLOCKED"),
        (False, False, 5, True, 1, "BLOCKED"),
    ],
    ids=["komplet-83", "obowiazkowy-skip", "blad-pytest", "stary-raport-zero-testow"],
)
def test_literal_workflow_runs_real_validator_and_preserves_pytest_outcome(
    tmp_path,
    migration_step,
    write_report,
    skip_required,
    pytest_exit,
    seed_old_report,
    expected_exit,
    expected_state,
):
    bash = shutil.which("bash")
    assert bash is not None, "Linuksowy job wymaga rzeczywistego bash"
    model = tmp_path / "python_model.py"
    model.write_text(_PYTHON_SHIM)
    shim_directory = tmp_path / "bin"
    shim_directory.mkdir()
    shim = shim_directory / "python"
    shim.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(model))} "$@"\n'
    )
    shim.chmod(0o700)
    template = tmp_path / "template.xml"
    _template(template, skip_required=skip_required)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    report = runner_temp / "s9-fv-01-junit.xml"
    if seed_old_report:
        shutil.copyfile(template, report)
    pytest_marker = tmp_path / "pytest-called.json"
    validator_marker = tmp_path / "validator-called.json"
    environment = {
        **os.environ,
        **migration_step.get("env", {}),
        "PATH": str(shim_directory) + os.pathsep + os.environ.get("PATH", ""),
        "RUNNER_TEMP": str(runner_temp),
        "S9_P2_PYTEST_MARKER": str(pytest_marker),
        "S9_P2_VALIDATOR_MARKER": str(validator_marker),
        "S9_P2_TEMPLATE": str(template),
        "S9_P2_WRITE_REPORT": str(int(write_report)),
        "S9_P2_PYTEST_EXIT": str(pytest_exit),
    }

    completed = subprocess.run(
        [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", migration_step["run"]],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == expected_exit, (completed.stdout, completed.stderr)
    assert f"S9-FV-01: {expected_state}" in completed.stdout
    if expected_state == "BLOCKED":
        assert "S9-FV-01: PASS" not in completed.stdout
    assert json.loads(pytest_marker.read_text())[:2] == ["-m", "pytest"]
    assert json.loads(validator_marker.read_text()) == [
        _VALIDATOR,
        str(report),
        "--pytest-exit-code",
        str(pytest_exit),
    ]
    if seed_old_report:
        assert not report.exists(), "Stary poprawny raport musi zostać usunięty przed pytest"

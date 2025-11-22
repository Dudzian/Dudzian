from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pythonpath_env() -> dict[str, str]:
    env = dict(os.environ)
    python_path = env.get("PYTHONPATH")
    root_str = str(ROOT)
    env["PYTHONPATH"] = root_str if not python_path else f"{root_str}{os.pathsep}{python_path}"
    return env


def test_stage5_training_report_command_default_subcommand(tmp_path: Path) -> None:
    env = _pythonpath_env()
    signing_key = base64.b64encode(b"stage5-training-signing-key-012345").decode("ascii")
    env["STAGE5_TRAINING_HMAC"] = signing_key

    command = [
        sys.executable,
        "-m",
        "scripts.log_stage5_training",
        "S5-TRAIN-2024-05-15",
        "Stage5 Compliance",
        "Anna Trainer",
        "--summary",
        "Przegląd wymagań Stage5",
        "--participants",
        "Anna,Bob",
        "--topics",
        "SLO,Resilience",
        "--materials",
        "slides.pdf,playbook.md",
        "--compliance-tags",
        "stage5,compliance",
        "--signing-key-env",
        "STAGE5_TRAINING_HMAC",
    ]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())

    output_path = Path(payload["output"])
    if not output_path.is_absolute():
        output_path = tmp_path / output_path

    content = json.loads(output_path.read_text(encoding="utf-8"))
    assert content["session_id"] == "S5-TRAIN-2024-05-15"
    assert content["title"] == "Stage5 Compliance"
    assert content["trainer"] == "Anna Trainer"
    assert content["signature"]["algorithm"] == "HMAC-SHA256"


def test_stage5_training_report_command_script_entrypoint_defaults(tmp_path: Path) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    signing_key = base64.b64encode(b"stage5-training-signing-key-012345").decode("ascii")
    env["STAGE5_TRAINING_HMAC"] = signing_key

    command = [
        sys.executable,
        str(ROOT / "scripts" / "log_stage5_training.py"),
        "S5-TRAIN-2024-05-16",
        "Stage5 Compliance",
        "Jan Trainer",
        "--summary",
        "Warsztat Stage5",
        "--participants",
        "Jan,Ewa",
        "--materials",
        "slides.pdf,notes.md",
        "--signing-key-env",
        "STAGE5_TRAINING_HMAC",
    ]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())

    output_path = Path(payload["output"])
    if not output_path.is_absolute():
        output_path = tmp_path / output_path

    content = json.loads(output_path.read_text(encoding="utf-8"))
    assert content["session_id"] == "S5-TRAIN-2024-05-16"
    assert content["trainer"] == "Jan Trainer"
    assert content["signature"]["algorithm"] == "HMAC-SHA256"


def test_stage5_training_register_command_script_entrypoint(tmp_path: Path) -> None:
    env = dict(os.environ)

    log_key_path = tmp_path / "secrets" / "stage5_training.key"
    log_key_path.parent.mkdir(parents=True, exist_ok=True)
    log_key_path.write_bytes(os.urandom(48))
    if os.name != "nt":
        os.chmod(log_key_path, 0o600)

    decision_key_path = tmp_path / "secrets" / "decision_log_stage5.key"
    decision_key_path.write_bytes(os.urandom(48))
    if os.name != "nt":
        os.chmod(decision_key_path, 0o600)

    artifact = tmp_path / "var" / "audit" / "training" / "slides.pdf"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"dummy")

    training_log = tmp_path / "var" / "audit" / "training" / "stage5_training_log.jsonl"
    decision_log = tmp_path / "audit" / "decision_logs" / "runtime.jsonl"
    decision_log.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "scripts/log_stage5_training.py",
        "--log-path",
        str(training_log),
        "--training-date",
        "2024-05-18",
        "--participant",
        "Jan Nowak",
        "--participant",
        "Ewa Wiśniewska",
        "--facilitator",
        "Anna Kowalska",
        "--location",
        "Sala 3A",
        "--material",
        "Prezentacja PDF",
        "--artifact",
        str(artifact),
        "--log-hmac-key-file",
        str(log_key_path),
        "--decision-log-path",
        str(decision_log),
        "--decision-log-hmac-key-file",
        str(decision_key_path),
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout_payload = json.loads(result.stdout.strip())
    assert stdout_payload["log_path"] == str(training_log)
    assert stdout_payload["session_id"].startswith("stage5-training-")

    log_lines = training_log.read_text(encoding="utf-8").splitlines()
    assert log_lines
    entry = json.loads(log_lines[-1])
    assert entry["schema"] == "stage5.training_log"
    assert entry["facilitator"] == "Anna Kowalska"
    assert entry["participants"] == ["Jan Nowak", "Ewa Wiśniewska"]
    assert entry["signature"]["algorithm"] == "HMAC-SHA256"

    decision_lines = decision_log.read_text(encoding="utf-8").splitlines()
    assert decision_lines
    decision_entry = json.loads(decision_lines[-1])
    assert decision_entry["schema"] == "stage5.training_session"
    assert decision_entry["location"] == "Sala 3A"
    assert decision_entry["signature"]["algorithm"] == "HMAC-SHA256"


def test_stage5_training_register_command_module_entrypoint(tmp_path: Path) -> None:
    env = _pythonpath_env()

    log_key_path = tmp_path / "secrets" / "stage5_training.key"
    log_key_path.parent.mkdir(parents=True, exist_ok=True)
    log_key_path.write_bytes(os.urandom(48))
    if os.name != "nt":
        os.chmod(log_key_path, 0o600)

    artifact = tmp_path / "var" / "audit" / "training" / "slides.pdf"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"dummy")

    training_log = tmp_path / "var" / "audit" / "training" / "stage5_training_log.jsonl"

    command = [
        sys.executable,
        "-m",
        "scripts.log_stage5_training",
        "register",
        "--log-path",
        str(training_log),
        "--training-date",
        "2024-05-20",
        "--participant",
        "Jan Nowak",
        "--facilitator",
        "Anna Kowalska",
        "--location",
        "Sala 2B",
        "--artifact",
        str(artifact),
        "--log-hmac-key-file",
        str(log_key_path),
    ]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout_payload = json.loads(result.stdout.strip())
    assert stdout_payload["log_path"] == str(training_log)
    assert stdout_payload["session_id"].startswith("stage5-training-")

    log_lines = training_log.read_text(encoding="utf-8").splitlines()
    assert log_lines
    entry = json.loads(log_lines[-1])
    assert entry["schema"] == "stage5.training_log"
    assert entry["facilitator"] == "Anna Kowalska"
    assert entry["participants"] == ["Jan Nowak"]
    assert entry["signature"]["algorithm"] == "HMAC-SHA256"


def test_stage5_training_register_command_module_entrypoint_defaults(tmp_path: Path) -> None:
    env = _pythonpath_env()

    log_key_path = tmp_path / "secrets" / "stage5_training.key"
    log_key_path.parent.mkdir(parents=True, exist_ok=True)
    log_key_path.write_bytes(os.urandom(48))
    if os.name != "nt":
        os.chmod(log_key_path, 0o600)

    training_log = tmp_path / "var" / "audit" / "training" / "stage5_training_log.jsonl"

    command = [
        sys.executable,
        "-m",
        "scripts.log_stage5_training",
        "--log-path",
        str(training_log),
        "--training-date",
        "2024-05-22",
        "--participant",
        "Alicja Example",
        "--facilitator",
        "Katarzyna Example",
        "--location",
        "Sala 5C",
        "--log-hmac-key-file",
        str(log_key_path),
    ]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout_payload = json.loads(result.stdout.strip())
    assert stdout_payload["log_path"] == str(training_log)
    assert stdout_payload["session_id"].startswith("stage5-training-")

    log_lines = training_log.read_text(encoding="utf-8").splitlines()
    assert log_lines
    entry = json.loads(log_lines[-1])
    assert entry["schema"] == "stage5.training_log"
    assert entry["participants"] == ["Alicja Example"]
    assert entry["signature"]["algorithm"] == "HMAC-SHA256"


def test_stage5_training_module_help_shows_top_level_usage() -> None:
    env = _pythonpath_env()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.log_stage5_training",
            "--help",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage: log_stage5_training.py [-h] {report|register}" in result.stdout


def test_stage5_training_script_help_shows_top_level_usage() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/log_stage5_training.py",
            "--help",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage: log_stage5_training.py [-h] {report|register}" in result.stdout


def test_stage5_training_module_register_help_shows_subcommand_usage() -> None:
    env = _pythonpath_env()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.log_stage5_training",
            "register",
            "--help",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage: log_stage5_training.py register" in result.stdout


def test_stage5_training_script_report_help_shows_subcommand_usage() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/log_stage5_training.py",
            "report",
            "--help",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage: log_stage5_training.py report" in result.stdout


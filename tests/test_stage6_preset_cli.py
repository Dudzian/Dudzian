from __future__ import annotations
import hashlib
import json
from pathlib import Path

import platform
import subprocess
import sys

import pytest
import yaml
from bot_core.security.file_storage import EncryptedFileSecretStorage
from bot_core.runtime import stage6_preset_cli as preset_editor_cli


def _copy_core_config(tmp_path: Path) -> Path:
    source = Path("config/core.yaml")
    target = tmp_path / "core.yaml"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_stage6_cli_migrates_preset_and_secrets(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_payload = {
        "fraction": 0.25,
        "risk": {
            "max_daily_loss_pct": 0.03,
            "risk_per_trade": 0.1,
            "portfolio_risk": 0.2,
            "max_open_positions": 4,
        },
    }
    preset_path.write_text(json.dumps(preset_payload, ensure_ascii=False), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text(
        "api:\n  key: secret_key\n  meta:\n    account: primary\n", encoding="utf-8"
    )
    secrets_output = tmp_path / "vault.json"
    destination = tmp_path / "result.yaml"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "CLI Stage6",
            "--runtime-entrypoint",
            "trading_gui",
            "--output",
            str(destination),
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(secrets_output),
            "--secret-passphrase",
            "stage6-pass",
        ]
    )

    assert exit_code == 0
    rendered = destination.read_text(encoding="utf-8")
    assert "cli_stage6" in rendered
    assert "trading_gui" in rendered

    payload = yaml.safe_load(rendered)
    budgets = payload["portfolio_governors"]["stage6_core"]["risk_budgets"]
    assert "cli_stage6" in budgets

    storage = EncryptedFileSecretStorage(secrets_output, "stage6-pass")
    assert storage.get_secret("api") is not None
    assert storage.get_secret("api") == '{"key":"secret_key","meta":{"account":"primary"}}'

    output = capsys.readouterr().out
    assert "Zapisano profil" in output
    assert "Zapisano 1 sekretów" in output


def test_stage6_cli_accepts_passphrase_file(tmp_path: Path) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.1}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text("exchange:\n  api_key: k\n", encoding="utf-8")
    secrets_output = tmp_path / "vault.json"
    pass_file = tmp_path / "pass.txt"
    pass_file.write_text("stage6-file-pass\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "file-pass",  # sprawdzamy tylko przebieg migracji
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(secrets_output),
            "--secret-passphrase-file",
            str(pass_file),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    storage = EncryptedFileSecretStorage(secrets_output, "stage6-file-pass")
    assert storage.get_secret("exchange") == '{"api_key":"k"}'

    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["secrets"]["output_passphrase"] == {
        "provided": True,
        "source": "file",
        "identifier": str(pass_file),
        "used": True,
        "rotated": False,
    }
    assert "legacy_security_passphrase" not in payload["secrets"]
    tool = payload["tool"]
    expected_python = platform.python_version()
    assert tool["module"] == "bot_core.runtime.stage6_preset_cli"
    assert tool["python"] == expected_python
    assert tool["platform"] == platform.platform()
    assert tool["package"] == "dudzian-bot"
    assert tool["package_available"] in {True, False}
    if tool["package_available"]:
        assert isinstance(tool["version"], str) and tool["version"]
    else:
        assert tool["version"] is None
    assert tool["executable"] == sys.executable
    if tool["git_available"]:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        assert tool["git_commit"] == commit
        assert tool["git_commit_error"] is None
    else:
        assert tool["git_commit"] is None
        assert tool["git_commit_error"] is None or isinstance(tool["git_commit_error"], str)
    assert payload["warnings"] == []


def test_stage6_cli_requires_matching_secret_flags(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.1}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text("api:\n  key: value\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        preset_editor_cli.main(
            [
                "--core-config",
                str(core_copy),
                "--preset",
                str(preset_path),
                "--secrets-input",
                str(secrets_input),
                "--secret-passphrase",
                "stage6-pass",
            ]
        )

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "--secrets-output" in stderr


def test_stage6_cli_rejects_legacy_security_flags(tmp_path: Path) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.2}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        preset_editor_cli.main(
            [
                "--core-config",
                str(core_copy),
                "--preset",
                str(preset_path),
                "--profile-name",
                "legacy-security",
                "--legacy-security-file",
                str(tmp_path / "api_keys.enc"),
            ]
        )

    message = str(excinfo.value)
    assert "SecurityManager" in message
    assert "dudzian-migrate" in message


def test_stage6_cli_defaults_desktop_vault(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.yaml"
    preset_path.write_text("fraction: 0.15\n", encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text("api:\n  token: abc123\n", encoding="utf-8")

    desktop_root = tmp_path / "desktop"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "auto-vault",
            "--secrets-input",
            str(secrets_input),
            "--secret-passphrase",
            "desktop-pass",
            "--desktop-root",
            str(desktop_root),
        ]
    )

    assert exit_code == 0
    storage = EncryptedFileSecretStorage(desktop_root / "api_keys.vault", "desktop-pass")
    assert storage.get_secret("api") == '{"token":"abc123"}'

    output = capsys.readouterr().out
    assert "Użyto domyślnego magazynu" in output
    assert str(desktop_root / "api_keys.vault") in output


def test_stage6_cli_allows_desktop_root_without_secrets(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.2}), encoding="utf-8")

    desktop_root = tmp_path / "desktop"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "desktop-only",
            "--desktop-root",
            str(desktop_root),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Zapisano profil" in output
    assert "Użyto domyślnego magazynu" not in output
    assert not (desktop_root / "api_keys.vault").exists()


def test_stage6_cli_dry_run_skips_writes(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    before = core_copy.read_text(encoding="utf-8")

    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(
        json.dumps(
            {
                "fraction": 0.3,
                "risk": {"max_daily_loss_pct": 0.02, "risk_per_trade": 0.05},
            }
        ),
        encoding="utf-8",
    )

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text("api:\n  token: dry\n", encoding="utf-8")
    secrets_output = tmp_path / "stage6.vault"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "dry-run-check",
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(secrets_output),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert core_copy.read_text(encoding="utf-8") == before
    assert not secrets_output.exists()

    output = capsys.readouterr().out
    assert "Tryb dry-run" in output
    assert "Zapisano profil" not in output


def test_stage6_cli_filters_secret_keys(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.42}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text(
        "api:\n  key: keep\nslack:\n  token: drop\nops:\n  alert: excluded\n",
        encoding="utf-8",
    )

    secrets_output = tmp_path / "stage6.vault"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "filters",  # tylko sprawdzamy filtrację sekretów
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(secrets_output),
            "--secret-passphrase",
            "filters-pass",
            "--secrets-include",
            "api",
            "--secrets-include",
            "missing",
            "--secrets-exclude",
            "ops",
        ]
    )

    assert exit_code == 0

    storage = EncryptedFileSecretStorage(secrets_output, "filters-pass")
    assert storage.get_secret("api") == '{"key":"keep"}'
    assert storage.get_secret("slack") is None
    assert storage.get_secret("ops") is None

    output = capsys.readouterr().out
    assert "Nie znaleziono sekretów wymaganych przez --secrets-include: missing" in output
    assert "Pominięto sekrety spoza listy --secrets-include: slack" in output
    assert "Pominięto sekrety oznaczone --secrets-exclude: ops" in output
    assert "Zapisano 1 sekretów" in output


def test_stage6_cli_previews_secret_keys(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.5}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text(
        "api:\n  key: preview\nslack:\n  token: keep-private\n",
        encoding="utf-8",
    )

    secrets_output = tmp_path / "stage6_preview.vault"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "preview",
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(secrets_output),
            "--secrets-preview",
            "--dry-run",
        ]
    )

    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Podgląd sekretów (2): api, slack" in output
    assert "key: preview" not in output
    assert "keep-private" not in output
    assert "Tryb dry-run" in output


def test_stage6_cli_preview_without_output_path(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.22}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text(
        "api:\n  key: preview\nslack:\n  token: dry-run\n",
        encoding="utf-8",
    )

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "preview-only",
            "--secrets-input",
            str(secrets_input),
            "--secrets-preview",
            "--dry-run",
        ]
    )

    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Podgląd sekretów (2): api, slack" in output
    assert "nie wskazano --secrets-output" in output
    assert "Zapisano" not in output


def test_stage6_cli_secret_filters_support_glob_patterns(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.37}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text(
        (
            "binance_api_key: keep\n"
            "binance_api_secret: remove\n"
            "slack_token: skip-by-include\n"
        ),
        encoding="utf-8",
    )

    secrets_output = tmp_path / "stage6.vault"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "glob-filters",
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(secrets_output),
            "--secret-passphrase",
            "glob-pass",
            "--secrets-include",
            "binance_api_*",
            "--secrets-exclude",
            "*secret",
        ]
    )

    assert exit_code == 0

    storage = EncryptedFileSecretStorage(secrets_output, "glob-pass")
    assert storage.get_secret("binance_api_key") == "keep"
    assert storage.get_secret("binance_api_secret") is None
    assert storage.get_secret("slack_token") is None

    output = capsys.readouterr().out
    assert (
        "Nie znaleziono sekretów wymaganych przez --secrets-include: binance_api_*"
        not in output
    )
    assert "Pominięto sekrety spoza listy --secrets-include: slack_token" in output
    assert "Pominięto sekrety oznaczone --secrets-exclude: binance_api_secret" in output


def test_stage6_cli_secret_filters_can_skip_all(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.13}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text("api:\n  key: keep\n", encoding="utf-8")
    secrets_output = tmp_path / "stage6.vault"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "filters-empty",
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(secrets_output),
            "--secret-passphrase",
            "filters-pass",
            "--secrets-include",
            "missing",
        ]
    )

    assert exit_code == 0
    assert not secrets_output.exists()

    output = capsys.readouterr().out
    assert "Nie znaleziono sekretów wymaganych przez --secrets-include: missing" in output
    assert "Pominięto zapis sekretów: brak dopasowanych wpisów (po filtrach)." in output


def test_stage6_cli_can_create_backup(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    original_payload = core_copy.read_text(encoding="utf-8")
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.55}), encoding="utf-8")

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "backup-test",
            "--core-backup",
        ]
    )

    assert exit_code == 0

    backup_path = core_copy.with_name("core.yaml.bak")
    assert backup_path.exists()
    assert backup_path.read_text(encoding="utf-8") == original_payload

    output = capsys.readouterr().out
    assert f"Utworzono kopię zapasową {core_copy}" in output
    assert "Zapisano profil" in output


def test_stage6_cli_skips_backup_in_dry_run(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.61}), encoding="utf-8")

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "backup-dry-run",
            "--core-backup",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    backup_path = core_copy.with_name("core.yaml.bak")
    assert not backup_path.exists()

    output = capsys.readouterr().out
    assert "Tryb dry-run: pominięto utworzenie kopii zapasowej (--core-backup)." in output


def test_stage6_cli_rejects_backup_same_path(tmp_path: Path) -> None:
    core_copy = _copy_core_config(tmp_path)
    original_payload = core_copy.read_text(encoding="utf-8")
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.72}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        preset_editor_cli.main(
            [
                "--core-config",
                str(core_copy),
                "--preset",
                str(preset_path),
                "--profile-name",
                "backup-same",
                "--core-backup",
                str(core_copy),
            ]
        )

    assert "Ścieżka kopii zapasowej" in str(excinfo.value)
    assert core_copy.read_text(encoding="utf-8") == original_payload


def test_stage6_cli_prints_core_diff(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.42}), encoding="utf-8")

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "diff-check",
            "--core-diff",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"Podgląd zmian {core_copy}:" in output
    assert f"{core_copy} (przed migracją)" in output
    assert f"{core_copy} (po migracji)" in output
    assert "+++" in output and "---" in output


def test_stage6_cli_core_diff_for_new_output(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.33}), encoding="utf-8")
    destination = tmp_path / "fresh_core.yaml"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "diff-new-output",
            "--output",
            str(destination),
            "--core-diff",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"Podgląd zmian {destination}:" in output
    assert f"{destination} (nowy plik)" in output
    assert f"{destination} (po migracji)" in output
    assert destination.exists()


def test_stage6_cli_writes_summary_file(tmp_path: Path) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.3}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text("exchange:\n  api_key: k\n", encoding="utf-8")

    destination = tmp_path / "stage6.yaml"
    backup_path = tmp_path / "stage6.backup.yaml"
    vault_path = tmp_path / "stage6.vault"
    summary_path = tmp_path / "artifacts" / "summary.json"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "summary-profile",
            "--output",
            str(destination),
            "--core-backup",
            str(backup_path),
            "--core-diff",
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(vault_path),
            "--secret-passphrase",
            "summary-pass",
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert summary_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["profile_name"] == "summary-profile"
    assert payload["core_config_destination"] == str(destination)
    assert payload["core_backup_requested"] is True
    assert payload["core_backup_path"] == str(backup_path)
    assert payload["core_backup_checksum"] == hashlib.sha256(
        backup_path.read_bytes()
    ).hexdigest()
    assert payload["core_diff_requested"] is True
    assert payload["dry_run"] is False
    assert payload["core_original_checksum"] is None
    assert payload["core_rendered_checksum"] == hashlib.sha256(
        destination.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    assert payload["warnings"] == []
    assert payload["secrets"]["planned"] == 1
    assert payload["secrets"]["written"] == 1
    assert payload["secrets"]["output_path"] == str(vault_path)
    assert payload["secrets"]["source_path"] == str(secrets_input)
    assert payload["secrets"]["source_checksum"] == hashlib.sha256(
        secrets_input.read_bytes()
    ).hexdigest()
    assert payload["secrets"]["filters"] == {"include": [], "exclude": []}
    assert payload["secrets"]["dry_run_skipped"] is False
    assert payload["secrets"]["output_checksum"] == hashlib.sha256(
        vault_path.read_bytes()
    ).hexdigest()
    assert "legacy_security_salt_path" not in payload["secrets"]
    assert "legacy_security_salt_checksum" not in payload["secrets"]
    assert payload["secrets"]["output_passphrase"] == {
        "provided": True,
        "source": "inline",
        "identifier": None,
        "used": True,
        "rotated": False,
    }
    assert "legacy_security_passphrase" not in payload["secrets"]
    invocation = payload["cli_invocation"]
    assert isinstance(invocation["argv"], list)
    assert "***REDACTED***" in invocation["argv"]
    assert all("summary-pass" not in item for item in invocation["argv"])
    assert "summary-pass" not in invocation["command"]


def test_stage6_cli_summary_tracks_secret_passphrase_env(
    tmp_path: Path, monkeypatch
) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.33}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text("exchange:\n  api_key: token\n", encoding="utf-8")
    vault_path = tmp_path / "stage6.vault"
    summary_path = tmp_path / "summary.json"

    monkeypatch.setenv("STAGE6_SECRET_PASS", "env-secret")

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "env-secret-profile",
            "--output",
            str(tmp_path / "stage6.yaml"),
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(vault_path),
            "--secret-passphrase-env",
            "STAGE6_SECRET_PASS",
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert summary_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["secrets"]["output_passphrase"] == {
        "provided": True,
        "source": "env",
        "identifier": "STAGE6_SECRET_PASS",
        "used": True,
        "rotated": False,
    }
    assert "legacy_security_passphrase" not in payload["secrets"]


def test_stage6_cli_summary_records_original_checksum(tmp_path: Path) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.35}), encoding="utf-8")

    original_bytes = core_copy.read_bytes()

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "checksum-profile",
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    )

    assert exit_code == 0

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["core_original_checksum"] == hashlib.sha256(original_bytes).hexdigest()
    assert summary["core_rendered_checksum"] == hashlib.sha256(
        core_copy.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()


def test_stage6_cli_summary_records_security_source_checksums(tmp_path: Path) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.41}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text("binance:\n  api_key: AAA\n  secret_key: BBB\n", encoding="utf-8")

    summary_path = tmp_path / "summary.json"
    vault_path = tmp_path / "stage6.vault"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "checksum-security",
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(vault_path),
            "--secret-passphrase",
            "stage6-pass",
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    secrets = payload["secrets"]
    assert secrets["source_label"] == f"plik {secrets_input}"
    assert secrets["source_path"] == str(secrets_input)
    assert secrets["source_checksum"] == hashlib.sha256(secrets_input.read_bytes()).hexdigest()
    assert secrets["output_path"] == str(vault_path)
    assert secrets["output_checksum"] == hashlib.sha256(vault_path.read_bytes()).hexdigest()
    assert "legacy_security_salt_path" not in secrets
    assert "legacy_security_passphrase" not in secrets


def test_stage6_cli_summary_redacts_inline_passphrases(tmp_path: Path) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.27}), encoding="utf-8")

    secrets_input = tmp_path / "legacy_secrets.yaml"
    secrets_input.write_text("binance:\n  api_key: AAA\n", encoding="utf-8")

    summary_path = tmp_path / "summary.json"
    vault_path = tmp_path / "preview.vault"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "redacted-profile",
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(vault_path),
            "--secret-passphrase",
            "inline-pass",
            "--dry-run",
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    invocation = payload["cli_invocation"]
    assert invocation["argv"].count("***REDACTED***") >= 1
    assert "inline-pass" not in invocation["command"]
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "inline-pass" not in summary_text


def test_stage6_cli_summary_includes_checksum_warnings(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.4}), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    backup_path = tmp_path / "core.backup.yaml"

    def failing_checksum(path: Path) -> str:
        raise OSError("checksum blocked")

    monkeypatch.setattr(preset_editor_cli, "_compute_file_checksum", failing_checksum)

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--preset",
            str(preset_path),
            "--profile-name",
            "warnings-profile",
            "--core-backup",
            str(backup_path),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert summary_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["warnings"]
    assert payload["warnings"][0].startswith(
        f"Ostrzeżenie: nie udało się obliczyć sumy SHA-256 dla {backup_path}"
    )
    assert payload["core_backup_checksum"] is None

    output = capsys.readouterr().out
    assert "Ostrzeżenie: nie udało się obliczyć sumy SHA-256" in output

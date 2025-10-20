from __future__ import annotations
import json
from pathlib import Path

import pytest
import yaml
from bot_core.security.file_storage import EncryptedFileSecretStorage
from KryptoLowca.managers.security_manager import SecurityManager
from KryptoLowca.scripts import preset_editor_cli


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
            "--legacy-preset",
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

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--legacy-preset",
            str(preset_path),
            "--profile-name",
            "file-pass",  # sprawdzamy tylko przebieg migracji
            "--secrets-input",
            str(secrets_input),
            "--secrets-output",
            str(secrets_output),
            "--secret-passphrase-file",
            str(pass_file),
        ]
    )

    assert exit_code == 0
    storage = EncryptedFileSecretStorage(secrets_output, "stage6-file-pass")
    assert storage.get_secret("exchange") == '{"api_key":"k"}'


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
                "--legacy-preset",
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


def test_stage6_cli_imports_security_manager_file(tmp_path: Path, capsys) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.2}), encoding="utf-8")

    legacy_file = tmp_path / "api_keys.enc"
    salt_file = tmp_path / "salt.bin"
    manager = SecurityManager(key_file=str(legacy_file), salt_file=str(salt_file))
    manager.save_encrypted_keys(
        {"binance": {"api_key": "AAA", "secret_key": "BBB"}},
        password="legacy-pass",
    )

    vault_path = tmp_path / "stage6.vault"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--legacy-preset",
            str(preset_path),
            "--profile-name",
            "legacy-security",
            "--secrets-output",
            str(vault_path),
            "--secret-passphrase",
            "stage6-pass",
            "--legacy-security-file",
            str(legacy_file),
            "--legacy-security-salt",
            str(salt_file),
            "--legacy-security-passphrase",
            "legacy-pass",
        ]
    )

    assert exit_code == 0

    storage = EncryptedFileSecretStorage(vault_path, "stage6-pass")
    assert storage.get_secret("binance") == '{"api_key":"AAA","secret_key":"BBB"}'

    output = capsys.readouterr().out
    assert "legacy SecurityManager" in output


def test_stage6_cli_imports_security_manager_passphrase_env(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.2}), encoding="utf-8")

    legacy_file = tmp_path / "api_keys.enc"
    salt_file = tmp_path / "salt.bin"
    manager = SecurityManager(key_file=str(legacy_file), salt_file=str(salt_file))
    manager.save_encrypted_keys(
        {"binance": {"api_key": "ENV", "secret_key": "PASS"}},
        password="env-pass",
    )

    monkeypatch.setenv("LEGACY_SECURITY_PASS", "env-pass")

    vault_path = tmp_path / "stage6.vault"

    exit_code = preset_editor_cli.main(
        [
            "--core-config",
            str(core_copy),
            "--legacy-preset",
            str(preset_path),
            "--profile-name",
            "legacy-security-env",
            "--secrets-output",
            str(vault_path),
            "--secret-passphrase",
            "stage6-pass",
            "--legacy-security-file",
            str(legacy_file),
            "--legacy-security-salt",
            str(salt_file),
            "--legacy-security-passphrase-env",
            "LEGACY_SECURITY_PASS",
        ]
    )

    assert exit_code == 0

    storage = EncryptedFileSecretStorage(vault_path, "stage6-pass")
    assert storage.get_secret("binance") == '{"api_key":"ENV","secret_key":"PASS"}'

    output = capsys.readouterr().out
    assert "legacy SecurityManager" in output


def test_stage6_cli_requires_legacy_passphrase_env(tmp_path: Path) -> None:
    core_copy = _copy_core_config(tmp_path)
    preset_path = tmp_path / "legacy.json"
    preset_path.write_text(json.dumps({"fraction": 0.2}), encoding="utf-8")

    legacy_file = tmp_path / "api_keys.enc"
    manager = SecurityManager(key_file=str(legacy_file))
    manager.save_encrypted_keys(
        {"ftx": {"api_key": "A", "secret_key": "B"}},
        password="missing",
    )

    vault_path = tmp_path / "stage6.vault"

    with pytest.raises(SystemExit) as excinfo:
        preset_editor_cli.main(
            [
                "--core-config",
                str(core_copy),
                "--legacy-preset",
                str(preset_path),
                "--profile-name",
                "legacy-missing-env",
                "--secrets-output",
                str(vault_path),
                "--secret-passphrase",
                "stage6-pass",
                "--legacy-security-file",
                str(legacy_file),
                "--legacy-security-passphrase-env",
                "LEGACY_MISSING",
            ]
        )

    assert (
        excinfo.value.code
        == "Zmienna środowiskowa LEGACY_MISSING nie została ustawiona lub jest pusta."
    )


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
            "--legacy-preset",
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
            "--legacy-preset",
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
            "--legacy-preset",
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
            "--legacy-preset",
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
            "--legacy-preset",
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
            "--legacy-preset",
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
            "--legacy-preset",
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
            "--legacy-preset",
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
            "--legacy-preset",
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
            "--legacy-preset",
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
                "--legacy-preset",
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
            "--legacy-preset",
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
            "--legacy-preset",
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

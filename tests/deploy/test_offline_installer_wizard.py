import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from deploy.offline import wizard as wizard_module
from deploy.offline.wizard import InstallerSession, InstallerWizard


def test_installer_session_initializes_directories(tmp_path: Path):
    session = InstallerSession.from_root(tmp_path / "bundle")
    wizard = InstallerWizard(session)

    assert session.root_dir.exists()
    assert session.config_dir.exists()
    assert session.logs_dir.exists()
    assert session.updates_dir.exists()

    summary = wizard.summary()
    assert summary["paths"]["root"] == str(session.root_dir)
    assert summary["paths"]["config"] == str(session.config_dir)
    assert summary["paths"]["logs"] == str(session.logs_dir)
    assert summary["paths"]["updates"] == str(session.updates_dir)
    assert summary["update"] is None


def test_apply_offline_update_copies_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    session = InstallerSession.from_root(tmp_path / "bundle")
    wizard = InstallerWizard(session)

    manifest = tmp_path / "update.json"
    manifest.write_text("{\"version\": \"1.0.0\"}", encoding="utf-8")
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    (payload_dir / "binary.bin").write_bytes(b"data")
    signature = tmp_path / "update.sig"
    signature.write_text("sig", encoding="utf-8")

    calls: list[dict[str, object]] = []

    class DummyResult:
        is_successful = True
        errors: list[str] = []
        warnings = ["test warning"]

    def fake_verify_update_bundle(**kwargs):
        calls.append(kwargs)
        return DummyResult()

    monkeypatch.setattr(wizard_module, "verify_update_bundle", fake_verify_update_bundle)

    target = wizard.apply_offline_update(
        manifest_path=manifest,
        payload_dir=payload_dir,
        signature_path=signature,
        hmac_keys={"bundle": b"secret"},
    )

    assert calls, "verify_update_bundle was not invoked"
    assert calls[0]["manifest_path"] == manifest
    assert calls[0]["base_dir"] == payload_dir
    assert target.exists()
    assert (target / "binary.bin").read_bytes() == b"data"
    assert (target / manifest.name).read_text(encoding="utf-8")
    assert (target / signature.name).read_text(encoding="utf-8")

    summary = wizard.summary()
    assert summary["update"]["status"] == "ok"
    assert summary["update"]["key_id"] == "bundle"
    assert summary["update"]["warnings"] == ["test warning"]
    assert summary["update"]["target"] == str(target)
    assert summary["update"]["payload_archive"] is None


def test_apply_offline_update_tries_all_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    session = InstallerSession.from_root(tmp_path / "bundle")
    wizard = InstallerWizard(session)

    manifest = tmp_path / "update.json"
    manifest.write_text("{\"version\": \"1.0.0\"}", encoding="utf-8")
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    (payload_dir / "binary.bin").write_bytes(b"data")

    attempts: list[bytes | None] = []

    class DummyResult:
        def __init__(self, success: bool, *, warnings: list[str] | None = None, errors: list[str] | None = None):
            self.is_successful = success
            self.warnings = warnings or []
            self.errors = errors or []

    responses = [
        DummyResult(False, warnings=["first"], errors=["signature mismatch"]),
        DummyResult(True, warnings=["second"]),
    ]

    def fake_verify_update_bundle(**kwargs):
        attempts.append(kwargs.get("hmac_key"))
        return responses.pop(0)

    monkeypatch.setattr(wizard_module, "verify_update_bundle", fake_verify_update_bundle)

    target = wizard.apply_offline_update(
        manifest_path=manifest,
        payload_dir=payload_dir,
        hmac_keys={"invalid": b"bad", "valid": b"good"},
    )

    assert attempts == [b"bad", b"good"]
    summary = wizard.summary()
    assert summary["update"]["status"] == "ok"
    assert summary["update"]["key_id"] == "valid"
    assert summary["update"]["warnings"] == ["first", "second"]
    assert summary["update"]["target"] == str(target)
    assert summary["update"]["payload_archive"] is None


def test_apply_offline_update_accepts_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    session = InstallerSession.from_root(tmp_path / "bundle")
    wizard = InstallerWizard(session)

    manifest = tmp_path / "update.json"
    manifest.write_text("{\"version\": \"1.0.0\"}", encoding="utf-8")

    archive_source = tmp_path / "archive_source"
    payload_dir = archive_source / "payload"
    nested_dir = payload_dir / "package"
    nested_dir.mkdir(parents=True)
    (nested_dir / "binary.bin").write_bytes(b"data")

    archive_path = Path(shutil.make_archive(str(tmp_path / "payload"), "zip", root_dir=archive_source, base_dir="payload"))

    recorded_base_dir: list[Path] = []

    class DummyResult:
        is_successful = True
        errors: list[str] = []
        warnings: list[str] = []

    def fake_verify_update_bundle(**kwargs):
        base_dir = kwargs["base_dir"]
        assert base_dir.is_dir()
        assert (base_dir / "package" / "binary.bin").exists()
        recorded_base_dir.append(base_dir)
        return DummyResult()

    monkeypatch.setattr(wizard_module, "verify_update_bundle", fake_verify_update_bundle)

    target = wizard.apply_offline_update(
        manifest_path=manifest,
        payload_archive=archive_path,
    )

    assert recorded_base_dir, "Oczekiwano wywołania verify_update_bundle"
    assert (target / "package" / "binary.bin").read_bytes() == b"data"

    summary = wizard.summary()
    assert summary["update"]["payload_archive"] == str(archive_path)


def test_apply_offline_update_missing_manifest(tmp_path: Path):
    session = InstallerSession.from_root(tmp_path / "bundle")
    wizard = InstallerWizard(session)

    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        wizard.apply_offline_update(manifest_path=tmp_path / "missing.json", payload_dir=payload_dir)


def test_apply_offline_update_records_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    session = InstallerSession.from_root(tmp_path / "bundle")
    wizard = InstallerWizard(session)

    manifest = tmp_path / "update.json"
    manifest.write_text("{\"version\": \"1.0.0\"}", encoding="utf-8")
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()

    class DummyResult:
        is_successful = False
        warnings: list[str] = ["warn"]
        errors: list[str] = ["bad"]

    def fake_verify_update_bundle(**_):
        return DummyResult()

    monkeypatch.setattr(wizard_module, "verify_update_bundle", fake_verify_update_bundle)

    with pytest.raises(RuntimeError):
        wizard.apply_offline_update(
            manifest_path=manifest,
            payload_dir=payload_dir,
            hmac_keys={"invalid": b"bad"},
        )

    summary = wizard.summary()
    assert summary["update"]["status"] == "invalid"
    assert summary["update"]["warnings"] == ["warn"]
    assert summary["update"]["errors"] == ["invalid: bad"]
    assert summary["update"]["payload_archive"] is None


def test_persist_summary_writes_default_file(tmp_path: Path):
    session = InstallerSession.from_root(tmp_path / "bundle")
    wizard = InstallerWizard(session)

    output_path = wizard.persist_summary()

    assert output_path == session.logs_dir / "offline_installer_summary.json"
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["paths"]["root"] == str(session.root_dir)
    assert "generated_at" in payload
    timestamp = payload["generated_at"].replace("Z", "+00:00")
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None


def test_persist_summary_accepts_custom_path(tmp_path: Path):
    session = InstallerSession.from_root(tmp_path / "bundle")
    wizard = InstallerWizard(session)

    custom_target = tmp_path / "reports" / "summary.json"
    persisted = wizard.persist_summary(custom_target)

    assert persisted == custom_target
    assert custom_target.exists()


def test_load_multiple_hmac_key_sources(tmp_path: Path):
    first = tmp_path / "keys1.json"
    second = tmp_path / "keys2.json"
    first.write_text(json.dumps({"keys": {"a": "hex:11"}}), encoding="utf-8")
    second.write_text(json.dumps({"keys": {"b": "hex:22"}}), encoding="utf-8")

    keys = wizard_module._load_hmac_keys([first, second])

    assert keys == {"a": bytes.fromhex("11"), "b": bytes.fromhex("22")}


def test_main_accepts_multiple_update_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    (payload_dir / "file.bin").write_text("data", encoding="utf-8")

    recorded_paths: list[Path] = []

    def fake_load_hmac_keys(paths):
        recorded_paths.extend(paths)
        return {"key": b"secret"}

    monkeypatch.setattr(wizard_module, "_load_hmac_keys", fake_load_hmac_keys)

    recorded_hmac_keys = {}

    def fake_apply(self, **kwargs):
        recorded_hmac_keys.update(kwargs.get("hmac_keys", {}))
        return self.session.updates_dir

    monkeypatch.setattr(InstallerWizard, "apply_offline_update", fake_apply)

    exit_code = wizard_module.main(
        [
            "--root",
            str(tmp_path / "bundle"),
            "--update-manifest",
            str(manifest),
            "--update-payload",
            str(payload_dir),
            "--apply-update",
            "--update-keys",
            str(tmp_path / "keys1.json"),
            "--update-keys",
            str(tmp_path / "keys2.json"),
            "--no-summary-file",
        ]
    )

    assert exit_code == 0
    assert recorded_paths == [str(tmp_path / "keys1.json"), str(tmp_path / "keys2.json")]
    assert recorded_hmac_keys == {"key": b"secret"}


def test_main_supports_update_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    archive_dir = tmp_path / "payload"
    inner = archive_dir / "pkg"
    inner.mkdir(parents=True)
    (inner / "file.bin").write_bytes(b"data")

    archive_path = Path(shutil.make_archive(str(tmp_path / "payload_archive"), "zip", root_dir=tmp_path, base_dir="payload"))

    captured_args: dict[str, object] = {}

    def fake_apply(self, **kwargs):
        captured_args.update(kwargs)
        return self.session.updates_dir

    monkeypatch.setattr(InstallerWizard, "apply_offline_update", fake_apply)

    exit_code = wizard_module.main(
        [
            "--root",
            str(tmp_path / "bundle"),
            "--update-manifest",
            str(manifest),
            "--update-archive",
            str(archive_path),
            "--apply-update",
            "--no-summary-file",
        ]
    )

    assert exit_code == 0
    assert captured_args["payload_archive"] == Path(archive_path)
    assert captured_args["payload_dir"] is None

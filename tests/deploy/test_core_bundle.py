import base64
import hashlib
import hmac
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, List

import pytest

import deploy.packaging.build_core_bundle as build_core_bundle_module

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.packaging import BundleInputs, CoreBundleBuilder, build_from_cli
from bot_core.security.signing import canonical_json_bytes


def _write_signing_key(path: Path, *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(48))
    if os.name != "nt":
        path.chmod(mode)
    return path


@pytest.mark.parametrize("platform,extension", [("linux", ".tar.gz"), ("macos", ".tar.gz"), ("windows", ".zip")])
def test_core_bundle_structure_and_signatures(tmp_path, platform, extension):
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "botd").write_text("daemon-bin", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "app").write_text("ui-bin", encoding="utf-8")

    core_yaml = tmp_path / "core.yaml"
    core_yaml.write_text("risk: conservative", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("TOKEN=demo", encoding="utf-8")

    key = os.urandom(48)

    inputs = BundleInputs(
        daemon_paths=[daemon_dir],
        ui_paths=[ui_dir],
        config_paths={"core.yaml": core_yaml, ".env": env_file},
        fingerprint_placeholder="FP-PLACEHOLDER",
    )

    output_dir = tmp_path / "dist"
    output_dir.mkdir()

    builder = CoreBundleBuilder(
        platform=platform,
        version="1.2.3",
        signing_key=key,
        output_dir=output_dir,
        inputs=inputs,
    )

    bundle_path = builder.build()
    assert bundle_path.exists()
    if extension == ".tar.gz":
        assert bundle_path.suffixes[-2:] == [".tar", ".gz"]
    else:
        assert bundle_path.suffixes[-1] == ".zip"

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()

    if bundle_path.suffix.endswith(".zip"):
        pytest.importorskip("zipfile")
        import zipfile

        with zipfile.ZipFile(bundle_path) as archive:
            archive.extractall(extract_dir)
    else:
        with tarfile.open(bundle_path) as archive:
            archive.extractall(extract_dir)

    bundle_root = extract_dir
    while True:
        entries = list(bundle_root.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            break
        bundle_root = entries[0]

    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["platform"] == platform

    manifest_sig_doc = json.loads((bundle_root / "manifest.sig").read_text(encoding="utf-8"))
    assert manifest_sig_doc["payload"]["path"] == "manifest.json"
    assert manifest_sig_doc["payload"]["sha384"] == hashlib.sha384(
        manifest_path.read_bytes()
    ).hexdigest()
    expected_sig = base64.b64encode(
        hmac.new(
            key,
            canonical_json_bytes(manifest_sig_doc["payload"]),
            hashlib.sha384,
        ).digest()
    ).decode("ascii")
    assert manifest_sig_doc["signature"]["value"] == expected_sig
    assert manifest_sig_doc["signature"]["algorithm"] == "HMAC-SHA384"

    config_dir = bundle_root / "config"
    assert (config_dir / "core.yaml").exists()
    assert (config_dir / "core.yaml.sig").exists()
    assert (config_dir / ".env").exists()
    assert (config_dir / ".env.sig").exists()

    files = {entry["path"] for entry in manifest["files"]}
    assert "config/core.yaml.sig" in files
    assert "config/.env.sig" in files
    assert "bootstrap/verify_fingerprint.py" in files

    verify_script = bundle_root / "bootstrap" / "verify_fingerprint.py"
    fingerprint_file = config_dir / "fingerprint.expected.json"

    env = os.environ.copy()
    env["OEM_BUNDLE_HMAC_KEY"] = base64.b64encode(key).decode("ascii")
    env["OEM_FINGERPRINT"] = "FP-PLACEHOLDER"

    subprocess.run(
        [sys.executable, str(verify_script), "--expected", str(fingerprint_file)],
        check=True,
        env=env,
    )

    install_sh = bundle_root / "bootstrap" / "install.sh"
    if install_sh.exists() and platform != "windows":
        assert os.access(install_sh, os.X_OK)

    for relative_name in ["core.yaml", ".env"]:
        signature_path = config_dir / f"{relative_name}.sig"
        document = json.loads(signature_path.read_text(encoding="utf-8"))
        payload = document["payload"]
        assert payload["path"].startswith("config/")
        target_file = bundle_root / payload["path"]
        assert target_file.exists()
        digest = hashlib.sha384(target_file.read_bytes()).hexdigest()
        assert payload["sha384"] == digest
        recomputed = base64.b64encode(
            hmac.new(key, canonical_json_bytes(payload), hashlib.sha384).digest()
        ).decode("ascii")
        assert document["signature"]["value"] == recomputed
        assert document["signature"]["algorithm"] == "HMAC-SHA384"

    fingerprint_doc = json.loads(
        (config_dir / "fingerprint.expected.json").read_text(encoding="utf-8")
    )
    assert fingerprint_doc["payload"]["fingerprint"] == "FP-PLACEHOLDER"
    fingerprint_sig = fingerprint_doc["signature"]
    fingerprint_expected = base64.b64encode(
        hmac.new(
            key,
            canonical_json_bytes(fingerprint_doc["payload"]),
            hashlib.sha384,
        ).digest()
    ).decode("ascii")
    assert fingerprint_sig["value"] == fingerprint_expected
    assert fingerprint_sig["algorithm"] == "HMAC-SHA384"


def test_core_bundle_builder_rejects_invalid_fingerprint_placeholder(tmp_path):
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "botd").write_text("daemon", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "app").write_text("ui", encoding="utf-8")

    core_yaml = tmp_path / "core.yaml"
    core_yaml.write_text("risk: balanced", encoding="utf-8")

    inputs = BundleInputs(
        daemon_paths=[daemon_dir],
        ui_paths=[ui_dir],
        config_paths={"core.yaml": core_yaml},
        fingerprint_placeholder="INVALID PLACEHOLDER",
    )

    with pytest.raises(ValueError, match="unsupported character"):
        CoreBundleBuilder(
            platform="linux",
            version="1.0.0",
            signing_key=os.urandom(48),
            output_dir=tmp_path,
            inputs=inputs,
        )


def test_core_bundle_cli_invocation_from_subprocess(tmp_path):
    script_path = ROOT / "deploy" / "packaging" / "build_core_bundle.py"

    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "botd").write_text("daemon", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "qtapp").write_text("ui", encoding="utf-8")

    core_yaml = tmp_path / "core.yaml"
    core_yaml.write_text("risk: balanced", encoding="utf-8")
    env_file = tmp_path / "bot.env"
    env_file.write_text("TOKEN=prod", encoding="utf-8")

    signing_key_path = _write_signing_key(tmp_path / "signing.key")

    output_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--platform",
            "linux",
            "--version",
            "9.9.9",
            "--signing-key-path",
            str(signing_key_path),
            "--daemon",
            str(daemon_dir),
            "--ui",
            str(ui_dir),
            "--config",
            f"core.yaml={core_yaml}",
            "--config",
            f"bot.env={env_file}",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=tmp_path,
    )

    archives = list(output_dir.glob("core-oem-9.9.9-linux*.tar.gz"))
    assert archives, "Bundle archive was not generated"
    archive_path = archives[0]

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()

    with tarfile.open(archive_path) as archive:
        archive.extractall(extract_dir)

    bundle_root = extract_dir
    while True:
        entries = list(bundle_root.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            break
        bundle_root = entries[0]

    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["platform"] == "linux"
    assert manifest["version"] == "9.9.9"

    for relative in ("core.yaml", "bot.env"):
        config_file = bundle_root / "config" / relative
        signature_file = config_file.parent / f"{config_file.name}.sig"
        assert config_file.exists()
        assert signature_file.exists()


def _create_basic_cli_environment(tmp_path: Path) -> Dict[str, Path]:
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "botd").write_text("daemon-bin", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "qtapp").write_text("ui-bin", encoding="utf-8")

    signing_key_path = _write_signing_key(tmp_path / "signing.key")

    output_dir = tmp_path / "out"

    return {
        "daemon_dir": daemon_dir,
        "ui_dir": ui_dir,
        "signing_key_path": signing_key_path,
        "output_dir": output_dir,
    }


def _base_cli_args(env: Dict[str, Path]) -> List[str]:
    return [
        "--platform",
        "linux",
        "--version",
        "1.0.0",
        "--signing-key-path",
        str(env["signing_key_path"]),
        "--daemon",
        str(env["daemon_dir"]),
        "--ui",
        str(env["ui_dir"]),
        "--output-dir",
        str(env["output_dir"]),
    ]


def test_build_from_cli_rejects_config_path_traversal(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: aggressive", encoding="utf-8")

    args = _base_cli_args(env) + ["--config", f"../core.yaml={config_file}"]

    with pytest.raises(ValueError):
        build_from_cli(args)


def test_build_from_cli_requires_version_without_dry_run(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = [
        "--platform",
        "linux",
        "--signing-key-path",
        str(env["signing_key_path"]),
        "--daemon",
        str(env["daemon_dir"]),
        "--ui",
        str(env["ui_dir"]),
        "--config",
        f"core.yaml={config_file}",
        "--output-dir",
        str(env["output_dir"]),
    ]

    with pytest.raises(ValueError, match="--version is required"):
        build_from_cli(args)


def test_build_from_cli_rejects_invalid_version(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    args = _base_cli_args(env)
    version_index = args.index("--version") + 1
    args[version_index] = "1/../../2"
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: aggressive", encoding="utf-8")
    args += ["--config", f"core.yaml={config_file}"]

    with pytest.raises(ValueError, match="unsupported character"):
        build_from_cli(args)


def test_build_from_cli_rejects_existing_bundle_archive(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + ["--config", f"core.yaml={config_file}"]

    first_output = build_from_cli(args)
    assert first_output.exists()

    with pytest.raises(FileExistsError, match="already exists"):
        build_from_cli(args)


def test_build_from_cli_dry_run_uses_defaults_and_creates_no_artifacts(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: conservative", encoding="utf-8")

    args = [
        "--platform",
        "linux",
        "--signing-key-path",
        str(env["signing_key_path"]),
        "--daemon",
        str(env["daemon_dir"]),
        "--ui",
        str(env["ui_dir"]),
        "--config",
        f"core.yaml={config_file}",
        "--output-dir",
        str(env["output_dir"]),
        "--dry-run",
    ]

    destination = build_from_cli(args)
    expected = env["output_dir"].resolve() / "core-oem-0.0.0-dry-run-linux.tar.gz"
    assert destination == expected
    assert not env["output_dir"].exists()
    assert not destination.exists()


def test_build_from_cli_dry_run_without_additional_arguments(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    destination = build_from_cli(["--dry-run", "--platform", "linux"])

    expected = (tmp_path / "var" / "dist" / "core-oem-0.0.0-dry-run-linux.tar.gz").resolve()
    assert destination == expected
    assert not destination.exists()
    assert not destination.parent.exists()


def test_build_from_cli_dry_run_without_samples(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(build_core_bundle_module, "_DRY_RUN_PLACEHOLDER_ASSETS", None)
    monkeypatch.setattr(
        build_core_bundle_module,
        "_DRY_RUN_SAMPLE_DAEMON",
        tmp_path / "missing" / "daemon",
    )
    monkeypatch.setattr(
        build_core_bundle_module,
        "_DRY_RUN_SAMPLE_UI",
        tmp_path / "missing" / "ui",
    )
    monkeypatch.setattr(
        build_core_bundle_module,
        "_DRY_RUN_SAMPLE_CONFIG",
        tmp_path / "missing" / "config" / "core.yaml",
    )
    monkeypatch.setattr(
        build_core_bundle_module,
        "_DRY_RUN_SAMPLE_RESOURCE_DIR",
        tmp_path / "missing" / "extras",
    )
    monkeypatch.setattr(
        build_core_bundle_module,
        "_DRY_RUN_SAMPLE_SIGNING_KEY",
        tmp_path / "missing" / "signing.key",
    )

    destination = build_from_cli(["--dry-run", "--platform", "linux"])

    expected = (tmp_path / "var" / "dist" / "core-oem-0.0.0-dry-run-linux.tar.gz").resolve()
    assert destination == expected
    assert not destination.exists()
    assert not destination.parent.exists()


def test_build_from_cli_dry_run_detects_existing_bundle(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--dry-run",
    ]

    env["output_dir"].mkdir()
    existing = env["output_dir"].resolve() / "core-oem-1.0.0-linux.tar.gz"
    existing.write_text("placeholder", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        build_from_cli(args)

    assert existing.exists()


def test_build_from_cli_rejects_symlink_signing_key(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    real_key = _write_signing_key(tmp_path / "keys" / "real.key")
    symlink_path = tmp_path / "keys" / "link.key"
    if not hasattr(os, "symlink"):
        pytest.skip("OS does not support symlinks")
    try:
        if sys.platform.startswith("win"):
            os.symlink(str(real_key), str(symlink_path))
        else:
            os.symlink(real_key, symlink_path)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env)
    args[args.index("--signing-key-path") + 1] = str(symlink_path)
    args += ["--config", f"core.yaml={config_file}"]

    with pytest.raises(ValueError, match="Signing key path must not be a symlink"):
        build_from_cli(args)


def test_build_from_cli_rejects_directory_signing_key(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    key_dir = tmp_path / "keys" / "dir"
    key_dir.mkdir(parents=True)

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env)
    args[args.index("--signing-key-path") + 1] = str(key_dir)
    args += ["--config", f"core.yaml={config_file}"]

    with pytest.raises(ValueError, match="must reference a file"):
        build_from_cli(args)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission model not available")
def test_build_from_cli_rejects_insecure_signing_key_permissions(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    insecure_key = _write_signing_key(tmp_path / "keys" / "insecure.key")
    insecure_key.chmod(0o644)

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env)
    args[args.index("--signing-key-path") + 1] = str(insecure_key)
    args += ["--config", f"core.yaml={config_file}"]

    with pytest.raises(ValueError, match="file permissions must restrict access"):
        build_from_cli(args)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="OS does not support symlinks")
def test_build_from_cli_rejects_symlink_output_dir(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: aggressive", encoding="utf-8")

    real_dir = tmp_path / "real_out"
    real_dir.mkdir()
    symlink_dir = tmp_path / "dist_link"
    if sys.platform.startswith("win"):
        os.symlink(real_dir, symlink_dir, target_is_directory=True)
    else:
        os.symlink(real_dir, symlink_dir)

    args = _base_cli_args(env)
    args[args.index("--output-dir") + 1] = str(symlink_dir)
    args += ["--config", f"core.yaml={config_file}"]

    with pytest.raises(ValueError, match="must not be a symlink"):
        build_from_cli(args)


def test_build_from_cli_rejects_file_output_dir(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: aggressive", encoding="utf-8")

    file_path = tmp_path / "not_a_dir"
    file_path.write_text("", encoding="utf-8")

    args = _base_cli_args(env)
    args[args.index("--output-dir") + 1] = str(file_path)
    args += ["--config", f"core.yaml={config_file}"]

    with pytest.raises(ValueError, match="is not a directory"):
        build_from_cli(args)


def test_core_bundle_builder_rejects_control_char_in_version(tmp_path):
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "svc").write_text("bin", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "app").write_text("ui", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: conservative", encoding="utf-8")

    with pytest.raises(ValueError, match="control character"):
        CoreBundleBuilder(
            platform="linux",
            version="1.0.0\n",
            signing_key=os.urandom(48),
            output_dir=tmp_path,
            inputs=BundleInputs(
                daemon_paths=[daemon_dir],
                ui_paths=[ui_dir],
                config_paths={"core.yaml": config_file},
            ),
        )


def test_core_bundle_builder_creates_missing_output_dir(tmp_path):
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "svc").write_text("bin", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "app").write_text("ui", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: conservative", encoding="utf-8")

    output_dir = tmp_path / "dist" / "nested"
    assert not output_dir.exists()

    builder = CoreBundleBuilder(
        platform="linux",
        version="1.0.0",
        signing_key=os.urandom(48),
        output_dir=output_dir,
        inputs=BundleInputs(
            daemon_paths=[daemon_dir],
            ui_paths=[ui_dir],
            config_paths={"core.yaml": config_file},
        ),
    )

    archive = builder.build()
    assert archive.parent == output_dir.resolve()
    assert output_dir.is_dir()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="OS does not support symlinks")
def test_core_bundle_builder_rejects_symlink_output_dir(tmp_path):
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "svc").write_text("bin", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "app").write_text("ui", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: conservative", encoding="utf-8")

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    symlink_dir = tmp_path / "link"
    if sys.platform.startswith("win"):
        os.symlink(real_dir, symlink_dir, target_is_directory=True)
    else:
        os.symlink(real_dir, symlink_dir)

    with pytest.raises(ValueError, match="must not be a symlink"):
        CoreBundleBuilder(
            platform="linux",
            version="1.0.0",
            signing_key=os.urandom(48),
            output_dir=symlink_dir,
            inputs=BundleInputs(
                daemon_paths=[daemon_dir],
                ui_paths=[ui_dir],
                config_paths={"core.yaml": config_file},
            ),
        )


def test_core_bundle_builder_rejects_file_output_dir(tmp_path):
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "svc").write_text("bin", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "app").write_text("ui", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: conservative", encoding="utf-8")

    output_file = tmp_path / "output"
    output_file.write_text("file", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        CoreBundleBuilder(
            platform="linux",
            version="1.0.0",
            signing_key=os.urandom(48),
            output_dir=output_file,
            inputs=BundleInputs(
                daemon_paths=[daemon_dir],
                ui_paths=[ui_dir],
                config_paths={"core.yaml": config_file},
            ),
        )


def test_core_bundle_builder_rejects_version_starting_with_non_alnum(tmp_path):
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "svc").write_text("bin", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "app").write_text("ui", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: conservative", encoding="utf-8")

    with pytest.raises(ValueError, match="start with an alphanumeric"):
        CoreBundleBuilder(
            platform="linux",
            version="-beta1",
            signing_key=os.urandom(48),
            output_dir=tmp_path,
            inputs=BundleInputs(
                daemon_paths=[daemon_dir],
                ui_paths=[ui_dir],
                config_paths={"core.yaml": config_file},
            ),
        )


def test_core_bundle_builder_rejects_version_ending_with_non_alnum(tmp_path):
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "svc").write_text("bin", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "app").write_text("ui", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: conservative", encoding="utf-8")

    with pytest.raises(ValueError, match="end with an alphanumeric"):
        CoreBundleBuilder(
            platform="linux",
            version="beta1-",
            signing_key=os.urandom(48),
            output_dir=tmp_path,
            inputs=BundleInputs(
                daemon_paths=[daemon_dir],
                ui_paths=[ui_dir],
                config_paths={"core.yaml": config_file},
            ),
        )


def test_core_bundle_builder_rejects_version_too_long(tmp_path):
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "svc").write_text("bin", encoding="utf-8")

    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "app").write_text("ui", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: conservative", encoding="utf-8")

    long_version = "v" + ("1234567890" * 7)
    assert len(long_version) > 64

    with pytest.raises(ValueError, match="at most 64 characters"):
        CoreBundleBuilder(
            platform="linux",
            version=long_version,
            signing_key=os.urandom(48),
            output_dir=tmp_path,
            inputs=BundleInputs(
                daemon_paths=[daemon_dir],
                ui_paths=[ui_dir],
                config_paths={"core.yaml": config_file},
            ),
        )


def test_build_from_cli_rejects_config_directory(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_dir = tmp_path / "config-dir"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env) + ["--config", f"core.yaml={config_dir}"]

    with pytest.raises(ValueError, match="must reference a file"):
        build_from_cli(args)


def test_build_from_cli_rejects_duplicate_config_entries(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_one = tmp_path / "core.yaml"
    config_one.write_text("risk: balanced", encoding="utf-8")
    config_two = tmp_path / "core-override.yaml"
    config_two.write_text("risk: conservative", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_one}",
        "--config",
        f"core.yaml={config_two}",
    ]

    with pytest.raises(ValueError):
        build_from_cli(args)


def test_build_from_cli_rejects_casefold_config_collisions(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_upper = tmp_path / "Core.yaml"
    config_upper.write_text("risk: balanced", encoding="utf-8")
    config_lower = tmp_path / "core.yaml"
    config_lower.write_text("risk: aggressive", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"Core.yaml={config_upper}",
        "--config",
        f"core.yaml={config_lower}",
    ]

    with pytest.raises(ValueError, match="case-insensitive filesystem"):
        build_from_cli(args)


def test_build_from_cli_rejects_reserved_config_name(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "fingerprint.expected.json"
    config_file.write_text("{}", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"fingerprint.expected.json={config_file}",
    ]

    with pytest.raises(ValueError, match="reserved name"):
        build_from_cli(args)


def test_build_from_cli_rejects_config_name_ending_with_sig(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "config.sig"
    config_file.write_text("{}", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"config.sig={config_file}",
    ]

    with pytest.raises(ValueError, match=r"must not end with '.sig'"):
        build_from_cli(args)


def test_build_from_cli_rejects_config_signature_collision(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    primary = tmp_path / "primary.yaml"
    primary.write_text("risk: balanced", encoding="utf-8")
    colliding = tmp_path / "primary.yaml.sig"
    colliding.write_text("irrelevant", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"primary.yaml={primary}",
        "--config",
        f"primary.yaml.sig={colliding}",
    ]

    with pytest.raises(ValueError, match=r"must not end with '.sig'"):
        build_from_cli(args)


def test_build_from_cli_rejects_windows_reserved_config_name(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env) + ["--config", f"CON={config_file}"]

    with pytest.raises(ValueError, match="Windows reserved device name"):
        build_from_cli(args)


def test_build_from_cli_rejects_windows_reserved_config_component(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env) + ["--config", f"profiles/PRN.json={config_file}"]

    with pytest.raises(ValueError, match="Windows reserved device name"):
        build_from_cli(args)


def test_build_from_cli_rejects_config_name_with_trailing_dot(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env) + ["--config", f"core.={config_file}"]

    with pytest.raises(ValueError, match="ending with a space or dot"):
        build_from_cli(args)


def test_build_from_cli_rejects_config_name_with_invalid_windows_character(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env) + ["--config", f"core|secure.yaml={config_file}"]

    with pytest.raises(ValueError, match="disallowed on Windows"):
        build_from_cli(args)


def test_build_from_cli_rejects_config_name_with_control_character(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")
    control_char = "\x1f"

    args = _base_cli_args(env) + ["--config", f"core{control_char}.yaml={config_file}"]

    with pytest.raises(ValueError, match="control character"):
        build_from_cli(args)


def test_build_from_cli_rejects_config_file_with_control_character(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    bad_name = f"core\x1f.yaml"
    config_file = tmp_path / bad_name
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env) + ["--config", f"core.yaml={config_file}"]

    with pytest.raises(ValueError, match="control character"):
        build_from_cli(args)


def test_build_from_cli_rejects_nested_config_entries(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    root_file = tmp_path / "root.yaml"
    root_file.write_text("risk: balanced", encoding="utf-8")
    nested_file = tmp_path / "nested.yaml"
    nested_file.write_text("risk: aggressive", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"root.yaml={root_file}",
        "--config",
        f"root.yaml/nested.yaml={nested_file}",
    ]

    with pytest.raises(ValueError, match="nests within another entry"):
        build_from_cli(args)


def test_build_from_cli_rejects_config_becoming_parent(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    leaf_file = tmp_path / "leaf.yaml"
    leaf_file.write_text("risk: balanced", encoding="utf-8")
    parent_file = tmp_path / "parent.yaml"
    parent_file.write_text("risk: conservative", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"parent.yaml/leaf.yaml={leaf_file}",
        "--config",
        f"parent.yaml={parent_file}",
    ]

    with pytest.raises(ValueError, match="parent directory of another entry"):
        build_from_cli(args)


def test_build_from_cli_rejects_signature_directory_collision(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    primary = tmp_path / "primary.yaml"
    primary.write_text("risk: balanced", encoding="utf-8")
    nested = tmp_path / "nested.yaml"
    nested.write_text("risk: conservative", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"primary.yaml={primary}",
        "--config",
        f"primary.yaml.sig/nested.yaml={nested}",
    ]

    with pytest.raises(ValueError, match="signature file"):
        build_from_cli(args)


def test_build_from_cli_rejects_duplicate_daemon_entries(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--daemon",
        str(env["daemon_dir"]),
    ]

    with pytest.raises(ValueError):
        build_from_cli(args)


def test_build_from_cli_rejects_casefold_daemon_name_collision(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    alt_daemon = tmp_path / "Daemon"
    alt_daemon.mkdir()
    (alt_daemon / "botd-alt").write_text("daemon", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--daemon",
        str(alt_daemon),
    ]

    with pytest.raises(ValueError, match="case-insensitive filesystem"):
        build_from_cli(args)


def test_build_from_cli_rejects_casefold_daemon_directory_contents(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    colliding = tmp_path / "daemon-colliding"
    colliding.mkdir()
    (colliding / "Botd").write_text("primary", encoding="utf-8")
    (colliding / "botd").write_text("secondary", encoding="utf-8")

    args = _base_cli_args(env) + ["--config", f"core.yaml={config_file}"]
    args[args.index("--daemon") + 1] = str(colliding)

    with pytest.raises(ValueError, match="case-insensitive filesystem"):
        build_from_cli(args)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink not supported")
def test_build_from_cli_rejects_symlink_daemon(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    real_daemon = tmp_path / "botd"
    real_daemon.write_text("daemon", encoding="utf-8")
    link_path = tmp_path / "daemon-link"
    os.symlink(real_daemon, link_path)

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
    ]
    args[args.index("--daemon") + 1] = str(link_path)

    with pytest.raises(ValueError, match="Daemon artifact must not be a symlink"):
        build_from_cli(args)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink not supported")
def test_build_from_cli_rejects_symlink_inside_daemon_directory(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    real_binary = tmp_path / "botd"
    real_binary.write_text("daemon", encoding="utf-8")
    os.symlink(real_binary, env["daemon_dir"] / "botd-link")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
    ]

    with pytest.raises(ValueError, match="Daemon artifact contains forbidden symlink"):
        build_from_cli(args)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink not supported")
def test_build_from_cli_rejects_symlink_config(tmp_path):
    env = _create_basic_cli_environment(tmp_path)

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")
    config_link = tmp_path / "core-link.yaml"
    os.symlink(config_file, config_link)

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_link}",
    ]

    with pytest.raises(ValueError, match="Config entry 'core.yaml' must not be a symlink"):
        build_from_cli(args)


def test_build_from_cli_rejects_duplicate_ui_entries(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--ui",
        str(env["ui_dir"]),
    ]

    with pytest.raises(ValueError):
        build_from_cli(args)


def test_build_from_cli_rejects_windows_reserved_daemon_entry(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    reserved = tmp_path / "CON.txt"
    reserved.write_text("daemon", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--daemon",
        str(reserved),
    ]

    with pytest.raises(ValueError, match="Windows reserved device name"):
        build_from_cli(args)


def test_build_from_cli_rejects_casefold_ui_name_collision(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    alt_ui = tmp_path / "UI"
    alt_ui.mkdir()
    (alt_ui / "qtapp-alt").write_text("ui", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--ui",
        str(alt_ui),
    ]

    with pytest.raises(ValueError, match="case-insensitive filesystem"):
        build_from_cli(args)


def test_build_from_cli_rejects_casefold_ui_directory_contents(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: balanced", encoding="utf-8")

    colliding = tmp_path / "ui-colliding"
    colliding.mkdir()
    (colliding / "QtApp").write_text("primary", encoding="utf-8")
    (colliding / "qtapp").write_text("secondary", encoding="utf-8")

    args = _base_cli_args(env) + ["--config", f"core.yaml={config_file}"]
    args[args.index("--ui") + 1] = str(colliding)

    with pytest.raises(ValueError, match="case-insensitive filesystem"):
        build_from_cli(args)


def test_build_from_cli_rejects_invalid_resource_directory(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    resource_dir = tmp_path / "extras"
    resource_dir.mkdir()
    (resource_dir / "readme.txt").write_text("notes", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--resource",
        f"../extras={resource_dir}",
    ]

    with pytest.raises(ValueError):
        build_from_cli(args)


def test_build_from_cli_rejects_reserved_resource_directory(tmp_path):
    env = _create_basic_cli_environment(tmp_path)

    resource_dir = tmp_path / "extra_docs"
    resource_dir.mkdir()
    (resource_dir / "notes.txt").write_text("notes", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--resource",
        f"config/additional={resource_dir}",
    ]

    with pytest.raises(ValueError, match="reserved prefix"):
        build_from_cli(args)


def test_build_from_cli_rejects_reserved_resource_directory_casefold(tmp_path):
    env = _create_basic_cli_environment(tmp_path)

    resource_dir = tmp_path / "extra_docs"
    resource_dir.mkdir()
    (resource_dir / "notes.txt").write_text("notes", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--resource",
        f"CONFIG/additional={resource_dir}",
    ]

    with pytest.raises(ValueError, match="reserved prefix"):
        build_from_cli(args)


def test_build_from_cli_rejects_duplicate_resources(tmp_path):
    env = _create_basic_cli_environment(tmp_path)

    extras_one = tmp_path / "extras_one"
    extras_one.mkdir()
    (extras_one / "notes.txt").write_text("one", encoding="utf-8")

    extras_two = tmp_path / "extras_two"
    extras_two.mkdir()
    (extras_two / "notes.txt").write_text("two", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--resource",
        f"extras={extras_one / 'notes.txt'}",
        "--resource",
        f"extras={extras_two / 'notes.txt'}",
    ]

    with pytest.raises(ValueError):
        build_from_cli(args)


def test_build_from_cli_rejects_resource_directory_casefold_collision(tmp_path):
    env = _create_basic_cli_environment(tmp_path)

    extras_dir = tmp_path / "extras"
    extras_dir.mkdir()
    (extras_dir / "notes.txt").write_text("one", encoding="utf-8")

    extras_dir_alt = tmp_path / "Extras"
    extras_dir_alt.mkdir()
    (extras_dir_alt / "notes.txt").write_text("two", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--resource",
        f"extras={extras_dir}",
        "--resource",
        f"Extras={extras_dir_alt}",
    ]

    with pytest.raises(ValueError, match="case-insensitive filesystem"):
        build_from_cli(args)


def test_build_from_cli_rejects_resource_basename_casefold_collision(tmp_path):
    env = _create_basic_cli_environment(tmp_path)

    extras_dir = tmp_path / "extras"
    extras_dir.mkdir()

    first = extras_dir / "Readme.txt"
    first.write_text("one", encoding="utf-8")
    second = extras_dir / "README.TXT"
    second.write_text("two", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--resource",
        f"extras={first}",
        "--resource",
        f"extras={second}",
    ]

    with pytest.raises(ValueError, match="Duplicate resource entry"):
        build_from_cli(args)


def test_build_from_cli_rejects_windows_reserved_resource_file(tmp_path):
    env = _create_basic_cli_environment(tmp_path)

    extras_dir = tmp_path / "extras"
    extras_dir.mkdir()
    forbidden = extras_dir / "PRN.log"
    forbidden.write_text("log", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--resource",
        f"extras={forbidden}",
    ]

    with pytest.raises(ValueError, match="Windows reserved device name"):
        build_from_cli(args)


def test_build_from_cli_rejects_resource_directory_casefold_contents(tmp_path):
    env = _create_basic_cli_environment(tmp_path)

    extras_dir = tmp_path / "extras"
    extras_dir.mkdir()
    nested = extras_dir / "Readme"
    nested.mkdir()
    (nested / "Guide.txt").write_text("one", encoding="utf-8")
    (nested / "guide.TXT").write_text("two", encoding="utf-8")

    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--resource",
        f"extras={extras_dir}",
    ]

    with pytest.raises(ValueError, match="case-insensitive filesystem"):
        build_from_cli(args)

def test_build_from_cli_rejects_empty_fingerprint_placeholder(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--fingerprint-placeholder",
        "",
    ]

    with pytest.raises(ValueError, match="cannot be empty"):
        build_from_cli(args)


def test_build_from_cli_rejects_fingerprint_placeholder_with_space(tmp_path):
    env = _create_basic_cli_environment(tmp_path)
    config_file = tmp_path / "core.yaml"
    config_file.write_text("risk: manual", encoding="utf-8")

    args = _base_cli_args(env) + [
        "--config",
        f"core.yaml={config_file}",
        "--fingerprint-placeholder",
        "INVALID PLACEHOLDER",
    ]

    with pytest.raises(ValueError, match="unsupported character"):
        build_from_cli(args)

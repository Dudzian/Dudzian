import base64
import hashlib
import hmac
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.packaging import BundleInputs, CoreBundleBuilder
from bot_core.security.signing import canonical_json_bytes


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

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Callable

import pytest

from bot_core.resilience.bundle import ResilienceBundleBuilder
from bot_core.security.signing import canonical_json_bytes
from scripts import verify_resilience_bundle as vrb


@pytest.fixture()
def signing_key() -> bytes:
    return b"integration-test-secret"


def _make_tar_bundle(
    tmp_path: Path,
    key: bytes,
    *,
    payload_digest: str | None = None,
    manifest_override: Callable[[dict[str, object]], None] | None = None,
    signature_override: Callable[[dict[str, object]], None] | None = None,
    include_payload: bool = True,
) -> Path:
    bundle_path = tmp_path / "bundle.tar.gz"
    payload_name = "artifacts/runbook.txt"
    payload_content = b"restart scheduler"
    payload_sha = hashlib.sha256(payload_content).hexdigest()

    manifest = {
        "schema": "stage6.resilience.bundle.manifest",
        "files": [
            {
                "path": payload_name,
                "sha256": payload_sha,
                "size_bytes": len(payload_content),
            }
        ],
    }
    if manifest_override:
        manifest_override(manifest)
    manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    payload_section = {
        "path": "manifest.json",
        "sha256": payload_digest or manifest_digest,
    }
    signature_bytes = hmac.new(key, canonical_json_bytes(payload_section), hashlib.sha256).digest()
    signature_doc = {
        "signature": {
            "algorithm": "HMAC-SHA256",
            "value": base64.b64encode(signature_bytes).decode("ascii"),
        },
        "payload": payload_section,
    }
    if signature_override:
        signature_override(signature_doc)
    signature_json = json.dumps(signature_doc, ensure_ascii=False).encode("utf-8")

    with tarfile.open(bundle_path, "w:gz") as archive:
        for name, data in (
            ("manifest.json", manifest_bytes),
            ("manifest.json.sig", signature_json),
            *( ((payload_name, payload_content),) if include_payload else tuple() ),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    return bundle_path


def test_verify_tar_bundle_valid(tmp_path: Path, signing_key: bytes) -> None:
    bundle_path = _make_tar_bundle(tmp_path, signing_key)

    summary = vrb._verify_tar_bundle(bundle_path=bundle_path, signing_key=signing_key)

    assert summary["bundle"].endswith("bundle.tar.gz")
    assert summary["manifest"] == "manifest.json (embedded)"
    assert summary["verified_files"] == 1


def test_verify_tar_bundle_rejects_digest_mismatch(tmp_path: Path, signing_key: bytes) -> None:
    bundle_path = _make_tar_bundle(tmp_path, signing_key, payload_digest="deadbeef")

    with pytest.raises(ValueError, match="Digest manifestu nie zgadza się"):
        vrb._verify_tar_bundle(bundle_path=bundle_path, signing_key=signing_key)


def test_verify_tar_bundle_rejects_invalid_manifest_structure(tmp_path: Path, signing_key: bytes) -> None:
    malformed = _make_tar_bundle(
        tmp_path,
        signing_key,
        manifest_override=lambda manifest: manifest.__setitem__("files", None),
    )
    with pytest.raises(ValueError, match="Manifest nie zawiera listy"):
        vrb._verify_tar_bundle(bundle_path=malformed, signing_key=signing_key)

    not_mapping = _make_tar_bundle(
        tmp_path,
        signing_key,
        manifest_override=lambda manifest: manifest.__setitem__("files", ["not-mapping"]),
    )
    with pytest.raises(ValueError, match="Pozycja manifestu musi być mapą"):
        vrb._verify_tar_bundle(bundle_path=not_mapping, signing_key=signing_key)

    invalid_fields = _make_tar_bundle(
        tmp_path,
        signing_key,
        manifest_override=lambda manifest: manifest.__setitem__("files", [{"path": 1, "sha256": None}]),
    )
    with pytest.raises(ValueError, match="Pozycja wymaga pól 'path' i 'sha256'"):
        vrb._verify_tar_bundle(bundle_path=invalid_fields, signing_key=signing_key)

    missing_artifact = _make_tar_bundle(tmp_path, signing_key, include_payload=False)
    with pytest.raises(ValueError, match="Brak artefaktu w paczce"):
        vrb._verify_tar_bundle(bundle_path=missing_artifact, signing_key=signing_key)

    wrong_digest = _make_tar_bundle(
        tmp_path,
        signing_key,
        manifest_override=lambda manifest: manifest["files"][0].__setitem__("sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match="Niezgodny SHA-256"):
        vrb._verify_tar_bundle(bundle_path=wrong_digest, signing_key=signing_key)


def test_normalise_name_trims_prefix() -> None:
    assert vrb._normalise_name("./path/to/file") == "path/to/file"
    assert vrb._normalise_name("nested/file") == "nested/file"


def test_verify_external_manifest_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "docs").mkdir(parents=True)
    (source / "docs" / "runbook.md").write_text("restart service", encoding="utf-8")

    builder = ResilienceBundleBuilder(source, include=("**",))
    artifacts = builder.build(
        bundle_name="stage6",
        output_dir=tmp_path / "artifacts",
        signing_key=b"signature-secret-key",
        signing_key_id="stage6",
    )

    summary = vrb._verify_external_manifest(
        bundle_path=artifacts.bundle_path,
        manifest_path=artifacts.manifest_path,
        signature_path=artifacts.signature_path,
        signing_key=b"signature-secret-key",
    )

    assert summary["verified_files"] == 1
    assert summary["bundle"].endswith(".zip")


def test_verify_external_manifest_warnings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "bundle"
    (source / "docs").mkdir(parents=True)
    (source / "docs" / "guide.md").write_text("guide", encoding="utf-8")

    builder = ResilienceBundleBuilder(source, include=("**",))
    with_signature = builder.build(
        bundle_name="stage6",
        output_dir=tmp_path / "signed",
        signing_key=b"stage6-signature-key",
    )

    vrb._verify_external_manifest(
        bundle_path=with_signature.bundle_path,
        manifest_path=with_signature.manifest_path,
        signature_path=with_signature.signature_path,
        signing_key=None,
    )
    warn_without_key = capsys.readouterr().err
    assert "podpis bez klucza" in warn_without_key

    without_signature = builder.build(
        bundle_name="stage6-nosig",
        output_dir=tmp_path / "unsigned",
    )

    vrb._verify_external_manifest(
        bundle_path=without_signature.bundle_path,
        manifest_path=without_signature.manifest_path,
        signature_path=without_signature.signature_path,
        signing_key=b"stage6-signature-key",
    )
    warn_without_signature = capsys.readouterr().err
    assert "klucz HMAC" in warn_without_signature


def test_verify_external_manifest_propagates_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "bundle"
    (source / "docs").mkdir(parents=True)
    (source / "docs" / "guide.md").write_text("guide", encoding="utf-8")

    builder = ResilienceBundleBuilder(source, include=("**",))
    artifacts = builder.build(
        bundle_name="stage6",
        output_dir=tmp_path / "errors",
    )

    class DummyVerifier(vrb.ResilienceBundleVerifier):
        def verify_files(self) -> list[str]:
            return ["problem"]

    monkeypatch.setattr(vrb, "ResilienceBundleVerifier", DummyVerifier)

    with pytest.raises(ValueError, match="Walidacja zewnętrznego manifestu"):
        vrb._verify_external_manifest(
            bundle_path=artifacts.bundle_path,
            manifest_path=artifacts.manifest_path,
            signature_path=artifacts.signature_path,
            signing_key=None,
        )


def test_assert_safe_member_detects_risky_entries() -> None:
    directory = tarfile.TarInfo("dir/")
    assert vrb._assert_safe_member(directory) is None

    symlink = tarfile.TarInfo("link")
    symlink.type = tarfile.SYMTYPE
    with pytest.raises(ValueError, match="linki"):
        vrb._assert_safe_member(symlink)

    unsafe = tarfile.TarInfo("../etc/passwd")
    with pytest.raises(ValueError, match="niebezpieczną ścieżkę"):
        vrb._assert_safe_member(unsafe)


def test_read_json_requires_mapping(tmp_path: Path) -> None:
    archive_path = tmp_path / "invalid.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = json.dumps([1, 2, 3]).encode("utf-8")
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember("manifest.json")
        with pytest.raises(ValueError, match="obiektu JSON"):
            vrb._read_json(archive, member)


def test_compute_digest_rejects_unknown_algorithm(tmp_path: Path) -> None:
    bundle_path = _make_tar_bundle(tmp_path, b"K" * 16)
    with tarfile.open(bundle_path, "r:gz") as archive:
        manifest_member = archive.getmember("manifest.json")
        with pytest.raises(ValueError, match="Nieobsługiwany algorytm digest"):
            vrb._compute_digest(archive, manifest_member, "sha999")


def test_verify_manifest_signature_embedded_validations(tmp_path: Path, signing_key: bytes) -> None:
    bundle_path = _make_tar_bundle(tmp_path, signing_key)
    with tarfile.open(bundle_path, "r:gz") as archive:
        manifest_member = archive.getmember("manifest.json")
        signature_member = archive.getmember("manifest.json.sig")
        signature_doc = vrb._read_json(archive, signature_member)

        bad_path = dict(signature_doc)
        bad_path["payload"] = dict(signature_doc["payload"])
        bad_path["payload"]["path"] = "other.json"
        with pytest.raises(ValueError, match="nieoczekiwanej ścieżki"):
            vrb._verify_manifest_signature_embedded(
                archive=archive,
                manifest_member=manifest_member,
                signature_doc=bad_path,
                signing_key=signing_key,
            )

        multiple_digest = dict(signature_doc)
        multiple_digest["payload"] = dict(signature_doc["payload"])
        multiple_digest["payload"]["sha512"] = "abc"
        with pytest.raises(ValueError, match="dokładnie jeden wpis digest"):
            vrb._verify_manifest_signature_embedded(
                archive=archive,
                manifest_member=manifest_member,
                signature_doc=multiple_digest,
                signing_key=signing_key,
            )

        not_string = dict(signature_doc)
        not_string["payload"] = {"path": "manifest.json", "sha256": 123}
        with pytest.raises(ValueError, match="łańcuchem heksadecymalnym"):
            vrb._verify_manifest_signature_embedded(
                archive=archive,
                manifest_member=manifest_member,
                signature_doc=not_string,
                signing_key=signing_key,
            )

        bad_algorithm = dict(signature_doc)
        bad_algorithm["signature"] = dict(signature_doc["signature"])
        bad_algorithm["signature"]["algorithm"] = "MD5"
        with pytest.raises(ValueError, match="Nieobsługiwany algorytm podpisu"):
            vrb._verify_manifest_signature_embedded(
                archive=archive,
                manifest_member=manifest_member,
                signature_doc=bad_algorithm,
                signing_key=signing_key,
            )

        bad_base64 = dict(signature_doc)
        bad_base64["signature"] = dict(signature_doc["signature"])
        bad_base64["signature"]["value"] = "@@@"
        with pytest.raises(ValueError, match="Nieprawidłowe base64 podpisu"):
            vrb._verify_manifest_signature_embedded(
                archive=archive,
                manifest_member=manifest_member,
                signature_doc=bad_base64,
                signing_key=signing_key,
            )

        wrong_mac = dict(signature_doc)
        wrong_mac["signature"] = dict(signature_doc["signature"])
        wrong_mac["signature"]["value"] = base64.b64encode(b"wrong").decode("ascii")
        with pytest.raises(ValueError, match="Weryfikacja podpisu manifestu nie powiodła się"):
            vrb._verify_manifest_signature_embedded(
                archive=archive,
                manifest_member=manifest_member,
                signature_doc=wrong_mac,
                signing_key=signing_key,
            )


def test_load_signing_key_merged_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inline = vrb._load_signing_key_merged(
        inline_value="A" * 16,
        file_path=None,
        env_name_primary=None,
        env_name_alt=None,
    )
    assert inline == b"A" * 16

    key_file = tmp_path / "key.bin"
    key_file.write_bytes(b"B" * 32)
    os.chmod(key_file, 0o600)
    from_file = vrb._load_signing_key_merged(
        inline_value=None,
        file_path=str(key_file),
        env_name_primary=None,
        env_name_alt=None,
    )
    assert from_file == b"B" * 32

    monkeypatch.setenv("VRB_KEY", "C" * 24)
    from_env = vrb._load_signing_key_merged(
        inline_value=None,
        file_path=None,
        env_name_primary="VRB_KEY",
        env_name_alt="VRB_KEY_FALLBACK",
    )
    assert from_env == b"C" * 24


def test_load_signing_key_merged_validates_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="co najmniej 16"):
        vrb._load_signing_key_merged(
            inline_value="short",
            file_path=None,
            env_name_primary=None,
            env_name_alt=None,
        )

    key_file = tmp_path / "short.key"
    key_file.write_bytes(b"too-short")
    os.chmod(key_file, 0o600)
    with pytest.raises(ValueError, match="co najmniej 16"):
        vrb._load_signing_key_merged(
            inline_value=None,
            file_path=str(key_file),
            env_name_primary=None,
            env_name_alt=None,
        )

    monkeypatch.setenv("VRB_EMPTY", "")
    with pytest.raises(ValueError, match="jest pusta"):
        vrb._load_signing_key_merged(
            inline_value=None,
            file_path=None,
            env_name_primary="VRB_EMPTY",
            env_name_alt=None,
        )

    monkeypatch.setenv("VRB_SHORT", "short-key")
    with pytest.raises(ValueError, match="co najmniej 16"):
        vrb._load_signing_key_merged(
            inline_value=None,
            file_path=None,
            env_name_primary=None,
            env_name_alt="VRB_SHORT",
        )


def test_load_signing_key_merged_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.key"
    with pytest.raises(ValueError, match="nie istnieje"):
        vrb._load_signing_key_merged(
            inline_value=None,
            file_path=str(missing),
            env_name_primary=None,
            env_name_alt=None,
        )


def test_load_signing_key_merged_requires_strict_permissions(tmp_path: Path) -> None:
    key_file = tmp_path / "perms.key"
    key_file.write_bytes(b"D" * 32)
    # domyślnie chmod 0o666 -> zmieniamy na 0o644, co powinno być odrzucone
    os.chmod(key_file, 0o644)
    with pytest.raises(ValueError, match="uprawnienia maks. 600"):
        vrb._load_signing_key_merged(
            inline_value=None,
            file_path=str(key_file),
            env_name_primary=None,
            env_name_alt=None,
        )


def test_load_signing_key_merged_returns_none_without_sources() -> None:
    assert (
        vrb._load_signing_key_merged(
            inline_value=None,
            file_path=None,
            env_name_primary=None,
            env_name_alt=None,
        )
        is None
    )


def test_main_handles_tar_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str], signing_key: bytes) -> None:
    bundle_path = _make_tar_bundle(tmp_path, signing_key)

    exit_code = vrb.main([
        "--bundle",
        str(bundle_path),
        "--hmac-key",
        signing_key.decode("ascii", "ignore") if signing_key.isascii() else "stage6-signature-key",
        "--log-level",
        "DEBUG",
    ])

    assert exit_code == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["verified_files"] == 1


def test_main_handles_external_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "src"
    (source / "runbooks").mkdir(parents=True)
    (source / "runbooks" / "plan.md").write_text("plan", encoding="utf-8")

    builder = ResilienceBundleBuilder(source, include=("**",))
    artifacts = builder.build(
        bundle_name="resilience",
        output_dir=tmp_path / "out",
        signing_key=b"stage6-signature-key",
    )

    exit_code = vrb.main([
        "--bundle",
        str(artifacts.bundle_path),
        "--manifest",
        str(artifacts.manifest_path),
        "--signature",
        str(artifacts.signature_path),
        "--hmac-key",
        "stage6-signature-key",
    ])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["bundle"].endswith(".zip")


def test_main_requires_key_for_tar_mode(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, signing_key: bytes
) -> None:
    bundle_path = _make_tar_bundle(tmp_path, signing_key)
    with caplog.at_level("ERROR", logger="verify_resilience_bundle"):
        exit_code = vrb.main(["--bundle", str(bundle_path)])

    assert exit_code == 2
    assert any("Tryb TAR wymaga" in message for message in caplog.messages)


def test_main_reports_key_loading_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, signing_key: bytes
) -> None:
    bundle_path = _make_tar_bundle(tmp_path, signing_key)
    missing_key = tmp_path / "missing.key"

    with caplog.at_level("ERROR", logger="verify_resilience_bundle"):
        exit_code = vrb.main(["--bundle", str(bundle_path), "--hmac-key-file", str(missing_key)])

    assert exit_code == 1
    assert any("Błąd odczytu klucza" in message for message in caplog.messages)


def test_main_handles_external_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "src"
    (source / "docs").mkdir(parents=True)
    (source / "docs" / "guide.md").write_text("guide", encoding="utf-8")

    builder = ResilienceBundleBuilder(source, include=("**",))
    artifacts = builder.build(bundle_name="stage6", output_dir=tmp_path / "out")

    def boom(**_: object) -> dict[str, object]:
        raise RuntimeError("explode")

    monkeypatch.setattr(vrb, "_verify_external_manifest", boom)

    with caplog.at_level("ERROR", logger="verify_resilience_bundle"):
        exit_code = vrb.main([
            "--bundle",
            str(artifacts.bundle_path),
            "--manifest",
            str(artifacts.manifest_path),
            "--signature",
            str(artifacts.signature_path),
        ])

    assert exit_code == 2
    assert any("Weryfikacja nie powiodła się" in message for message in caplog.messages)

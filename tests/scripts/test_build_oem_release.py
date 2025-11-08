from __future__ import annotations

import json

from pathlib import Path

from scripts import build_oem_release


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_build_oem_release_generates_release_manifest(tmp_path, capsys):
    config_path = tmp_path / "pipeline.json"
    manifest_path = tmp_path / "manifest.json"
    staging_root = tmp_path / "staging"
    archive_path = tmp_path / "installer.zip"
    release_dir = tmp_path / "releases"
    output_path = tmp_path / "summary.json"

    staging_root.mkdir()
    archive_path.write_text("dummy archive", encoding="utf-8")

    _write_json(config_path, {})
    _write_json(
        manifest_path,
        {
            "bundle": "core-bot",
            "version": "1.2.3",
            "platform": "linux-x86_64",
        },
    )

    build_oem_release.main(
        [
            "--pipeline-config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--staging-root",
            str(staging_root),
            "--archive",
            str(archive_path),
            "--release-dir",
            str(release_dir),
            "--output",
            str(output_path),
        ]
    )

    stdout = capsys.readouterr().out
    summary = json.loads(stdout)
    assert summary == json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["archive"] == str(archive_path)
    assert summary["update_packages"] == []

    manifest_files = list(release_dir.glob("*.json"))
    assert len(manifest_files) == 1
    release_payload = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert release_payload["bundle"] == "core-bot"
    assert release_payload["version"] == "1.2.3"
    assert release_payload["platform"] == "linux-x86_64"
    assert release_payload["report"] == {}

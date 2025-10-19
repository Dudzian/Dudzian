from __future__ import annotations

import importlib
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import hashlib

import pytest
import yaml

from deploy.packaging.build_pyinstaller_bundle import build_bundle

bundle_module = importlib.import_module("deploy.packaging.build_pyinstaller_bundle")


def _fake_pyinstaller(path: Path, binary_name: str) -> None:
    dist_dir = path / "pyinstaller" / "dist" / binary_name
    dist_dir.mkdir(parents=True, exist_ok=True)
    executable = dist_dir / binary_name
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")


def test_build_bundle_with_qt_dist(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    qt_dist = tmp_path / "qt"
    (qt_dist / "bin").mkdir(parents=True)
    (qt_dist / "bin" / "app").write_text("binary", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001, D401 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": qt_dist,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": ["demo", "paper"],
        "metadata": ["channel=stable", "features=[\"risk\", \"hedge\"]"],
        "metadata_file": None,
        "metadata_yaml": None,
    })()

    archive_path = build_bundle(args)
    assert archive_path.exists()

    manifest_path = archive_path.with_suffix("") / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = {entry["path"] for entry in manifest["artifacts"]}
    assert "daemon/bot_core_runtime" in artifact_paths
    assert any(path.startswith("ui/") for path in artifact_paths)
    assert manifest["metadata"]["channel"] == "stable"
    assert manifest["metadata"]["features"] == ["risk", "hedge"]
    assert manifest["allowed_profiles"] == ["demo", "paper"]


def test_build_bundle_rejects_invalid_metadata(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": ["broken"],
        "metadata_file": None,
        "metadata_yaml": None,
    })()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_loads_metadata_from_file(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({"channel": "stable", "build": 123}), encoding="utf-8")

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": ["flavor=enterprise"],
        "metadata_file": [str(metadata_file)],
        "metadata_yaml": None,
    })()

    archive_path = build_bundle(args)
    manifest_path = archive_path.with_suffix("") / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable", "build": 123, "flavor": "enterprise"}


def test_build_bundle_loads_metadata_from_yaml(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_yaml = tmp_path / "metadata.yaml"
    metadata_yaml.write_text(
        yaml.safe_dump(
            {
                "channel": "stable",
                "release": {"commit": "abc123"},
                "release.branch": "main",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": None,
        "metadata_file": None,
        "metadata_yaml": [str(metadata_yaml)],
    })()

    archive_path = build_bundle(args)
    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {
        "channel": "stable",
        "release": {"commit": "abc123", "branch": "main"},
    }


def test_build_bundle_loads_metadata_from_ini(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_ini = tmp_path / "metadata.ini"
    metadata_ini.write_text(
        "\n".join(
            [
                "channel = stable",
                "",
                "[release]",
                "commit = \"abc123\"",
                "build = 42",
                "",
                "[release.branch]",
                "name = main",
            ]
        ),
        encoding="utf-8",
    )

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": ["release.branch.region=eu"],
        "metadata_file": None,
        "metadata_ini": [str(metadata_ini)],
        "metadata_toml": None,
        "metadata_yaml": None,
    })()

    archive_path = build_bundle(args)
    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {
        "channel": "stable",
        "release": {"commit": "abc123", "build": 42, "branch": {"name": "main", "region": "eu"}},
    }


def test_build_bundle_loads_metadata_from_toml(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_toml = tmp_path / "metadata.toml"
    metadata_toml.write_text(
        """
        channel = "beta"

        [release]
        commit = "abc123"
        branch = "main"
        """.strip(),
        encoding="utf-8",
    )

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": ["release.build=42"],
        "metadata_file": None,
        "metadata_toml": [str(metadata_toml)],
        "metadata_yaml": None,
    })()

    archive_path = build_bundle(args)
    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {
        "channel": "beta",
        "release": {"commit": "abc123", "branch": "main", "build": 42},
    }


def test_build_bundle_allows_nested_metadata_keys(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(
        json.dumps({"release": {"commit": "abc123"}}, ensure_ascii=False), encoding="utf-8"
    )

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": ["release.branch=main"],
        "metadata_file": [str(metadata_file)],
        "metadata_yaml": None,
    })()

    archive_path = build_bundle(args)
    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {
        "release": {
            "commit": "abc123",
            "branch": "main",
        }
    }


def test_build_bundle_rejects_nested_conflicts(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(
        json.dumps({"release": "1.2.3"}, ensure_ascii=False), encoding="utf-8"
    )

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": ["release.commit=abc123"],
        "metadata_file": [str(metadata_file)],
        "metadata_yaml": None,
    })()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_invalid_metadata_file(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text("[]", encoding="utf-8")

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": None,
        "metadata_file": [str(metadata_file)],
        "metadata_yaml": None,
    })()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_invalid_metadata_ini(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_ini = tmp_path / "metadata.ini"
    metadata_ini.write_text("[broken\nvalue", encoding="utf-8")

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": None,
        "metadata_file": None,
        "metadata_ini": [str(metadata_ini)],
        "metadata_toml": None,
        "metadata_yaml": None,
    })()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_invalid_metadata_toml(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_toml = tmp_path / "metadata.toml"
    metadata_toml.write_text("channel = [", encoding="utf-8")

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": None,
        "metadata_file": None,
        "metadata_toml": [str(metadata_toml)],
        "metadata_yaml": None,
    })()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_invalid_metadata_yaml(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_yaml = tmp_path / "metadata.yaml"
    metadata_yaml.write_text("- item", encoding="utf-8")

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": None,
        "metadata_file": None,
        "metadata_yaml": [str(metadata_yaml)],
    })()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_toml_conflicts(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({"channel": "stable"}), encoding="utf-8")

    metadata_toml = tmp_path / "metadata.toml"
    metadata_toml.write_text('channel = "beta"', encoding="utf-8")

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": None,
        "metadata_file": [str(metadata_file)],
        "metadata_toml": [str(metadata_toml)],
        "metadata_yaml": None,
    })()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_yaml_conflicts(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({"channel": "stable"}), encoding="utf-8")

    metadata_yaml = tmp_path / "metadata.yaml"
    metadata_yaml.write_text("channel: beta", encoding="utf-8")

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": None,
        "metadata_file": [str(metadata_file)],
        "metadata_yaml": [str(metadata_yaml)],
    })()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_loads_metadata_from_environment(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("BUNDLE_META_release__commit", '"abc123"')
    monkeypatch.setenv("BUNDLE_META_channel", '"stable"')

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": ["release.build=42"],
        "metadata_file": None,
        "metadata_yaml": None,
        "metadata_env_prefix": ["BUNDLE_META_"],
    })()

    archive_path = build_bundle(args)
    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {
        "channel": "stable",
        "release": {"commit": "abc123", "build": 42},
    }


def test_build_bundle_rejects_empty_env_prefix(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": None,
        "metadata_file": None,
        "metadata_yaml": None,
        "metadata_env_prefix": [""],
    })()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_loads_metadata_from_dotenv(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    dotenv = tmp_path / "metadata.env"
    dotenv.write_text(
        """
        # komentarz
        channel=stable
        export release__commit="abc123"
        """.strip(),
        encoding="utf-8",
    )

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": ["release.build=42"],
        "metadata_file": None,
        "metadata_yaml": None,
        "metadata_dotenv": [str(dotenv)],
        "metadata_env_prefix": None,
    })()

    archive_path = build_bundle(args)
    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {
        "channel": "stable",
        "release": {"commit": "abc123", "build": 42},
    }


def test_build_bundle_rejects_invalid_dotenv_line(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    dotenv = tmp_path / "metadata.env"
    dotenv.write_text("BROKEN_LINE", encoding="utf-8")

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": None,
        "metadata_file": None,
        "metadata_yaml": None,
        "metadata_dotenv": [str(dotenv)],
        "metadata_env_prefix": None,
    })()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_dotenv_conflicts(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    dotenv = tmp_path / "metadata.env"
    dotenv.write_text("channel=stable", encoding="utf-8")

    args = type("Args", (), {
        "entrypoint": entrypoint,
        "qt_dist": None,
        "briefcase_project": None,
        "platform": "linux",
        "version": "1.0.0",
        "output_dir": tmp_path / "out",
        "workdir": workdir,
        "hidden_import": None,
        "runtime_name": None,
        "include": None,
        "signing_key": None,
        "signing_key_id": None,
        "allowed_profile": None,
        "metadata": None,
        "metadata_file": [str(tmp_path / "metadata.json")],
        "metadata_yaml": None,
        "metadata_dotenv": [str(dotenv)],
        "metadata_env_prefix": None,
    })()

    (tmp_path / "metadata.json").write_text(json.dumps({"channel": "beta"}), encoding="utf-8")

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_loads_metadata_from_url(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata = {
        "channel": "stable",
        "release": {"commit": "abc123"},
        "release.branch": "main",
    }

    received_auth: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: D401 - minimal handler
            received_auth.append(self.headers.get("Authorization"))
            body = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A003 - suppress noisy logs
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/metadata.json"

    try:
        args = type(
            "Args",
            (),
            {
                "entrypoint": entrypoint,
                "qt_dist": None,
                "briefcase_project": None,
                "platform": "linux",
                "version": "1.0.0",
                "output_dir": tmp_path / "out",
                "workdir": workdir,
                "hidden_import": None,
                "runtime_name": None,
                "include": None,
                "signing_key": None,
                "signing_key_id": None,
                "allowed_profile": None,
                "metadata": ["release.build=42"],
                "metadata_file": None,
                "metadata_yaml": None,
                "metadata_url": [url],
                "metadata_url_header": ["Authorization=Bearer token"],
                "metadata_url_timeout": None,
                "metadata_url_max_size": None,
                "metadata_url_allow_http": True,
                "metadata_url_allowed_host": None,
            },
        )()

        archive_path = build_bundle(args)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {
        "channel": "stable",
        "release": {"commit": "abc123", "branch": "main", "build": 42},
    }
    assert received_auth == ["Bearer token"]


def test_build_bundle_rejects_http_metadata_url_without_flag(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    called: list[str] = []

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - ensure no network call
        called.append(request.full_url)
        raise AssertionError("urlopen should not be called for HTTP metadata without flag")

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["http://example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)

    assert called == []


def test_build_bundle_enforces_metadata_url_allowed_hosts(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - should not be reached
        raise AssertionError("urlopen should not be called when host is blocked")

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://api.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": ["updates.example.com"],
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_accepts_allowed_metadata_url_hosts(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()

        def read(self, size=None):  # noqa: D401 - return entire payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    observed_urls: list[str] = []

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        observed_urls.append(request.full_url)
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": ["updates.example.com"],
        },
    )()

    archive_path = build_bundle(args)

    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable"}
    assert observed_urls == ["https://updates.example.com/meta.json"]


def test_build_bundle_configures_metadata_url_ssl_context(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("CA", encoding="utf-8")
    ca_dir = tmp_path / "ca_dir"
    ca_dir.mkdir()
    client_cert = tmp_path / "client.pem"
    client_cert.write_text("CERT", encoding="utf-8")
    client_key = tmp_path / "client.key"
    client_key.write_text("KEY", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    observed_contexts: list[object] = []
    observed_create_calls: list[tuple[str | None, str | None]] = []

    class FakeContext:
        def __init__(self) -> None:
            self.loaded: list[tuple[str, str | None]] = []

        def load_cert_chain(self, certfile, keyfile=None):  # noqa: ANN001 - compatibility shim
            self.loaded.append((certfile, keyfile))

    fake_context = FakeContext()

    def fake_create_default_context(*, cafile=None, capath=None):  # noqa: ANN001 - test helper
        observed_create_calls.append((cafile, capath))
        return fake_context

    monkeypatch.setattr(bundle_module.ssl, "create_default_context", fake_create_default_context)

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        observed_contexts.append(context)
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://secure.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_ca": str(ca_file),
            "metadata_url_capath": str(ca_dir),
            "metadata_url_client_cert": str(client_cert),
            "metadata_url_client_key": str(client_key),
        },
    )()

    archive_path = build_bundle(args)

    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable"}
    assert observed_create_calls == [(str(ca_file), str(ca_dir))]
    assert fake_context.loaded == [(str(client_cert), str(client_key))]
    assert observed_contexts == [fake_context]


def test_build_bundle_accepts_metadata_url_cert_fingerprint(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"
    expected_fingerprint = hashlib.sha256(cert_bytes).hexdigest()

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": [f"sha256:{expected_fingerprint}"],
            "metadata_url_cert_subject": None,
        },
    )()

    archive_path = build_bundle(args)

    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable"}


def test_build_bundle_rejects_metadata_url_cert_fingerprint_mismatch(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": ["sha256:" + "0" * 64],
            "metadata_url_cert_subject": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_parse_cert_subject_requirements_accepts_entries() -> None:
    result = bundle_module._parse_cert_subject_requirements(
        ["commonName=updates.example.com", "O=BotCore"],
        option="--metadata-url-cert-subject",
    )
    assert result == {
        "commonname": {"updates.example.com"},
        "o": {"BotCore"},
    }


def test_parse_cert_subject_requirements_rejects_invalid_format() -> None:
    with pytest.raises(SystemExit):
        bundle_module._parse_cert_subject_requirements(
            ["invalid"], option="--metadata-url-cert-subject"
        )


def test_parse_cert_issuer_requirements_accepts_entries() -> None:
    result = bundle_module._parse_cert_issuer_requirements(
        ["organizationName=Trusted CA", "commonName=Root"],
        option="--metadata-url-cert-issuer",
    )
    assert result == {
        "organizationname": {"Trusted CA"},
        "commonname": {"Root"},
    }


def test_parse_cert_issuer_requirements_rejects_invalid_format() -> None:
    with pytest.raises(SystemExit):
        bundle_module._parse_cert_issuer_requirements(
            ["invalid"], option="--metadata-url-cert-issuer"
        )


def test_parse_cert_san_requirements_accepts_entries() -> None:
    result = bundle_module._parse_cert_san_requirements(
        ["DNS=updates.example.com", "URI=https://updates.example.com/meta.json"],
        option="--metadata-url-cert-san",
    )
    assert result == {
        "dns": {"updates.example.com"},
        "uri": {"https://updates.example.com/meta.json"},
    }


def test_parse_cert_san_requirements_rejects_invalid_format() -> None:
    with pytest.raises(SystemExit):
        bundle_module._parse_cert_san_requirements(
            ["invalid"], option="--metadata-url-cert-san"
        )


def test_parse_cert_extended_key_usage_accepts_alias_and_oid() -> None:
    result = bundle_module._parse_cert_extended_key_usage(
        ["serverAuth", "1.3.6.1.5.5.7.3.2"],
        option="--metadata-url-cert-eku",
    )
    assert result == {"1.3.6.1.5.5.7.3.1", "1.3.6.1.5.5.7.3.2"}


def test_parse_cert_extended_key_usage_rejects_invalid_identifier() -> None:
    with pytest.raises(SystemExit):
        bundle_module._parse_cert_extended_key_usage(
            ["not-an-oid"], option="--metadata-url-cert-eku"
        )


def test_parse_cert_policy_requirements_accepts_alias_and_oid() -> None:
    result = bundle_module._parse_cert_policy_requirements(
        ["anyPolicy", "1.2.3.4.5"], option="--metadata-url-cert-policy"
    )
    assert result == {"2.5.29.32.0", "1.2.3.4.5"}


def test_parse_cert_policy_requirements_rejects_invalid_identifier() -> None:
    with pytest.raises(SystemExit):
        bundle_module._parse_cert_policy_requirements(
            ["invalid"], option="--metadata-url-cert-policy"
        )


def test_parse_cert_serial_requirements_accepts_multiple_formats() -> None:
    result = bundle_module._parse_cert_serial_requirements(
        ["0x0A", "01:23:AB", "1234", "000000FF"],
        option="--metadata-url-cert-serial",
    )
    assert result == {"a", "123ab", "4d2", "ff"}


def test_parse_cert_serial_requirements_rejects_invalid_input() -> None:
    with pytest.raises(SystemExit):
        bundle_module._parse_cert_serial_requirements(
            ["XYZ"], option="--metadata-url-cert-serial"
        )


def test_build_bundle_accepts_metadata_url_cert_subject(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_subject",
        lambda data: {"commonname": ["updates.example.com"], "o": ["BotCore"]},
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": ["commonName=updates.example.com", "O=BotCore"],
            "metadata_url_cert_san": None,
        },
    )()

    archive_path = build_bundle(args)

    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable"}


def test_build_bundle_accepts_metadata_url_cert_issuer(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_issuer",
        lambda data: {"organizationname": ["Trusted CA"], "commonname": ["Root"]},
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_issuer": [
                "organizationName=Trusted CA",
                "commonName=Root",
            ],
            "metadata_url_cert_san": None,
        },
    )()

    archive_path = build_bundle(args)

    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable"}


def test_build_bundle_rejects_metadata_url_cert_subject_missing_attribute(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_subject",
        lambda data: {"organizationname": ["BotCore"]},
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": ["commonName=updates.example.com"],
            "metadata_url_cert_san": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_metadata_url_cert_subject_mismatch(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_subject",
        lambda data: {"commonname": ["other.example.com"]},
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": ["commonName=updates.example.com"],
            "metadata_url_cert_san": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_metadata_url_cert_issuer_missing_attribute(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_issuer",
        lambda data: {"organizationname": ["Trusted CA"]},
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_issuer": ["commonName=Root"],
            "metadata_url_cert_san": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_metadata_url_cert_issuer_mismatch(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_issuer",
        lambda data: {"commonname": ["Other"]},
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_issuer": ["commonName=Root"],
            "metadata_url_cert_san": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_accepts_metadata_url_cert_san(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_subject_alternative_names",
        lambda data: {"dns": ["updates.example.com"], "uri": ["https://updates.example.com/meta.json"]},
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": [
                "DNS=updates.example.com",
                "URI=https://updates.example.com/meta.json",
            ],
        },
    )()

    archive_path = build_bundle(args)

    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable"}


def test_build_bundle_rejects_metadata_url_cert_san_missing_entry(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_subject_alternative_names",
        lambda data: {"uri": ["https://updates.example.com/meta.json"]},
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": [
                "DNS=updates.example.com",
            ],
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_metadata_url_cert_san_mismatch(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_subject_alternative_names",
        lambda data: {"dns": ["mirror.example.com"]},
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": [
                "DNS=updates.example.com",
            ],
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_accepts_metadata_url_cert_eku(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_extended_key_usage",
        lambda data: (True, {"1.3.6.1.5.5.7.3.1", "1.3.6.1.5.5.7.3.2"}),
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": None,
            "metadata_url_cert_eku": ["serverAuth", "1.3.6.1.5.5.7.3.2"],
        },
    )()

    archive_path = build_bundle(args)

    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable"}


def test_build_bundle_rejects_metadata_url_cert_eku_missing_extension(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_extended_key_usage",
        lambda data: (False, set()),
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": None,
            "metadata_url_cert_eku": ["serverAuth"],
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_metadata_url_cert_eku_mismatch(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_extended_key_usage",
        lambda data: (True, {"1.3.6.1.5.5.7.3.2"}),
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": None,
            "metadata_url_cert_eku": ["serverAuth"],
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_accepts_metadata_url_cert_policy(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_policies",
        lambda data: (True, {"2.5.29.32.0", "1.2.3.4.5"}),
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": None,
            "metadata_url_cert_eku": None,
            "metadata_url_cert_policy": ["anyPolicy", "1.2.3.4.5"],
        },
    )()

    archive_path = build_bundle(args)

    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable"}


def test_build_bundle_rejects_metadata_url_cert_policy_missing_extension(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_policies",
        lambda data: (False, set()),
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": None,
            "metadata_url_cert_eku": None,
            "metadata_url_cert_policy": ["anyPolicy"],
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_metadata_url_cert_policy_mismatch(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        bundle_module,
        "_extract_certificate_policies",
        lambda data: (True, {"1.3.6.1.5.5.7.3.1"}),
    )

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": None,
            "metadata_url_cert_eku": None,
            "metadata_url_cert_policy": ["anyPolicy"],
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_accepts_metadata_url_cert_serial(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(bundle_module, "_extract_certificate_serial_number", lambda data: "abcd")

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": None,
            "metadata_url_cert_eku": None,
            "metadata_url_cert_policy": None,
            "metadata_url_cert_serial": ["0xABCD"],
        },
    )()

    archive_path = build_bundle(args)

    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable"}


def test_build_bundle_rejects_metadata_url_cert_serial_mismatch(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cert_bytes = b"dummy-cert"

    class FakeSocket:
        def getpeercert(self, binary_form=False):  # noqa: ANN001 - urllib compatibility
            assert binary_form
            return cert_bytes

    class FakeRaw:
        def __init__(self) -> None:
            self._sock = FakeSocket()

    class FakeFP:
        def __init__(self) -> None:
            self.raw = FakeRaw()

    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.fp = FakeFP()

        def read(self, size=None):  # noqa: D401 - return metadata payload
            return json.dumps({"channel": "stable"}).encode("utf-8")

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network
        return FakeResponse()

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)
    monkeypatch.setattr(bundle_module, "_extract_certificate_serial_number", lambda data: "beef")

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": None,
            "metadata_url_cert_subject": None,
            "metadata_url_cert_san": None,
            "metadata_url_cert_eku": None,
            "metadata_url_cert_policy": None,
            "metadata_url_cert_serial": ["0xABCD"],
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_metadata_url_cert_fingerprint_algorithm(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - should not be used
        raise AssertionError("metadata download should not be attempted when fingerprint is invalid")

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://updates.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_cert_fingerprint": ["sha1:" + "0" * 40],
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_metadata_url_client_key_without_cert(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    client_key = tmp_path / "client.key"
    client_key.write_text("KEY", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - should not be used
        raise AssertionError("metadata download should not be attempted when client cert is invalid")

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://secure.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_ca": None,
            "metadata_url_capath": None,
            "metadata_url_client_cert": None,
            "metadata_url_client_key": str(client_key),
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_missing_metadata_url_ca(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://secure.example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": None,
            "metadata_url_allow_http": False,
            "metadata_url_allowed_host": None,
            "metadata_url_ca": str(tmp_path / "missing_ca.pem"),
            "metadata_url_capath": None,
            "metadata_url_client_cert": None,
            "metadata_url_client_key": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_invalid_metadata_url_payload(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: D401 - minimal handler
            body = json.dumps(["not", "a", "mapping"]).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A003 - suppress noisy logs
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/metadata.json"

    try:
        args = type(
            "Args",
            (),
            {
                "entrypoint": entrypoint,
                "qt_dist": None,
                "briefcase_project": None,
                "platform": "linux",
                "version": "1.0.0",
                "output_dir": tmp_path / "out",
                "workdir": workdir,
                "hidden_import": None,
                "runtime_name": None,
                "include": None,
                "signing_key": None,
                "signing_key_id": None,
                "allowed_profile": None,
                "metadata": None,
                "metadata_file": None,
                "metadata_yaml": None,
                "metadata_url": [url],
                "metadata_url_header": None,
                "metadata_url_timeout": None,
                "metadata_url_max_size": None,
                "metadata_url_allow_http": True,
                "metadata_url_allowed_host": None,
            },
        )()

        with pytest.raises(SystemExit):
            build_bundle(args)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_build_bundle_rejects_unreachable_metadata_url(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def failing_urlopen(request, timeout=None):  # noqa: ANN001 - helper raising expected error
        raise bundle_module.urlerror.URLError("boom")

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", failing_urlopen)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_invalid_metadata_url_header(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": None,
            "metadata_url_header": ["InvalidHeader"],
            "metadata_url_timeout": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_duplicate_metadata_url_header(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://example.com/meta.json"],
            "metadata_url_header": ["Authorization=foo", "authorization=bar"],
            "metadata_url_timeout": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_nonpositive_metadata_url_timeout(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": 0.0,
            "metadata_url_max_size": None,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)


def test_build_bundle_rejects_non_json_metadata_url_content_type(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: D401 - minimal handler
            body = json.dumps({"channel": "stable"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A003 - suppress noisy logs
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/metadata.json"

    try:
        args = type(
            "Args",
            (),
            {
                "entrypoint": entrypoint,
                "qt_dist": None,
                "briefcase_project": None,
                "platform": "linux",
                "version": "1.0.0",
                "output_dir": tmp_path / "out",
                "workdir": workdir,
                "hidden_import": None,
                "runtime_name": None,
                "include": None,
                "signing_key": None,
                "signing_key_id": None,
                "allowed_profile": None,
                "metadata": None,
                "metadata_file": None,
                "metadata_yaml": None,
                "metadata_url": [url],
                "metadata_url_header": None,
                "metadata_url_timeout": None,
                "metadata_url_max_size": None,
                "metadata_url_allow_http": True,
                "metadata_url_allowed_host": None,
            },
        )()

        with pytest.raises(SystemExit):
            build_bundle(args)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_build_bundle_rejects_metadata_url_larger_than_limit_via_header(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: D401 - minimal handler
            body = json.dumps({"channel": "stable"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body) + 50))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A003 - suppress noisy logs
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/metadata.json"

    try:
        args = type(
            "Args",
            (),
            {
                "entrypoint": entrypoint,
                "qt_dist": None,
                "briefcase_project": None,
                "platform": "linux",
                "version": "1.0.0",
                "output_dir": tmp_path / "out",
                "workdir": workdir,
                "hidden_import": None,
                "runtime_name": None,
                "include": None,
                "signing_key": None,
                "signing_key_id": None,
                "allowed_profile": None,
                "metadata": None,
                "metadata_file": None,
                "metadata_yaml": None,
                "metadata_url": [url],
                "metadata_url_header": None,
                "metadata_url_timeout": None,
                "metadata_url_max_size": 64,
                "metadata_url_allow_http": True,
                "metadata_url_allowed_host": None,
            },
        )()

        with pytest.raises(SystemExit):
            build_bundle(args)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_build_bundle_rejects_metadata_url_larger_than_limit_without_header(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: D401 - minimal handler
            body = b"{" + b"x" * 200 + b"}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A003 - suppress noisy logs
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/metadata.json"

    try:
        args = type(
            "Args",
            (),
            {
                "entrypoint": entrypoint,
                "qt_dist": None,
                "briefcase_project": None,
                "platform": "linux",
                "version": "1.0.0",
                "output_dir": tmp_path / "out",
                "workdir": workdir,
                "hidden_import": None,
                "runtime_name": None,
                "include": None,
                "signing_key": None,
                "signing_key_id": None,
                "allowed_profile": None,
                "metadata": None,
                "metadata_file": None,
                "metadata_yaml": None,
                "metadata_url": [url],
                "metadata_url_header": None,
                "metadata_url_timeout": None,
                "metadata_url_max_size": 64,
                "metadata_url_allow_http": True,
                "metadata_url_allowed_host": None,
            },
        )()

        with pytest.raises(SystemExit):
            build_bundle(args)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_build_bundle_rejects_invalid_metadata_url_content_length_header(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: D401 - minimal handler
            body = json.dumps({"channel": "stable"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "invalid")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A003 - suppress noisy logs
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/metadata.json"

    try:
        args = type(
            "Args",
            (),
            {
                "entrypoint": entrypoint,
                "qt_dist": None,
                "briefcase_project": None,
                "platform": "linux",
                "version": "1.0.0",
                "output_dir": tmp_path / "out",
                "workdir": workdir,
                "hidden_import": None,
                "runtime_name": None,
                "include": None,
                "signing_key": None,
                "signing_key_id": None,
                "allowed_profile": None,
                "metadata": None,
                "metadata_file": None,
                "metadata_yaml": None,
                "metadata_url": [url],
                "metadata_url_header": None,
                "metadata_url_timeout": None,
                "metadata_url_max_size": 64,
                "metadata_url_allow_http": True,
                "metadata_url_allowed_host": None,
            },
        )()

        with pytest.raises(SystemExit):
            build_bundle(args)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_build_bundle_uses_metadata_url_timeout(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    observed_timeout: list[float | None] = []
    observed_header: list[str | None] = []
    responses: list[FakeResponse] = []

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = type(
                "Headers",
                (),
                {
                    "get_content_charset": lambda self: "utf-8",
                    "get": lambda self, name, default=None: None,
                },
            )()
            self.read_sizes: list[int | None] = []

        def read(self, size=None):  # noqa: D401 - symuluje odpowiedź HTTP
            self.read_sizes.append(size)
            payload = json.dumps({"channel": "stable"}).encode("utf-8")
            if size is None:
                return payload
            return payload[:size]

        def getcode(self):  # noqa: D401 - symuluje API urllib
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ANN001 - fake network call
        observed_timeout.append(timeout)
        observed_header.append(request.get_header("Authorization"))
        response = FakeResponse()
        responses.append(response)
        return response

    monkeypatch.setattr(bundle_module.urlrequest, "urlopen", fake_urlopen)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://example.com/meta.json"],
            "metadata_url_header": ["Authorization=Bearer token"],
            "metadata_url_timeout": 0.5,
            "metadata_url_max_size": None,
        },
    )()

    archive_path = build_bundle(args)
    manifest = json.loads((archive_path.with_suffix("") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"channel": "stable"}
    assert observed_timeout == [0.5]
    assert observed_header == ["Bearer token"]
    assert responses[0].read_sizes == [None]


def test_build_bundle_rejects_nonpositive_metadata_url_max_size(tmp_path, monkeypatch) -> None:
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("print('hello')", encoding="utf-8")

    workdir = tmp_path / "work"

    def fake_run(cmd, check, cwd=None, env=None):  # noqa: ANN001 - test helper
        if cmd and cmd[0] == "pyinstaller":
            _fake_pyinstaller(workdir, "bot_core_runtime")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {
            "entrypoint": entrypoint,
            "qt_dist": None,
            "briefcase_project": None,
            "platform": "linux",
            "version": "1.0.0",
            "output_dir": tmp_path / "out",
            "workdir": workdir,
            "hidden_import": None,
            "runtime_name": None,
            "include": None,
            "signing_key": None,
            "signing_key_id": None,
            "allowed_profile": None,
            "metadata": None,
            "metadata_file": None,
            "metadata_yaml": None,
            "metadata_url": ["https://example.com/meta.json"],
            "metadata_url_header": None,
            "metadata_url_timeout": None,
            "metadata_url_max_size": 0,
        },
    )()

    with pytest.raises(SystemExit):
        build_bundle(args)

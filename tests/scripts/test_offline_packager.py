from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.deploy import offline_packager


class DummyArgs(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.fixture(autouse=True)
def freeze_default_docs(monkeypatch, tmp_path):
    """Ensure DEFAULT_DOCS points to a deterministic location during tests."""
    docs_root = tmp_path / "default_docs"
    docs_root.mkdir()
    default_doc = docs_root / "README.txt"
    default_doc.write_text("default docs", encoding="utf-8")
    monkeypatch.setattr(offline_packager, "DEFAULT_DOCS", docs_root)
    return docs_root


def _create_sample_tree(root: Path, name: str, files: dict[str, str]) -> Path:
    base = root / name
    base.mkdir(parents=True)
    for relative, content in files.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return base


def test_build_offline_bundle_copies_resources(tmp_path, monkeypatch):
    output = tmp_path / "dist"
    ui_build = tmp_path / "ui_build"
    ui_build.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()

    config_dir = _create_sample_tree(tmp_path, "config_src", {"core.yaml": "config: value"})
    dataset_dir = _create_sample_tree(tmp_path, "dataset_src", {"data/sample.csv": "1,2,3"})
    docs_dir = _create_sample_tree(tmp_path, "docs_src", {"operator.md": "# Ops"})
    extra_dir = _create_sample_tree(tmp_path, "extra_dir", {"bin/tool.sh": "#!/bin/sh"})
    extra_file = tmp_path / "extra.txt"
    extra_file.write_text("log", encoding="utf-8")

    installer_archive = output / "installer" / "bot_shell.tar.gz"

    def fake_build_bundle(args: argparse.Namespace) -> Path:
        installer_archive.parent.mkdir(parents=True, exist_ok=True)
        installer_archive.write_text("installer", encoding="utf-8")
        return installer_archive

    monkeypatch.setattr(offline_packager, "build_bundle", fake_build_bundle)

    args = DummyArgs(
        ui_build=str(ui_build),
        output=str(output),
        config=str(config_dir),
        reports=str(reports),
        datasets=[str(dataset_dir)],
        docs=str(docs_dir),
        extra=[str(extra_dir), str(extra_file)],
        updater_script="scripts/updater.py",
        signing_key=None,
        platform="linux",
    )

    archive_path = offline_packager.build_offline_bundle(args)

    assert archive_path.exists()
    assert archive_path.name == "offline_bundle.tar.gz"

    offline_root = output / "bundle"
    manifest_path = offline_root / "MANIFEST.json"
    assert manifest_path.exists(), "Manifest should be written"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["installer"] == installer_archive.name
    assert manifest["datasets"] == ["datasets/dataset_src/data/sample.csv"]
    assert manifest["docs"] == ["docs/operator.md"]
    assert sorted(manifest["extras"]) == [
        "extras/extra.txt",
        "extras/extra_dir/bin/tool.sh",
    ]

    copied_config = offline_root / "config" / "core.yaml"
    assert copied_config.exists()
    assert copied_config.read_text(encoding="utf-8") == "config: value"

    copied_dataset = offline_root / "datasets" / "dataset_src" / "data" / "sample.csv"
    assert copied_dataset.exists()

    copied_extra_file = offline_root / "extras" / "extra.txt"
    assert copied_extra_file.exists()

    copied_extra_dir_file = offline_root / "extras" / "extra_dir" / "bin" / "tool.sh"
    assert copied_extra_dir_file.exists()


def test_build_offline_bundle_uses_default_docs(tmp_path, monkeypatch, freeze_default_docs):
    output = tmp_path / "dist"
    ui_build = tmp_path / "ui_build"
    ui_build.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()

    config_dir = _create_sample_tree(tmp_path, "config_src", {"core.yaml": "config: value"})
    dataset_dir = _create_sample_tree(tmp_path, "dataset_src", {"data/sample.csv": "1,2,3"})

    installer_archive = output / "installer" / "bot_shell.tar.gz"

    def fake_build_bundle(args: argparse.Namespace) -> Path:
        installer_archive.parent.mkdir(parents=True, exist_ok=True)
        installer_archive.write_text("installer", encoding="utf-8")
        return installer_archive

    monkeypatch.setattr(offline_packager, "build_bundle", fake_build_bundle)

    args = DummyArgs(
        ui_build=str(ui_build),
        output=str(output),
        config=str(config_dir),
        reports=str(reports),
        datasets=[str(dataset_dir)],
        docs=None,
        extra=[],
        updater_script="scripts/updater.py",
        signing_key=None,
        platform="linux",
    )

    archive_path = offline_packager.build_offline_bundle(args)
    assert archive_path.exists()

    manifest_path = output / "bundle" / "MANIFEST.json"
    docs_entries = json.loads(manifest_path.read_text(encoding="utf-8"))["docs"]
    assert docs_entries == ["docs/README.txt"], "Default docs should be copied when --docs is omitted"

    default_doc_copy = output / "bundle" / "docs" / "README.txt"
    assert default_doc_copy.exists()

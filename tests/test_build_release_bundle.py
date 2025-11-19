from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import build_release_bundle as brb


def _build_catalog(packages: list[dict[str, object]]) -> dict[str, object]:
    return {"schema_version": "1.1", "packages": packages}


def test_ensure_minimum_qa_reviews_passes() -> None:
    catalog = _build_catalog(
        [
            {
                "package_id": "alpha",
                "release": {
                    "review_status": "approved",
                    "reviewers": [
                        {"name": "QA Guild", "role": "quality"},
                        {"name": "Compliance", "role": "compliance"},
                    ],
                },
            },
            {
                "package_id": "beta",
                "release": {
                    "review_status": "qa_approved",
                    "reviewers": [
                        {"name": "Risk", "role": "risk"},
                        {"name": "QA Ops", "role": "QA"},
                    ],
                },
            },
        ]
    )

    approved = brb.ensure_minimum_qa_reviews(catalog, minimum=2)
    assert approved == ["alpha", "beta"]


def test_ensure_minimum_qa_reviews_raises() -> None:
    catalog = _build_catalog(
        [
            {
                "package_id": "gamma",
                "release": {
                    "review_status": "pending",
                    "reviewers": [{"name": "QA Guild", "role": "quality"}],
                },
            }
        ]
    )

    with pytest.raises(brb.ReleaseBundleError):
        brb.ensure_minimum_qa_reviews(catalog, minimum=1)


def test_copy_catalog_assets(tmp_path: Path) -> None:
    marketplace_root = tmp_path / "marketplace"
    packages_dir = marketplace_root / "packages"
    packages_dir.mkdir(parents=True)
    (packages_dir / "preset.json").write_text("{}", encoding="utf-8")

    catalog_path = marketplace_root / "catalog.json"
    catalog_path.write_text(json.dumps(_build_catalog([])), encoding="utf-8")
    catalog_sig = catalog_path.with_suffix(".json.sig")
    catalog_sig.write_text("signature", encoding="utf-8")

    markdown_path = marketplace_root / "catalog.md"
    markdown_path.write_text("# Catalog", encoding="utf-8")
    markdown_sig = markdown_path.with_suffix(".md.sig")
    markdown_sig.write_text("signature", encoding="utf-8")

    installer_root = tmp_path / "installer"
    outputs = brb.copy_catalog_assets(
        catalog_path=catalog_path,
        markdown_path=markdown_path,
        packages_dir=packages_dir,
        installer_root=installer_root,
    )

    assert (installer_root / "config/marketplace/catalog.json").exists()
    assert (installer_root / "config/marketplace/catalog.md").exists()
    assert (installer_root / "config/marketplace/packages/preset.json").exists()
    assert (installer_root / "config/marketplace/catalog.json").read_text(encoding="utf-8").strip().startswith("{")
    assert (installer_root / "config/marketplace/catalog.md").read_text(encoding="utf-8").strip().startswith("# Catalog")
    assert outputs


def test_ensure_git_clean_detects_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=repo, check=True)

    catalog_md = repo / "catalog.md"
    catalog_md.write_text("initial", encoding="utf-8")
    subprocess.run(["git", "add", "catalog.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)

    catalog_md.write_text("changed", encoding="utf-8")

    monkeypatch.setattr(brb, "REPO_ROOT", repo)
    with pytest.raises(brb.ReleaseBundleError):
        brb.ensure_git_clean([catalog_md])

from pathlib import Path
from types import SimpleNamespace

import json

from deployment.windows_stage9_evidence_contract import PRODUCTION_TRUST_CEREMONY_ID
from deployment import windows_stage9_trust_publisher as publisher


def test_operator_publisher_is_manual_self_hosted_and_keeps_receipt_separate() -> None:
    workflow = Path(".github/workflows/stage9-production-trust-publisher.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "self-hosted, Windows, X64, cryptohunter-production-authority" in workflow
    assert "stage9-public-production-trust-publication-evidence" in workflow
    package_upload = workflow.split("Upload canonical twelve-file public package", 1)[1]
    package_upload = package_upload.split("Upload public publication evidence separately", 1)[0]
    assert "stage9-publication-evidence.json" not in package_upload
    assert "*.json" in package_upload
    assert PRODUCTION_TRUST_CEREMONY_ID in workflow
    assert "windows_stage9_trust_publisher" in workflow


def test_producer_and_consumer_share_canonical_identity_and_no_regeneration() -> None:
    publisher = Path("deployment/windows_stage9_trust_publisher.py").read_text()
    consumer = Path("deployment/windows_stage9_clean_install.py").read_text()
    contract = Path("deployment/windows_stage9_evidence_contract.py").read_text()
    assert "PRODUCTION_TRUST_CEREMONY_ID" in contract
    assert "CEREMONY_ID" in publisher and "CEREMONY_ID" in consumer
    assert "PUBLIC_PACKAGE_FILENAMES" in publisher
    assert "load_production_trust(source)" in publisher
    assert "shutil.copyfile" in publisher
    assert "generate" not in publisher.lower()
    assert 'proofs["frozen_production_trust"] = "PASS"' in consumer


def test_publication_is_byte_exact_twelve_files_with_separate_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    final = tmp_path / "final"
    source = final / PRODUCTION_TRUST_CEREMONY_ID
    source.mkdir(parents=True)
    names = {"package_manifest.json", *(f"public-{number}.json" for number in range(11))}
    original = {}
    for number, name in enumerate(sorted(names)):
        original[name] = f'{{"value":{number}}}\n'.encode()
        (source / name).write_bytes(original[name])
    monkeypatch.setattr(publisher, "SOURCE_ROOT", final)
    monkeypatch.setattr(publisher, "PUBLIC_PACKAGE_FILENAMES", frozenset(names))
    monkeypatch.setattr(publisher, "validate_public_package_layout", lambda _path: None)
    monkeypatch.setattr(
        publisher,
        "load_production_trust",
        lambda _path: SimpleNamespace(ceremony_id=PRODUCTION_TRUST_CEREMONY_ID),
    )
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    output = tmp_path / "out"
    receipt = tmp_path / "publication.json"

    publisher.publish(source, output, receipt)

    assert {path.name for path in (output / PRODUCTION_TRUST_CEREMONY_ID).iterdir()} == names
    assert {name: (source / name).read_bytes() for name in names} == original
    assert {
        name: (output / PRODUCTION_TRUST_CEREMONY_ID / name).read_bytes() for name in names
    } == original
    assert receipt.parent != output / PRODUCTION_TRUST_CEREMONY_ID
    assert json.loads(receipt.read_text())["source_package_verification"] == "PASS"

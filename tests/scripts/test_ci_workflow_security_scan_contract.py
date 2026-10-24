from pathlib import Path


WORKFLOW_PATH = Path(".github/workflows/ci.yml")


def _dependency_scan_block() -> str:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    marker = "dependency-vulnerability-scan:"
    assert marker in text, "Brak joba 'dependency-vulnerability-scan' w .github/workflows/ci.yml"

    block = text.split(marker, 1)[1]
    next_job_marker = "\n  lint-and-test:"
    if next_job_marker in block:
        block = block.split(next_job_marker, 1)[0]
    return block


def test_dependency_vulnerability_scan_pip_audit_contract() -> None:
    block = _dependency_scan_block()

    assert "name: Dependency vulnerability scan" in block
    assert "python -m pip_audit" in block
    assert "--requirement deploy/packaging/requirements-desktop.lock" in block
    assert "--format json" in block
    assert "--output pip-audit-report.json" in block

    assert "--ignore-vuln" not in block
    assert "continue-on-error" not in block
    assert "--fix" not in block
    assert "if: always()" in block

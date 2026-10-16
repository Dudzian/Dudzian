"""Contracts for fail-closed artifact inventory and retention workflow."""

from pathlib import Path

import yaml


WORKFLOW = Path(".github/workflows/cleanup-artifacts.yml")


def test_inventory_retries_transient_server_errors_without_partial_deletes():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert document["permissions"]["actions"] == "write"
    step = document["jobs"]["delete-old-artifacts"]["steps"][0]
    assert step["uses"] == "actions/github-script@v9"
    script = step["with"]["script"]
    assert "for (let attempt = 1; attempt <= 3; attempt++)" in script
    assert "await github.paginate(" in script
    assert "status === 429 || (status >= 500 && status <= 599)" in script
    assert "if (!retryable || attempt === 3)" in script
    assert "throw error;" in script
    assert "if (!Array.isArray(allArtifacts))" in script
    assert "throw new Error('Artifact inventory never completed')" in script
    # All eligibility calculations and deletion happen after a complete listing.
    assert script.index("if (!Array.isArray(allArtifacts))") < script.index(
        "const deleteCandidates = []"
    )
    assert script.index("const deleteCandidates = []") < script.index(
        "await github.rest.actions.deleteArtifact("
    )


def test_retention_guards_remain_intact():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    script = document["jobs"]["delete-old-artifacts"]["steps"][0]["with"]["script"]
    for guard in (
        "retentionDays: 7",
        "keepRecentGlobal: 50",
        "keepPerName: 3",
        "minAgeHoursToDelete: 24",
        "if (updatedMs >= freshCutoffMs)",
        "if (globalKeepIds.has(artifact.id))",
        "if (perNameKeepIds.has(artifact.id))",
        "if (updatedMs >= retentionCutoffMs)",
    ):
        assert guard in script

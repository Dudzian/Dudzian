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
    assert "if (!retryable || status === 429)" in script
    assert "if (attempt === 3)" in script
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


def test_fallback_only_deletes_with_positive_proof_of_both_keeper_sets():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    script = document["jobs"]["delete-old-artifacts"]["steps"][0]["with"]["script"]

    assert "inventoryMode = 'verified-run-subset'" in script
    assert "const dayOffsets = [0, 1, 2, 7, 8]" in script
    assert "github.rest.actions.listWorkflowRunsForRepo" in script
    assert "github.rest.actions.listWorkflowRunArtifacts" in script
    assert "dayRuns.length !== first.data.total_count" in script
    assert "runs.length > 1200" in script
    assert "await Promise.all(Array.from({ length: 8 }, () => readRun()))" in script
    assert "if (inventoryMode === 'verified-run-subset')" in script
    assert "new Date(other.updated_at).getTime() > updatedMs" in script
    assert "newer.length < config.keepRecentGlobal" in script
    assert "newerSameName.length < config.keepPerName" in script
    assert "fallback: insufficient newer-artifact keeper proof" in script
    assert script.index("await Promise.all(") < script.index("const deleteCandidates = []")
    assert script.index("const deleteCandidates = []") < script.index(
        "await github.rest.actions.deleteArtifact("
    )


def test_rate_limit_and_incomplete_run_inventory_are_not_accepted_as_success():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    script = document["jobs"]["delete-old-artifacts"]["steps"][0]["with"]["script"]

    assert "if (!retryable || status === 429)" in script
    assert "throw new Error(`Incomplete workflow-run inventory for" in script
    assert "throw new Error('Run-level artifact scan exceeds safe API budget')" in script
    assert "throw new Error('Artifact inventory never completed')" in script

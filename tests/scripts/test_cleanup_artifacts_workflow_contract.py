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
    assert "if (status === 429)" in script
    assert "if (!retryable)" in script
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

    assert "if (status === 429)" in script
    assert "DEFERRED_NO_DELETION" in script
    assert "throw new Error(`Incomplete workflow-run inventory for" in script
    assert "throw new Error('Run-level artifact scan exceeds safe API budget')" in script
    assert "throw new Error('Artifact inventory never completed')" in script


def test_unavailable_fallback_defers_with_no_delete_and_keeps_hard_failures_visible():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    script = document["jobs"]["delete-old-artifacts"]["steps"][0]["with"]["script"]
    assert "if (status === 429)" in script
    assert "core.warning('Artifact cleanup deferred: repository inventory rate-limited" in script
    assert "const unavailable = status === 429 ||" in script
    assert "const safetyBoundary = " in script
    assert "if (!unavailable && !safetyBoundary)" in script
    assert "throw error;" in script  # Auth/authorization/programming failures stay fatal.
    assert "DEFERRED_NO_DELETION" in script
    assert script.index("DEFERRED_NO_DELETION") < script.index("const deleteCandidates = []")


def test_real_embedded_javascript_defers_before_any_deletion_on_api_outage():
    # Execute the exact github-script JS body with a mocked transport (no network
    # or GitHub mutation), not just a duplicated implementation of the policy.
    import json
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute github-script contract fixtures")
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    script = document["jobs"]["delete-old-artifacts"]["steps"][0]["with"]["script"]
    harness = """
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
const script = __SCRIPT__;
const scenario = process.argv[1];
const calls = [];
let deletes = 0;
const error = (code) => Object.assign(new Error('mock API unavailable'), {status:code});
const actions = {
  listArtifactsForRepo: async () => {calls.push('repo'); throw error(scenario === '429' ? 429 : 500);},
  listWorkflowRunsForRepo: async () => {calls.push('runs'); throw error(502);},
  listWorkflowRunArtifacts: async () => {calls.push('run-artifacts'); throw error(502);},
  deleteArtifact: async () => {deletes++;},
};
const github = {rest:{actions}, paginate: async (fn, args, callback) => callback(await fn(args))};
const log = [];
const core = {warning:(m) => log.push('WARNING: '+m), info:(m) => log.push('INFO: '+m)};
const context = {repo:{owner:'Dudzian',repo:'Dudzian'}};
(async () => {
  try {
    await new AsyncFunction('github','core','context',script)(github, core, context);
    process.stdout.write(JSON.stringify({ok:true, deletes, calls, log}));
  } catch(err) {
    process.stdout.write(JSON.stringify({ok:false, deletes, calls, reason:String(err)}));
    process.exitCode = 1;
  }
})();
""".replace("__SCRIPT__", json.dumps(script))
    for scenario, expected_repo_calls, expected_run_calls in [
        ("429", 1, 0),
        ("repo500-run502", 3, 1),
    ]:
        result = subprocess.run(
            [node, "-e", harness, scenario],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        data = json.loads(result.stdout)
        assert data["ok"] is True
        assert data["deletes"] == 0
        assert data["calls"].count("repo") == expected_repo_calls
        assert data["calls"].count("runs") == expected_run_calls
        assert any("deferred" in message.lower() for message in data["log"])

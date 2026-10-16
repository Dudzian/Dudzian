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
    assert "(response) => response.data.artifacts || []" not in script
    assert "response.data as an array" in script
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
    assert "const dayOffsets = [0, 1, 2, 7, 8, backlogOffset]" in script
    assert "const backlogOffset = 9 + ((utcDay * 2) % 82)" in script
    assert "const created = new Date(startMs).toISOString()" in script
    assert "github.rest.actions.listWorkflowRunsForRepo" in script
    assert "created, per_page: 100, page" in script
    assert "expectedTotal !== null && total !== expectedTotal" in script
    assert "createdMs < startMs || createdMs > endMs" in script
    assert "const uniqueArtifacts = new Map()" in script
    assert "github.rest.actions.listWorkflowRunsForRepo" in script
    assert "github.rest.actions.listWorkflowRunArtifacts" in script
    assert "if (count === total) break;" in script
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
    assert "throw new Error('Incomplete workflow-run inventory for '" in script
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
const deletedIds = [];
const error = (code) => Object.assign(new Error('mock API unavailable'), {status:code});
const timestamp = (ageDays) => new Date(Date.now() - ageDays * 86400000).toISOString();
const young = Array.from({length:50}, (_,i) => ({
  id:i+1, name:i<3 ? 'qualified-msi' : 'recent-'+i, updated_at:timestamp(2)
}));
const old = [
  {id:999, name:'qualified-msi', updated_at:timestamp(10)},
  {id:1000, name:'rare', updated_at:timestamp(10)},
];
const actions = {
  listArtifactsForRepo: async () => {
    calls.push('repo');
    if (scenario === 'healthy') return {data:{artifacts:[...young,...old]}};
    throw error(scenario === '429' ? 429 : scenario === '401' ? 401 : 500);
  },
  listWorkflowRunsForRepo: async () => {calls.push('runs'); throw error(502);},
  listWorkflowRunArtifacts: async () => {calls.push('run-artifacts'); throw error(502);},
  deleteArtifact: async (args) => {deletedIds.push(args.artifact_id);},
};
// Match Octokit: map callbacks receive normalized response.data arrays.
const github = {rest:{actions}, paginate: async (fn, args, callback) => {
  const response = await fn(args);
  const normalized = {
    ...response,
    data: Array.isArray(response.data) ? response.data :
      response.data.artifacts || response.data.workflow_runs || [],
  };
  return callback ? callback(normalized) : normalized.data;
}};
const log = [];
const core = {warning:(m) => log.push('WARNING: '+m), info:(m) => log.push('INFO: '+m)};
const context = {repo:{owner:'Dudzian',repo:'Dudzian'}};
(async () => {
  try {
    await new AsyncFunction('github','core','context',script)(github, core, context);
    process.stdout.write(JSON.stringify({ok:true, deletedIds, calls, log}));
  } catch(err) {
    process.stdout.write(JSON.stringify({ok:false, deletedIds, calls, reason:String(err)}));
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
        assert data["deletedIds"] == []
        assert data["calls"].count("repo") == expected_repo_calls
        assert data["calls"].count("runs") == expected_run_calls
        assert any("deferred" in message.lower() for message in data["log"])

    good = subprocess.run(
        [node, "-e", harness, "healthy"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert good.returncode == 0, good.stderr + good.stdout
    assert json.loads(good.stdout)["deletedIds"] == [999]  # Old protected name survives.
    denied = subprocess.run(
        [node, "-e", harness, "401"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert denied.returncode != 0
    assert json.loads(denied.stdout)["deletedIds"] == []


def test_real_embedded_fallback_paginates_frozen_days_and_deletes_proven_old_artifacts():
    """Exercise YAML JavaScript with mocked REST, including >100 runs/day."""

    import json
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the embedded cleanup script")
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    script = doc["jobs"]["delete-old-artifacts"]["steps"][0]["with"]["script"]
    harness = r"""
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
const source = __SCRIPT__;
const execute = new AsyncFunction('github','core','context','setTimeout',source);
const instantWait = (callback) => { callback(); return 0; };
const dayLength = 86400000;
const day = Math.floor(Date.now() / dayLength);
const backOffset = 9 + ((day * 2) % 82);
const atDay = (offset) => new Date((day-offset)*dayLength+3600000).toISOString();
const stamp = (age) => new Date(Date.now()-age*dayLength).toISOString();
const run = (id, offset) => ({id, created_at:atDay(offset)});
const artifact = (id,name,age) => ({id,name,updated_at:stamp(age)});
const recent = Array.from({length:106},(_,i)=>run(100+i,1));
const older = [run(777,8), run(778,backOffset)];
const runs = [...recent,...older];
const artifacts = new Map(recent.map((r,i)=>[
  r.id,[artifact(1000+i,i<4?'qualified':'recent-'+i,1)]
]));
artifacts.set(777,[artifact(999,'qualified',8),artifact(998,'rare',8)]);
artifacts.set(778,[artifact(997,'qualified',backOffset)]);
const testModes = [
  'healthy-fallback','unstable-pages','duplicate-id',
  'fewer-global','fewer-same-name','run-artifact-retry',
  'multiple-artifact-pages','empty-run-artifacts'
];
(async () => {
  const results = {};
  for (const scenario of testModes) {
    const calls = [];
    const deletedIds = [];
    const logs = [];
    let transient = 0;
    const failed = (status) => Object.assign(new Error('mock outage'),{status});
    const actions = {
      listArtifactsForRepo: async () => { calls.push('repo'); throw failed(500); },
      listWorkflowRunsForRepo: async (p) => {
        calls.push('runs');
        const [begin,end] = p.created.split('..').map(Date.parse);
        const selected = runs.filter(r=>{
          const time=Date.parse(r.created_at);
          return time>=begin&&time<=end;
        });
        const total = selected.length +
          ((scenario==='unstable-pages' && p.page===2) ? 1 : 0);
        return {
          data:{
            total_count:total,
            workflow_runs:selected.slice((p.page-1)*100,p.page*100)
          }
        };
      },
      listWorkflowRunArtifacts: async (p) => {
        calls.push('artifacts');
        if(scenario==='run-artifact-retry' && p.run_id===100 && transient++===0) {
          throw failed(502);
        }
        let available = artifacts.get(p.run_id)||[];
        if(scenario==='empty-run-artifacts') available = [];
        if(scenario==='duplicate-id' && p.run_id===101) {
          available = [artifact(1000,'incorrect-name',1)];
        }
        if(scenario==='fewer-global' && p.run_id>109 && p.run_id<206) {
          available = [];
        }
        if(scenario==='fewer-same-name' && p.run_id>=101 && p.run_id<=103) {
          available = [artifact(1000+p.run_id-100,'not-qualified-'+p.run_id,1)];
        }
        if(scenario==='multiple-artifact-pages' && p.run_id===100) {
          available = Array.from({length:102},(_,i)=>
            artifact(5000+i,'extra-'+i,1));
          available.push(artifact(1000,'qualified',1));
        }
        return {data:{total_count:available.length,artifacts:
          available.slice((p.page-1)*p.per_page,p.page*p.per_page)}};
      },
      deleteArtifact: async p => {deletedIds.push(p.artifact_id);},
    };
    const github = {
      rest:{actions},
      // Octokit normalizes wrapped REST response.data.artifacts to an
      // array BEFORE passing it to the optional mapper.
      paginate:async (method,params,callback) => {
        const records = [];
        for (let page=1; page<=30; page++) {
          const response = await method({...params,page});
          const array = Array.isArray(response.data) ? response.data :
            response.data.artifacts || response.data.workflow_runs || [];
          const normalized = {...response,data:array};
          const mapped = callback ? callback(normalized) : normalized.data;
          records.push(...mapped);
          if (array.length < params.per_page) return records;
        }
        throw new Error('Mock exceeded 30 pages');
      }
    };
    const core = {
      info:message=>logs.push(String(message)),
      warning:message=>logs.push('WARNING '+message)
    };
    let status = 'PASS';
    try {
      await execute(github,core,{repo:{owner:'Dudzian',repo:'Dudzian'}},instantWait);
    } catch (error) {status = 'FAIL '+String(error);}
    results[scenario] = {status,deletedIds,calls,logs};
  }
  process.stdout.write(JSON.stringify(results));
})().catch(error=>{console.error(error);process.exitCode=1;});
""".replace("__SCRIPT__", json.dumps(script))
    result = subprocess.run(
        [node, "-e", harness],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    cases = json.loads(result.stdout)

    healthy = cases["healthy-fallback"]
    assert healthy["status"] == "PASS"
    assert healthy["deletedIds"] == [999, 997]
    assert healthy["calls"].count("repo") == 3
    assert healthy["calls"].count("runs") >= 7
    assert healthy["calls"].count("artifacts") == 108
    assert any("Fallback window" in msg for msg in healthy["logs"])
    assert any("rotating 9-90 day window" in msg for msg in healthy["logs"])
    assert not any(msg.startswith("WARNING") for msg in healthy["logs"])

    for mode in (
        "unstable-pages",
        "duplicate-id",
        "empty-run-artifacts",
    ):
        case = cases[mode]
        assert case["status"] == "PASS"  # Explicitly deferred housekeeping.
        assert case["deletedIds"] == []
        assert any("DEFERRED_NO_DELETION" in msg for msg in case["logs"])

    for mode in ("fewer-global", "fewer-same-name"):
        case = cases[mode]
        assert case["status"] == "PASS"
        assert case["deletedIds"] == []
        assert not any("DEFERRED_NO_DELETION" in msg for msg in case["logs"])

    retried = cases["run-artifact-retry"]
    assert retried["status"] == "PASS"
    assert retried["deletedIds"] == [999, 997]
    assert retried["calls"].count("artifacts") == 109  # One transient 502.

    paginated = cases["multiple-artifact-pages"]
    assert paginated["status"] == "PASS"
    assert paginated["deletedIds"] == [999, 997]
    assert paginated["calls"].count("artifacts") == 109  # 102+1 items in one run.

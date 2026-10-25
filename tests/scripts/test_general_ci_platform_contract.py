"""Regression contracts for the cross-platform General CI security slice."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tomllib

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
EXTERNAL_POSTGRESQL_TESTS = (
    "tests/security/test_postgresql_entitlement_registry.py",
    "tests/security/test_postgresql_security_definer_principal_semantics.py",
    "tests/security/test_postgresql_freshness_authority_core.py",
    "tests/security/test_postgresql_history_successor_reproposal.py",
    "tests/security/test_history_successor_reproposal_legacy.py",
)

SERIAL_LICENSING_GLOBS = (
    "tests/licensing/test_lppi_*.py",
    "tests/licensing/test_pdsa_*.py",
    "tests/licensing/test_production_*.py",
    "tests/licensing/test_cha_*.py",
)

POSTGRESQL_ISSUER_KEYWORDS = (
    "root_proof_issuance_authority or root_proof_preparation_v2 or "
    "root_proof_retention or root_proof_signing_intent or "
    "root_proof_issuer_orchestration_integration"
)
POSTGRESQL_HISTORY_KEYWORDS = "history_successor_reproposal"
GENERAL_MARKERS = (
    "not external_postgresql and not integration and not e2e_demo_paper "
    "and not retraining and not soak and not performance and not qml"
)
POSTGRESQL_MARKERS = (
    "external_postgresql and not integration and not e2e_demo_paper "
    "and not retraining and not soak and not performance and not qml"
)


def _general_ci_jobs() -> dict:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return document["jobs"]


_PYTEST_STEP_NAMES = frozenset(
    {
        "Run fast pytest suite",
        "Run general fast pytest suite (excluding dedicated PostgreSQL qualification)",
        "Run external PostgreSQL fast pytest qualification (serial)",
    }
)


def _pytest_step(job: dict) -> dict:
    matching = [step for step in job["steps"] if step["name"] in _PYTEST_STEP_NAMES]
    assert len(matching) == 1, "exactly one qualified fast pytest step required"
    return matching[0]


def _pytest_command(job: dict) -> str:
    return _pytest_step(job)["run"]


def _git_bash_for_executable(git_executable: str) -> str:
    """Find Git Bash from the PATH entry, not git.exe's resolved MSYS target."""
    # Git for Windows exposes Git through bin/, cmd/ or ucrt64/bin/.
    # Resolving git.exe follows a shim into ucrt64/bin and breaks root discovery.
    for parent in Path(git_executable).parents:
        candidate = parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    raise AssertionError(f"Git Bash not found for Git executable: {git_executable}")


def _contract_bash() -> str:
    """Use Git Bash on Windows, not the WSL bash.exe launcher on PATH."""
    if sys.platform != "win32":
        return "bash"
    git_executable = shutil.which("git.exe")
    assert git_executable is not None, "Git for Windows is required for CI shell contracts"
    return _git_bash_for_executable(git_executable)


@pytest.mark.parametrize(
    "executable_relative", ["bin/git.exe", "cmd/git.exe", "ucrt64/bin/git.exe"]
)
def test_git_bash_resolves_installation_root_for_shim_locations(tmp_path, executable_relative):
    root = tmp_path / "Git"
    executable = root / executable_relative
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.touch()
    bash = root / "bin" / "bash.exe"
    bash.parent.mkdir(parents=True, exist_ok=True)
    bash.touch()
    assert _git_bash_for_executable(str(executable)) == str(bash)


def test_git_bash_discovery_fails_closed_when_missing(tmp_path):
    executable = tmp_path / "Git" / "cmd" / "git.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    with pytest.raises(AssertionError, match="Git Bash not found"):
        _git_bash_for_executable(str(executable))


def _postgresql_shard_arguments(job: dict, shard: str, directory: Path) -> list[str]:
    """Execute the real shell selection without starting pytest or a database."""
    directory.mkdir(parents=True)
    command = _pytest_command(job).replace("${{ matrix.shard }}", shard)
    recorder = 'python() { printf \'%s\\0\' "$#" "$@" >> pytest-argv; }\n'
    result = subprocess.run(
        [_contract_bash(), "-c", recorder + command],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    arguments = (directory / "pytest-argv").read_bytes().split(b"\0")
    assert arguments.pop() == b""
    count = int(arguments.pop(0))
    assert count == len(arguments), "exactly one pytest invocation required"
    return [argument.decode() for argument in arguments]


def _without_pytest_reporting(arguments: list[str]) -> list[str]:
    """Permit only one approved summary option at the pytest option boundary."""
    assert arguments[:2] == ["-m", "pytest"]
    if arguments[2:3] == ["-ra"]:
        return arguments[:2] + arguments[3:]
    return arguments


def _assert_fast_pytest_arguments(arguments: list[str], *, selector: str | None = None) -> None:
    expected = ["-m", "pytest", "-p", "asyncio", "--fast"]
    if selector is not None:
        expected.extend(["-k", selector])
    expected.extend(
        [
            "-m",
            POSTGRESQL_MARKERS if selector is not None else GENERAL_MARKERS,
            "--ignore=tests/e2e",
            "--ignore=tests/integration",
            "--ignore=tests/performance",
            "--ignore=tests/ui_pyside",
            "--ignore=tests/ui_backend",
            *(f"--ignore-glob={pattern}" for pattern in SERIAL_LICENSING_GLOBS),
            "--maxfail=1",
            "--durations=25" if selector is not None else "--durations=10",
            "--junitxml=test-results/pytest.xml",
        ]
    )
    assert _without_pytest_reporting(arguments) == expected


def _logical_pytest_arguments(command: str, executable: str = "python") -> list[str]:
    """Parse one continued invocation in steps with setup commands or heredocs."""
    prefix = f"{executable} -m pytest"
    lines = command.splitlines()
    starts = [index for index, line in enumerate(lines) if prefix in line]
    assert len(starts) == 1, "exactly one pytest invocation required"
    index = starts[0]
    invocation = lines[index].split(prefix, 1)[1]
    while invocation.rstrip().endswith("\\"):
        index += 1
        invocation = invocation.rstrip()[:-1] + " " + lines[index].strip()
    return ["-m", "pytest", *shlex.split(invocation)]


@pytest.mark.parametrize("reporting", ["", "-ra "])
@pytest.mark.parametrize(
    ("shard", "selector"),
    [
        (
            "issuer-durability",
            f"({POSTGRESQL_ISSUER_KEYWORDS}) and not {POSTGRESQL_HISTORY_KEYWORDS}",
        ),
        ("history-reproposal", POSTGRESQL_HISTORY_KEYWORDS),
        (
            "remaining-security",
            f"not ({POSTGRESQL_ISSUER_KEYWORDS} or {POSTGRESQL_HISTORY_KEYWORDS})",
        ),
    ],
)
def test_postgresql_shell_contract_accepts_only_approved_reporting(
    tmp_path, reporting, shard, selector
):
    job = _general_ci_jobs()["py-tests-ubuntu-postgresql"]
    step = _pytest_step(job)
    step["run"] = step["run"].replace("python -m pytest -ra ", f"python -m pytest {reporting}")
    arguments = _postgresql_shard_arguments(job, shard, tmp_path / "recorded")
    _assert_fast_pytest_arguments(arguments, selector=selector)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param("-ra", "-rA", id="unapproved-summary"),
        pytest.param("-ra", "-ra -ra", id="duplicate-summary"),
        pytest.param("-ra", "--collect-only", id="collection-without-execution"),
        pytest.param("-ra", "--disable-warnings", id="hidden-warnings"),
        pytest.param("-p asyncio", "-p no:asyncio", id="disabled-asyncio"),
        pytest.param("--fast", "", id="missing-fast"),
        pytest.param('-k "$SHARD_KEYWORDS"', "-k $SHARD_KEYWORDS", id="split-selector"),
        pytest.param(
            '-k "$SHARD_KEYWORDS"',
            '-k "history_successor_reproposal"',
            id="wrong-shard-selector",
        ),
        pytest.param(
            '-m "external_postgresql and',
            '-m "not external_postgresql and',
            id="wrong-database-marker",
        ),
        pytest.param("--durations=25", "--durations=25 -n 2", id="parallel-database-tests"),
        pytest.param(
            "--durations=25",
            "--durations=25 tests/security/test_one.py",
            id="restricted-test-path",
        ),
        pytest.param(
            "--durations=25",
            "--durations=25 --ignore=tests/security",
            id="missing-security-coverage",
        ),
        pytest.param(
            "--ignore-glob=tests/licensing/test_cha_*.py",
            "--ignore-glob=tests/licensing/test_cha_root_proof*.py",
            id="changed-licensing-partition",
        ),
        pytest.param("--maxfail=1", "--maxfail=2", id="changed-failure-policy"),
        pytest.param("--fast \\\n", "--fast\n", id="broken-shell-continuation"),
        pytest.param(
            "--junitxml=test-results/pytest.xml",
            "--junitxml=test-results/pytest.xml\npython -m pytest --collect-only",
            id="second-pytest-invocation",
        ),
    ],
)
def test_postgresql_shell_contract_rejects_selection_and_execution_drift(tmp_path, before, after):
    job = _general_ci_jobs()["py-tests-ubuntu-postgresql"]
    step = _pytest_step(job)
    assert step["run"].count(before) == 1
    step["run"] = step["run"].replace(before, after)
    with pytest.raises(AssertionError):
        arguments = _postgresql_shard_arguments(job, "issuer-durability", tmp_path / "recorded")
        _assert_fast_pytest_arguments(
            arguments,
            selector=f"({POSTGRESQL_ISSUER_KEYWORDS}) and not {POSTGRESQL_HISTORY_KEYWORDS}",
        )


def test_postgresql_shell_contract_rejects_unknown_shard(tmp_path):
    job = _general_ci_jobs()["py-tests-ubuntu-postgresql"]
    with pytest.raises(AssertionError, match="Unknown PostgreSQL qualification shard"):
        _postgresql_shard_arguments(job, "unknown", tmp_path / "recorded")


def test_external_postgresql_marker_is_registered_and_applied():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.startswith("external_postgresql:") for marker in markers)
    for relative_path in EXTERNAL_POSTGRESQL_TESTS:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "pytestmark = pytest.mark.external_postgresql" in source


def test_ubuntu_runs_external_postgresql_against_postgresql_16_service(tmp_path):
    jobs = _general_ci_jobs()
    general = jobs["py-tests-ubuntu"]
    postgres_job = jobs["py-tests-ubuntu-postgresql"]
    strategy = postgres_job["strategy"]
    assert strategy["fail-fast"] is False
    shards = ["issuer-durability", "history-reproposal", "remaining-security"]
    assert strategy["matrix"]["shard"] == shards
    assert general["runs-on"] == postgres_job["runs-on"] == "ubuntu-24.04"
    assert general["needs"] == postgres_job["needs"] == "prepare-wheelhouse"
    producer = jobs[general["needs"]]
    # Resolve Linux wheels on the consumers' glibc baseline, while keeping the
    # other platforms and their existing artifact matrix labels available.
    assert producer["runs-on"] == (
        "${{ matrix.os == 'ubuntu-latest' && '" + general["runs-on"] + "' || matrix.os }}"
    )
    assert producer["strategy"]["matrix"]["os"] == [
        "ubuntu-latest",
        "windows-latest",
        "macos-latest",
    ]

    # The previous Ubuntu selection must be partitioned exhaustively, not
    # silently dropped or redundantly run on a shared service.
    common_markers = (
        "not integration and not e2e_demo_paper and not retraining "
        "and not soak and not performance and not qml"
    )
    general_command = _pytest_command(general)
    postgres_command = _pytest_command(postgres_job)
    assert f'-m "not external_postgresql and {common_markers}"' in general_command
    assert f'-m "external_postgresql and {common_markers}"' in postgres_command
    # History has priority over issuer keywords; the third shard is their complement.
    assert 'case "${{ matrix.shard }}" in' in postgres_command
    assert "issuer-durability)" in postgres_command
    assert "history-reproposal)" in postgres_command
    assert "remaining-security)" in postgres_command
    assert f"ISSUER_KEYWORDS='{POSTGRESQL_ISSUER_KEYWORDS}'" in postgres_command
    assert f"HISTORY_KEYWORDS='{POSTGRESQL_HISTORY_KEYWORDS}'" in postgres_command
    assert '-k "$SHARD_KEYWORDS"' in postgres_command
    expected_selectors = {
        "issuer-durability": (
            f"({POSTGRESQL_ISSUER_KEYWORDS}) and not {POSTGRESQL_HISTORY_KEYWORDS}"
        ),
        "history-reproposal": POSTGRESQL_HISTORY_KEYWORDS,
        "remaining-security": (
            f"not ({POSTGRESQL_ISSUER_KEYWORDS} or {POSTGRESQL_HISTORY_KEYWORDS})"
        ),
    }
    for shard in shards:
        arguments = _postgresql_shard_arguments(postgres_job, shard, tmp_path / shard)
        # A doubled shell continuation would introduce a literal path or try to
        # execute --fast as a command; inspecting YAML text alone cannot catch it.
        _assert_fast_pytest_arguments(arguments, selector=expected_selectors[shard])
    _assert_fast_pytest_arguments(
        _postgresql_shard_arguments(general, "unused", tmp_path / "general")
    )
    for command in (general_command, postgres_command):
        assert "--fast" in command
        assert "--maxfail=1" in command
    assert {
        token
        for token in general_command.split()
        if token.startswith(("--ignore=", "--ignore-glob="))
    } == {
        token
        for token in postgres_command.split()
        if token.startswith(("--ignore=", "--ignore-glob="))
    }

    postgres = postgres_job["services"]["postgres"]
    assert postgres["image"] == "postgres:16"
    assert postgres["env"]["POSTGRES_HOST_AUTH_METHOD"] == "trust"
    assert [str(mapping) for mapping in postgres["ports"]] == ["127.0.0.1:55432:5432"]
    assert postgres_job["timeout-minutes"] == 150
    test_step = _pytest_step(postgres_job)
    assert test_step["timeout-minutes"] == 120
    for variable in ("DUDZIAN_TEST_POSTGRES_DSN", "ENTITLEMENT_REGISTRY_POSTGRES_ADMIN_DSN"):
        assert "127.0.0.1 port=55432" in test_step["env"][variable]
    for variable in (
        "CRYPTOHUNTER_TEST_POSTGRES_UNSAFE_DSN",
        "CRYPTOHUNTER_TEST_POSTGRES_OTHER_DSN",
    ):
        assert variable in test_step["env"]
    assert test_step["env"]["DUDZIAN_TEST_POSTGRES_RESTART_CONTAINER_ID"] == (
        "${{ job.services.postgres.id }}"
    )
    assert test_step["env"]["CRYPTOHUNTER_TEST_POSTGRES_CONTAINER"] == (
        "${{ job.services.postgres.id }}"
    )
    assert test_step["env"]["DUDZIAN_TEST_POSTGRES_CRASH_COMMAND"] == (
        '["docker","exec","--user","postgres","${{ job.services.postgres.id }}",'
        '"sh","-c","kill -KILL \\"$1\\"","sh"]'
    )
    assert "Start two independent PostgreSQL 16 negative-control clusters" in {
        step["name"] for step in postgres_job["steps"]
    }
    cluster_preflight = next(
        step
        for step in postgres_job["steps"]
        if step["name"] == "Start two independent PostgreSQL 16 negative-control clusters"
    )["run"]
    for setting in ("fsync", "synchronous_commit", "full_page_writes"):
        assert f"current_setting('{setting}')" in cluster_preflight
    assert "len({row[0] for row in observed}) == 3" in cluster_preflight
    assert '[row[1] for row in observed] == ["on", "off", "on"]' in cluster_preflight
    assert 'all(row[2:] == ("on", "on") for row in observed)' in cluster_preflight
    assert "postgres:16 -c fsync=off" in cluster_preflight
    assert "Stop isolated PostgreSQL negative-control clusters" in {
        step["name"] for step in postgres_job["steps"]
    }
    assert jobs["integration-linux"]["needs"] == [
        "py-tests-ubuntu",
        "py-tests-ubuntu-postgresql",
    ]
    assert (
        next(step for step in postgres_job["steps"] if step["name"] == "Upload pytest results")[
            "with"
        ]["name"]
        == "py-tests-ubuntu-postgresql-${{ matrix.shard }}-pytest-results"
    )


@pytest.mark.parametrize("reporting", [(), ("-ra",)])
def test_postgresql_keyword_shards_are_disjoint_and_exhaustive_in_pytest(tmp_path, reporting):
    """Exercise pytest -k itself for every issuer/history keyword combination."""
    terms = POSTGRESQL_ISSUER_KEYWORDS.split(" or ") + [POSTGRESQL_HISTORY_KEYWORDS]
    source = []
    expected = {
        name: set() for name in ("issuer-durability", "history-reproposal", "remaining-security")
    }
    for bits in range(1 << len(terms)):
        active = [term for index, term in enumerate(terms) if bits & (1 << index)]
        name = "test_case_" + str(bits) + ("_" + "_".join(active) if active else "")
        source.append(f"def {name}():\n    pass\n")
        shard = (
            "history-reproposal"
            if terms[-1] in active
            else "issuer-durability"
            if active
            else "remaining-security"
        )
        expected[shard].add("test_partition.py::" + name)
    (tmp_path / "test_partition.py").write_text("\n".join(source), encoding="utf-8")
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    observed = {}
    job = _general_ci_jobs()["py-tests-ubuntu-postgresql"]
    for shard in expected:
        arguments = _postgresql_shard_arguments(job, shard, tmp_path / shard)
        selector = arguments[arguments.index("-k") + 1]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *reporting,
                "-c",
                str(config),
                "--collect-only",
                "-q",
                "--color=no",
                "-k",
                selector,
                "test_partition.py",
            ],
            cwd=tmp_path,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        nodeids = [
            line for line in result.stdout.splitlines() if line.startswith("test_partition.py::")
        ]
        assert len(nodeids) == len(set(nodeids))
        observed[shard] = set(nodeids)
        assert observed[shard] == expected[shard]
    assert set.union(*observed.values()) == set.union(*expected.values())
    assert sum(map(len, observed.values())) == 1 << len(terms)


def test_postgresql_restart_proof_controls_exact_service_and_waits_on_test_dsn():
    path = ROOT / "tests/security/test_postgresql_freshness_authority_core.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_restart_controlled_postgresql"
    )
    helper_source = ast.get_source_segment(source, helper)
    assert helper_source is not None
    assert "DUDZIAN_TEST_POSTGRES_RESTART_CONTAINER_ID" in helper_source
    assert '["docker", "restart", container_id]' in helper_source
    assert "psycopg.connect(DSN" in helper_source
    assert "time.monotonic() + 30.0" in helper_source
    assert "shell=True" not in helper_source
    assert '["pg_ctlcluster", "16", "main", "restart"]' not in source


def test_non_linux_general_ci_excludes_only_the_explicit_database_marker():
    jobs = _general_ci_jobs()
    for job_name in ("py-tests-windows", "py-tests-macos"):
        assert "not external_postgresql" in _pytest_command(jobs[job_name])


def test_heavy_licensing_security_suite_is_split_without_coverage_gap():
    jobs = _general_ci_jobs()
    general_jobs = {
        "py-tests-windows": (45, 60),
        "py-tests-ubuntu": (120, 150),
        "py-tests-ubuntu-postgresql": (120, 150),
        "py-tests-macos": (45, 60),
    }
    for job_name, (test_timeout_minutes, job_timeout_minutes) in general_jobs.items():
        job = jobs[job_name]
        command = _pytest_command(job)
        for pattern in SERIAL_LICENSING_GLOBS:
            assert f"--ignore-glob={pattern}" in command
        test_step = _pytest_step(job)
        assert test_step["timeout-minutes"] == test_timeout_minutes
        assert job["timeout-minutes"] == job_timeout_minutes

    heavy = jobs["licensing-security-durability"]
    assert heavy["strategy"]["fail-fast"] is False
    variants = {
        (item["os"], item["shard"]): (
            item["label"],
            item["job_timeout_minutes"],
            item["test_timeout_minutes"],
        )
        for item in heavy["strategy"]["matrix"]["include"]
    }
    assert variants == {
        ("ubuntu-latest", "all"): ("ubuntu-latest", 75, 65),
        ("macos-latest", "all"): ("macos-latest", 75, 65),
        ("windows-latest", "lppi-core"): ("windows-latest", 45, 35),
        ("windows-latest", "lppi-durability"): (
            "windows-latest, lppi-durability",
            45,
            35,
        ),
        ("windows-latest", "pdsa-production"): (
            "windows-latest, pdsa-production",
            40,
            30,
        ),
        ("windows-latest", "cha-core"): ("windows-latest, cha-core", 70, 60),
        ("windows-latest", "cha-root-proof"): ("windows-latest, cha-root-proof", 60, 50),
        ("windows-latest", "cha-signed-attempt"): ("windows-latest, cha-signed-attempt", 45, 35),
    }
    assert heavy["name"] == "Licensing security/durability (${{ matrix.label }})"
    assert heavy["timeout-minutes"] == "${{ matrix.job_timeout_minutes }}"

    step = next(
        step
        for step in heavy["steps"]
        if step["name"] == "Run serial licensing security/durability suite"
    )
    command = step["run"]
    for pattern in SERIAL_LICENSING_GLOBS:
        assert pattern in command
    assert 'case "${{ matrix.shard }}" in' in command
    assert "lppi-core)" in command
    assert "lppi-durability)" in command
    assert "pdsa-production)" in command
    assert "cha-core)" in command
    assert "cha-root-proof)" in command
    assert "cha-signed-attempt)" in command
    assert "tests/licensing/test_cha_root_proof_signed*.py) continue ;;" in command
    assert "tests/licensing/test_lppi_authenticated_operation.py" in command
    assert "tests/licensing/test_lppi_authenticated_operation_durability.py" in command
    assert "tests/licensing/test_lppi_authenticated_operation_native.py" in command
    assert "tests/licensing/test_lppi_authority_lifecycle.py" in command
    assert "tests/licensing/test_pdsa_*.py" in command
    assert "tests/licensing/test_production_*.py" in command
    assert "for candidate in tests/licensing/test_cha_*.py; do" in command
    assert 'case "${candidate}" in' in command
    assert "tests/licensing/test_cha_root_proof*.py) continue ;;" in command
    assert 'licensing_tests+=("${candidate}")' in command
    assert "tests/licensing/test_cha_root_proof*.py" in command
    assert "--ignore-glob=tests/licensing/test_cha_root_proof*.py" not in command
    assert "shopt -s nullglob" in command
    assert '"${licensing_tests[@]}"' in command
    arguments = _without_pytest_reporting(_logical_pytest_arguments(command))
    assert arguments[:5] == ["-m", "pytest", "-p", "asyncio", "--fast"]
    assert "--fast" in command
    assert "-n " not in command
    assert "--numprocesses" not in command
    assert step["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"

    upload = next(
        step for step in heavy["steps"] if step["name"] == "Upload licensing security diagnostics"
    )
    assert "${{ matrix.shard }}" in upload["with"]["name"]


@pytest.mark.parametrize("job_name", ["py-tests-ubuntu", "py-tests-ubuntu-postgresql"])
def test_native_pg16_prerequisite_uses_supported_pinned_ubuntu_before_its_consumer(job_name):
    job = _general_ci_jobs()[job_name]
    # These jobs install host PG16 binaries from Ubuntu's package repository;
    # a moving runner image can drop that package before any DB proof executes.
    assert job["runs-on"] == "ubuntu-24.04"
    prerequisites = [
        step
        for step in job["steps"]
        if step.get("name") == "Install native PostgreSQL peer-auth prerequisites"
    ]
    assert len(prerequisites) == 1
    prerequisite = prerequisites[0]
    commands = [shlex.split(line) for line in prerequisite["run"].splitlines() if line.strip()]
    assert ["sudo", "apt-get", "update"] in commands
    installs = [tokens for tokens in commands if tokens[:3] == ["sudo", "apt-get", "install"]]
    assert len(installs) == 1
    assert "--yes" in installs[0] and "postgresql-16" in installs[0]
    assert "postgresql" not in installs[0]  # The unversioned package follows the OS default.
    assert not prerequisite.get("continue-on-error", False)
    consumer = (
        next(
            step
            for step in job["steps"]
            if step.get("name") == "Run authoritative native Linux peer-auth proof"
        )
        if job_name == "py-tests-ubuntu"
        else _pytest_step(job)
    )
    assert job["steps"].index(prerequisite) < job["steps"].index(consumer)


def test_native_qt_sdk_builds_keep_version_and_supported_compiler_baseline():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert document["env"]["PYSIDE6_VERSION"] == "6.7.2"
    jobs = document["jobs"]
    sdk_jobs = {
        name: job
        for name, job in jobs.items()
        if any(
            "aqt install-qt linux desktop" in step.get("run", "") for step in job.get("steps", [])
        )
    }
    assert set(sdk_jobs) == {"ui-ctest", "ui-packaging"}
    for job in sdk_jobs.values():
        assert job["runs-on"] == jobs["py-tests-ubuntu"]["runs-on"] == "ubuntu-24.04"
        sdk_installs = [
            step["run"]
            for step in job["steps"]
            if "aqt install-qt linux desktop" in step.get("run", "")
        ]
        assert len(sdk_installs) == 1
        assert 'install-qt linux desktop "${PYSIDE6_VERSION}"' in sdk_installs[0]
    native_commands = "\n".join(step.get("run", "") for step in sdk_jobs["ui-ctest"]["steps"])
    assert "cmake -S ui" in native_commands and "cmake --build ui/build-tests" in native_commands
    assert "ctest --test-dir ui/build-tests" in native_commands
    packaging_commands = "\n".join(
        step.get("run", "") for step in sdk_jobs["ui-packaging"]["steps"]
    )
    assert "python scripts/packaging/qt_bundle.py" in packaging_commands
    assert "--platform linux" in packaging_commands


def test_ubuntu_has_authoritative_elevated_native_peer_auth_execution():
    jobs = _general_ci_jobs()
    step_name = "Run authoritative native Linux peer-auth proof"
    jobs_with_step = [
        job_name
        for job_name, job in jobs.items()
        if any(step.get("name") == step_name for step in job.get("steps", []))
    ]
    assert jobs_with_step == ["py-tests-ubuntu"]
    ubuntu_steps = jobs["py-tests-ubuntu"]["steps"]
    matching_steps = [step for step in ubuntu_steps if step.get("name") == step_name]
    assert len(matching_steps) == 1

    command = matching_steps[0]["run"]
    assert "sudo " in command
    assert _without_pytest_reporting(_logical_pytest_arguments(command, '"${PYTHON_BIN}"')) == [
        "-m",
        "pytest",
        "-q",
        "tests/security/test_freshness_semantic_verifier_postgresql.py",
    ]
    assert "tests/security/test_freshness_semantic_verifier_postgresql.py" in command
    assert "--collect-only" not in command
    assert '-m "' not in command
    assert "-m '" not in command
    assert "os.geteuid() == 0" in command
    assert "mktemp -d /tmp/dz-peer-source-" in command
    assert 'rm -rf "${SOURCE_STAGE}/checkout/.git"' in command
    assert 'chown -R root:root "${SOURCE_STAGE}"' in command
    assert 'chmod -R a-w,u+rwX,go+rX "${SOURCE_STAGE}"' in command
    assert 'runuser -u "${identity}" -- test -r' in command
    assert 'runuser -u "${identity}" -- test -w' in command
    assert 'PYTHON_BIN="$(command -v python)"' in command
    assert '"${PYTHON_BIN}" != /* || ! -x "${PYTHON_BIN}"' in command
    preflight = (
        '"${PYTHON_BIN}" -c '
        "'import bot_core; import bot_core.freshness_semantic_verifier; import psycopg'"
    )
    assert preflight in command
    assert "\n            python -c " not in command
    assert command.index(preflight) < command.index('"${PYTHON_BIN}" -m pytest')
    for tool in ("runuser", "useradd", "pg_config", "initdb", "pg_ctl"):
        assert f'"{tool}"' in command

    for job_name in ("py-tests-windows", "py-tests-macos"):
        commands = "\n".join(str(step.get("run", "")) for step in jobs[job_name]["steps"])
        assert step_name not in [step.get("name") for step in jobs[job_name]["steps"]]
        assert "test_freshness_semantic_verifier_postgresql.py" not in commands


def test_peer_auth_platform_guard_precedes_pwd_and_preserves_linux_execution():
    path = ROOT / "tests/security/test_freshness_semantic_verifier_postgresql.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pwd_import_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(alias.name == "pwd" for alias in node.names)
    )
    platform_guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and "sys.platform.startswith" in ast.unparse(node.test)
        and "linux" in ast.unparse(node.test)
    )
    assert platform_guard.lineno < pwd_import_line
    assert isinstance(platform_guard.test, ast.UnaryOp)  # non-Linux only; Linux executes the proof


def test_peer_auth_module_stops_before_pwd_on_simulated_windows():
    script = """
import importlib.abc
import runpy
import sys
import psycopg
import pytest

class PwdBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "pwd":
            raise ModuleNotFoundError("blocked pwd", name="pwd")
        return None

sys.modules.pop("pwd", None)
sys.meta_path.insert(0, PwdBlocker())
sys.platform = "win32"
try:
    runpy.run_path(sys.argv[1])
except pytest.skip.Exception:
    pass
else:
    raise AssertionError("non-Linux peer-auth module was not explicitly skipped")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "tests/security/test_freshness_semantic_verifier_postgresql.py"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_direct_peer_auth_fixture_stops_before_pwd_on_simulated_windows():
    path = ROOT / "tests/security/test_postgresql_freshness_production_local_authentication.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pwd_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(alias.name == "pwd" for alias in node.names)
    )
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and "sys.platform.startswith" in ast.unparse(node.test)
    )
    assert guard.lineno < pwd_line
    assert isinstance(guard.test, ast.UnaryOp)

    script = """
import importlib.abc
import runpy
import sys
import pytest

class PwdBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "pwd":
            raise AssertionError("pwd import was attempted before the platform boundary")
        return None

sys.modules.pop("pwd", None)
sys.meta_path.insert(0, PwdBlocker())
sys.platform = "win32"
try:
    runpy.run_path(sys.argv[1])
except pytest.skip.Exception:
    pass
else:
    raise AssertionError("direct non-Linux peer-auth collection was not skipped")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], cwd=ROOT, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_stale_socket_proof_remains_portable_and_uses_short_tmp_root():
    path = ROOT / "tests/security/test_freshness_production_local_deployment.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    stale = functions["test_only_exact_owned_stale_socket_is_recreated"]
    decorators = "\n".join(ast.unparse(item) for item in stale.decorator_list)
    assert "sys.platform.startswith('linux')" not in decorators
    assert "short_unix_socket_root" in [argument.arg for argument in stale.args.args]
    helper = functions["short_unix_socket_root"]
    helper_source = ast.get_source_segment(source, helper)
    assert 'dir="/tmp"' in helper_source
    assert "< 100" in helper_source


def test_clean_tls_fixture_is_dynamic_and_production_expiry_policy_is_unchanged():
    fixture = (ROOT / "tests/test_audit_tls_assets_script.py").read_text(encoding="utf-8")
    baseline = (ROOT / "tests/test_audit_security_baseline_script.py").read_text(encoding="utf-8")
    policy = (ROOT / "bot_core/security/certificates.py").read_text(encoding="utf-8")
    assert "_healthy_certificate_pair" in baseline
    assert "timedelta(days=400)" in fixture
    assert "_CERT" not in fixture
    assert "warn_expiring_within_days: float = 30.0" in policy
    assert "remaining_days <= warn_expiring_within_days" in policy


def test_semantic_verifier_keeps_crypto_portable_and_guards_only_linux_ipc_proofs():
    path = ROOT / "tests/security/test_freshness_semantic_verifier.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level_calls = [node for node in tree.body if isinstance(node, ast.Expr)]
    assert not any("pytest.skip(" in ast.unparse(node) for node in top_level_calls)

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    portable = functions["test_three_real_ed25519_signatures_produce_one_derived_preparation"]
    assert "requires_linux_unix_ipc" not in [ast.unparse(item) for item in portable.decorator_list]

    linux_ipc_tests = {
        name
        for name, function in functions.items()
        if name.startswith("test_")
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in {"_ipc_server", "_ipc_exchange"}
            for call in ast.walk(function)
        )
    }
    assert linux_ipc_tests == {
        "test_unix_ipc_process_accepts_only_closed_valid_request",
        "test_unix_ipc_has_closed_non_crypto_failure_outcomes",
        "test_unix_ipc_process_rejects_injection_and_malformed_without_oracle",
        "test_unix_ipc_process_rejects_bad_frame_lengths",
        "test_unix_ipc_wrong_peer_cannot_invoke_and_there_is_no_tcp_listener",
    }
    for name in linux_ipc_tests:
        decorators = [ast.unparse(item) for item in functions[name].decorator_list]
        assert "requires_linux_unix_ipc" in decorators

    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "requires_linux_unix_ipc"
            for target in node.targets
        )
    )
    guard_source = ast.unparse(guard.value)
    assert "not sys.platform.startswith('linux')" in guard_source
    assert all(token in guard_source for token in ("AF_UNIX", "fork", "SO_PEERCRED", "UID"))


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_non_linux_semantic_verifier_marks_native_ipc_before_runtime(platform):
    script = """
import runpy
import sys
import tests.security.test_freshness_semantic_verifier

sys.platform = sys.argv[2]
namespace = runpy.run_path(sys.argv[1])
native = namespace["test_unix_ipc_wrong_peer_cannot_invoke_and_there_is_no_tcp_listener"]
marks = getattr(native, "pytestmark", ())
assert any(mark.name == "skipif" and mark.args == (True,) for mark in marks)
portable = namespace["test_three_real_ed25519_signatures_produce_one_derived_preparation"]
assert not any(mark.name == "skipif" for mark in getattr(portable, "pytestmark", ()))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "tests/security/test_freshness_semantic_verifier.py"),
            platform,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_cha_attempt_store_keeps_portable_semantics_separate_from_posix_mode_proof():
    path = ROOT / "tests/security/test_cha_attempt_store_production_local.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    portable = functions["test_creates_dedicated_store_with_identity_and_effective_pragmas"]
    assert portable.decorator_list == []
    portable_source = ast.get_source_segment(source, portable)
    assert portable_source is not None
    for proof in (
        "path.is_file()",
        "ProviderRole.CHA_ATTEMPT_STORE",
        "SecurityProfile.PRODUCTION_LOCAL",
        '("wal", 2, 1)',
        "store.credential_identities() == ()",
    ):
        assert proof in portable_source
    assert "st_mode" not in portable_source

    posix = functions["test_posix_store_file_is_owner_read_write_only"]
    decorators = [ast.unparse(item) for item in posix.decorator_list]
    assert any("os.name != 'posix'" in decorator for decorator in decorators)
    assert any(
        "exact 0600 is a POSIX-native filesystem permission proof" in decorator
        for decorator in decorators
    )
    mode_assertions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert) and "st_mode" in ast.unparse(node.test)
    ]
    assert len(mode_assertions) == 1
    assert mode_assertions[0] in ast.walk(posix)
    mode_source = ast.get_source_segment(source, mode_assertions[0].test)
    assert mode_source == "path.stat().st_mode & 0o777 == 0o600"


def test_cha_attempt_store_production_keeps_posix_chmod_0600_guard():
    path = ROOT / "bot_core/cha_attempt_store.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guarded_chmods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "os.name == 'posix'"
        and any(
            isinstance(child, ast.Call)
            and ast.unparse(child.func) == "os.chmod"
            and [ast.unparse(argument) for argument in child.args] == ["path", "384"]
            for child in ast.walk(node)
        )
    ]
    assert len(guarded_chmods) == 1


def test_local_signing_custody_keeps_fail_closed_unsupported_platform_lock_contract():
    path = ROOT / "bot_core/local_signing_custody.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    custody_lock = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_custody_lock"
    )
    guards = [
        node
        for node in custody_lock.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.Or)
        and {ast.unparse(part) for part in node.test.values}
        == {"os.name not in {'posix', 'nt'}", "type(exclusive) is not bool"}
    ]
    # The lock must reject BOTH unsupported OS backends and untrusted lock-mode
    # values before touching the filesystem. Regressing either half is unsafe.
    assert len(guards) == 1
    assert any(
        isinstance(node, ast.Raise)
        and "production-local custody locking is unavailable" in ast.unparse(node)
        for node in ast.walk(guards[0])
    )


def test_local_signing_custody_native_boundary_is_granular_and_complete():
    expected_portable = {
        "tests/security/test_freshness_signing_custody.py": {
            "test_test_profile_cannot_be_provisioned_as_production",
        },
        "tests/security/test_local_signing_custody.py": {
            "test_arbitrary_or_plaintext_secret_backend_is_not_production_custody",
            "test_malicious_path_subclass_is_rejected_before_semantic_methods",
            "test_exact_security_role_and_lifecycle_boundaries",
        },
    }
    for relative_path, portable_names in expected_portable.items():
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
        tests = {
            node.name: {ast.unparse(decorator) for decorator in node.decorator_list}
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        }
        assert portable_names < tests.keys()
        assert all("requires_native_custody_locking" not in tests[name] for name in portable_names)
        assert all(
            "requires_native_custody_locking" in decorators
            for name, decorators in tests.items()
            if name not in portable_names
        )
        assert not any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
            for node in tree.body
        )


def test_simulated_windows_keeps_native_custody_tests_enabled():
    script = """
import importlib
import runpy
import sys
import os

# Load the module and all platform-sensitive stdlib dependencies before the
# simulation, then rebuild only the custody marker under the non-POSIX name.
importlib.import_module(sys.argv[2])
import tests.security._local_signing_platform as boundary

os.name = "nt"
importlib.reload(boundary)
namespace = runpy.run_path(sys.argv[1])
locked = namespace[sys.argv[3]]
portable = namespace[sys.argv[4]]
assert any(mark.name == "skipif" and mark.args == (False,) for mark in locked.pytestmark)
assert not any(mark.name == "skipif" for mark in getattr(portable, "pytestmark", ()))
"""
    cases = (
        (
            "tests/security/test_freshness_signing_custody.py",
            "tests.security.test_freshness_signing_custody",
            "test_distinct_provisioning_restart_snapshot_and_active_signing",
            "test_test_profile_cannot_be_provisioned_as_production",
        ),
        (
            "tests/security/test_local_signing_custody.py",
            "tests.security.test_local_signing_custody",
            "test_offline_provisioning_is_distinct_and_restart_stable",
            "test_arbitrary_or_plaintext_secret_backend_is_not_production_custody",
        ),
    )
    for relative_path, module_name, locked, portable in cases:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(ROOT / relative_path),
                module_name,
                locked,
                portable,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr

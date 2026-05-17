from pathlib import Path


WORKFLOW_PATH = Path(".github/workflows/ci.yml")


def _extract_lint_and_test_step_blocks() -> list[list[str]]:
    lines = WORKFLOW_PATH.read_text(encoding="utf-8").splitlines()

    lint_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip() == "lint-and-test:"),
        None,
    )
    assert lint_idx is not None, "Brak joba 'lint-and-test' w .github/workflows/ci.yml"

    steps_idx = next(
        (idx for idx in range(lint_idx + 1, len(lines)) if lines[idx].startswith("    steps:")),
        None,
    )
    assert steps_idx is not None, "Brak sekcji steps w jobs.lint-and-test"

    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines[steps_idx + 1 :]:
        if line.startswith("  ") and not line.startswith("    "):
            break

        if line.startswith("      - name: "):
            if current:
                blocks.append(current)
            current = [line]
            continue

        if current:
            current.append(line)

    if current:
        blocks.append(current)

    return blocks


def _step_name(block: list[str]) -> str:
    first = block[0].strip()
    return first.replace("- name: ", "", 1).strip().strip('"')


def test_lint_and_test_contains_autonomy_matrix_step_with_expected_command_and_order() -> None:
    step_blocks = _extract_lint_and_test_step_blocks()
    step_names = [_step_name(block) for block in step_blocks]

    assert "Run autonomy matrix" in step_names, (
        "Brak kroku 'Run autonomy matrix' w jobs.lint-and-test.steps"
    )
    autonomy_idx = step_names.index("Run autonomy matrix")

    autonomy_block = step_blocks[autonomy_idx]
    autonomy_body = "\n".join(autonomy_block)
    assert "python scripts/ci/run_autonomy_matrix.py" in autonomy_body, (
        "Krok 'Run autonomy matrix' musi uruchamiać 'python scripts/ci/run_autonomy_matrix.py'"
    )

    dependency_step_names = ["Set up Python", "Download wheelhouse", "Install dependencies"]
    for dependency_step_name in dependency_step_names:
        assert dependency_step_name in step_names, (
            f"Brak kroku zależności '{dependency_step_name}' wymaganego przez kontrakt CI"
        )
        assert autonomy_idx > step_names.index(dependency_step_name), (
            f"Krok 'Run autonomy matrix' musi być po kroku '{dependency_step_name}'"
        )

    assert "Pytest (unit suite) with coverage" in step_names, (
        "Brak kroku 'Pytest (unit suite) with coverage' w jobs.lint-and-test.steps"
    )
    pytest_idx = step_names.index("Pytest (unit suite) with coverage")

    assert autonomy_idx < pytest_idx, (
        "Krok 'Run autonomy matrix' musi występować przed 'Pytest (unit suite) with coverage'"
    )


def test_lint_and_test_contains_risk_execution_matrix_step_with_expected_command_and_order() -> (
    None
):
    step_blocks = _extract_lint_and_test_step_blocks()
    step_names = [_step_name(block) for block in step_blocks]

    assert "Run risk/execution matrix" in step_names, (
        "Brak kroku 'Run risk/execution matrix' w jobs.lint-and-test.steps"
    )
    risk_idx = step_names.index("Run risk/execution matrix")

    risk_block = step_blocks[risk_idx]
    risk_body = "\n".join(risk_block)
    assert "python scripts/ci/run_risk_execution_matrix.py" in risk_body, (
        "Krok 'Run risk/execution matrix' musi uruchamiać 'python scripts/ci/run_risk_execution_matrix.py'"
    )

    for dependency_step_name in ("Set up Python", "Download wheelhouse", "Install dependencies"):
        assert dependency_step_name in step_names, (
            f"Brak kroku zależności '{dependency_step_name}' wymaganego przez kontrakt CI"
        )
        assert risk_idx > step_names.index(dependency_step_name), (
            f"Krok 'Run risk/execution matrix' musi być po kroku '{dependency_step_name}'"
        )

    assert "Run autonomy matrix" in step_names, (
        "Brak kroku 'Run autonomy matrix' wymaganego przez kontrakt kolejności"
    )
    autonomy_idx = step_names.index("Run autonomy matrix")
    assert risk_idx > autonomy_idx, (
        "Krok 'Run risk/execution matrix' musi występować po 'Run autonomy matrix'"
    )

    assert "Pytest (unit suite) with coverage" in step_names, (
        "Brak kroku 'Pytest (unit suite) with coverage' w jobs.lint-and-test.steps"
    )
    pytest_idx = step_names.index("Pytest (unit suite) with coverage")
    assert risk_idx < pytest_idx, (
        "Krok 'Run risk/execution matrix' musi występować przed 'Pytest (unit suite) with coverage'"
    )


def test_lint_and_test_contains_recovery_matrix_step_with_expected_command_and_order() -> None:
    step_blocks = _extract_lint_and_test_step_blocks()
    step_names = [_step_name(block) for block in step_blocks]

    assert "Run recovery matrix" in step_names, (
        "Brak kroku 'Run recovery matrix' w jobs.lint-and-test.steps"
    )
    recovery_idx = step_names.index("Run recovery matrix")

    recovery_block = step_blocks[recovery_idx]
    recovery_body = "\n".join(recovery_block)
    assert "python scripts/ci/run_recovery_matrix.py" in recovery_body, (
        "Krok 'Run recovery matrix' musi uruchamiać 'python scripts/ci/run_recovery_matrix.py'"
    )

    for dependency_step_name in ("Set up Python", "Download wheelhouse", "Install dependencies"):
        assert dependency_step_name in step_names, (
            f"Brak kroku zależności '{dependency_step_name}' wymaganego przez kontrakt CI"
        )
        assert recovery_idx > step_names.index(dependency_step_name), (
            f"Krok 'Run recovery matrix' musi być po kroku '{dependency_step_name}'"
        )

    assert "Run risk/execution matrix" in step_names, (
        "Brak kroku 'Run risk/execution matrix' wymaganego przez kontrakt kolejności"
    )
    risk_idx = step_names.index("Run risk/execution matrix")
    assert recovery_idx > risk_idx, (
        "Krok 'Run recovery matrix' musi występować po 'Run risk/execution matrix'"
    )

    assert "Pytest (unit suite) with coverage" in step_names, (
        "Brak kroku 'Pytest (unit suite) with coverage' w jobs.lint-and-test.steps"
    )
    pytest_idx = step_names.index("Pytest (unit suite) with coverage")
    assert recovery_idx < pytest_idx, (
        "Krok 'Run recovery matrix' musi występować przed 'Pytest (unit suite) with coverage'"
    )

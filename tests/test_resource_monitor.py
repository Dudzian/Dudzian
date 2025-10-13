from bot_core.runtime.resource_monitor import (
    ResourceBudgetEvaluation,
    ResourceBudgets,
    ResourceSample,
    evaluate_resource_sample,
)


def test_resource_budget_ok_state() -> None:
    budgets = ResourceBudgets(cpu_percent=70.0, memory_mb=2048.0, io_read_mb_s=80.0, io_write_mb_s=60.0)
    sample = ResourceSample(cpu_percent=40.0, memory_mb=1024.0, io_read_mb_s=20.0, io_write_mb_s=15.0)

    evaluation = evaluate_resource_sample(budgets, sample)

    assert isinstance(evaluation, ResourceBudgetEvaluation)
    assert evaluation.status == "ok"
    assert not evaluation.breaches
    assert not evaluation.warnings


def test_resource_budget_warn_and_error() -> None:
    budgets = ResourceBudgets(cpu_percent=60.0, memory_mb=1024.0, io_read_mb_s=50.0, io_write_mb_s=40.0)
    sample = ResourceSample(cpu_percent=59.0, memory_mb=1200.0, io_read_mb_s=30.0, io_write_mb_s=35.0)

    evaluation = evaluate_resource_sample(budgets, sample)

    assert evaluation.status == "error"
    assert "memory_mb" in evaluation.breaches
    assert "cpu_percent" in evaluation.warnings
    assert evaluation.as_dict()["status"] == "error"

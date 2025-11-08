import math

from core.perf import profile_block


def _busy_work(iterations: int = 250) -> float:
    total = 0.0
    for index in range(iterations):
        total += math.sqrt(index)
    return total


def test_profile_block_collects_structured_cpu_top() -> None:
    with profile_block("test.profiler", enable_gpu=False, limit=5) as session:
        for _ in range(3):
            _busy_work()

    report = session.report
    assert report is not None, "Profiling session should produce a report"
    assert report.cpu_top, "Report should expose structured CPU hot spots"
    assert any("_busy_work" in entry["function"] for entry in report.cpu_top)

    payload = report.as_dict()
    assert "cpu_top" in payload
    assert payload["cpu_top"][0]["total_time"] >= 0.0

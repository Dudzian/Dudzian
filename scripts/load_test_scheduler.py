"""CLI do szybkiego benchmarku scheduler-a multi-strategy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bot_core.runtime.scheduler_load_test import (
    LoadTestSettings,
    execute_scheduler_load_test,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10, help="Liczba cykli scheduler-a")
    parser.add_argument("--schedules", type=int, default=3, help="Liczba rejestrowanych harmonogramów")
    parser.add_argument("--signals", type=int, default=4, help="Liczba sygnałów na snapshot")
    parser.add_argument("--latency-ms", type=float, default=2.0, help="Symulowana latencja strategii w ms")
    parser.add_argument("--cpu-budget", type=float, default=70.0, help="Budżet CPU w procentach")
    parser.add_argument("--memory-budget", type=float, default=3072.0, help="Budżet pamięci w MB")
    parser.add_argument("--output", help="Opcjonalny plik JSON na wynik load testu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = LoadTestSettings(
        iterations=max(1, args.iterations),
        schedules=max(1, args.schedules),
        signals_per_snapshot=max(1, args.signals),
        simulated_latency_ms=max(0.0, args.latency_ms),
        cpu_budget_percent=max(1.0, args.cpu_budget),
        memory_budget_mb=max(1.0, args.memory_budget),
    )
    result = execute_scheduler_load_test(settings)
    payload = result.as_dict()
    payload["settings"] = {
        "iterations": settings.iterations,
        "schedules": settings.schedules,
        "signals_per_snapshot": settings.signals_per_snapshot,
        "simulated_latency_ms": settings.simulated_latency_ms,
        "cpu_budget_percent": settings.cpu_budget_percent,
        "memory_budget_mb": settings.memory_budget_mb,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

"""Uruchamia scheduler multi-strategy zgodnie z konfiguracją core."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Mapping

from bot_core.runtime.pipeline import build_multi_strategy_runtime
from bot_core.runtime.multi_strategy_scheduler import MultiStrategyScheduler
from bot_core.security import SecretManager


def _build_scheduler(
    *,
    config_path: Path,
    environment: str,
    scheduler_name: str | None,
) -> MultiStrategyScheduler:
    runtime = build_multi_strategy_runtime(
        environment_name=environment,
        scheduler_name=scheduler_name,
        config_path=config_path,
        secret_manager=SecretManager(),
        adapter_factories=None,
        telemetry_emitter=lambda name, payload: print(
            f"[telemetry] schedule={name} signals={payload.get('signals', 0)} latency_ms={payload.get('latency_ms', 0.0):.2f}"
        ),
    )
    return runtime.scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-strategy scheduler")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku core.yaml")
    parser.add_argument("--environment", required=True, help="Nazwa środowiska (np. binance_paper)")
    parser.add_argument("--scheduler", default=None, help="Nazwa schedulera z sekcji multi_strategy_schedulers")
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Wykonaj pojedynczy cykl harmonogramu i zakończ (tryb smoke/audit)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    scheduler = _build_scheduler(
        config_path=config_path,
        environment=args.environment,
        scheduler_name=args.scheduler,
    )

    try:
        if args.run_once:
            asyncio.run(scheduler.run_once())
        else:
            asyncio.run(scheduler.run_forever())
    except KeyboardInterrupt:
        scheduler.stop()


if __name__ == "__main__":
    main()

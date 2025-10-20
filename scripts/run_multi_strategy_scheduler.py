"""Uruchamia scheduler multi-strategy zgodnie z konfiguracją core."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Mapping, Sequence, cast

from bot_core.exchanges.base import ExchangeAdapterFactory
from bot_core.runtime.bootstrap import parse_adapter_factory_cli_specs
from bot_core.runtime.pipeline import MultiStrategyRuntime, build_multi_strategy_runtime
from bot_core.security import SecretManager


def _build_scheduler(
    *,
    config_path: Path,
    environment: str,
    scheduler_name: str | None,
    adapter_factories: Mapping[str, object] | None,
) -> object:
    runtime = build_multi_strategy_runtime(
        environment_name=environment,
        scheduler_name=scheduler_name,
        config_path=config_path,
        secret_manager=SecretManager(),
        adapter_factories=cast(Mapping[str, ExchangeAdapterFactory] | None, adapter_factories),
        telemetry_emitter=lambda name, payload: print(
            f"[telemetry] schedule={name} signals={payload.get('signals', 0)} latency_ms={payload.get('latency_ms', 0.0):.2f}"
        ),
    )
    scheduler = runtime.scheduler
    setattr(scheduler, "_runtime", runtime)
    return scheduler


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-strategy scheduler")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku core.yaml")
    parser.add_argument("--environment", required=True, help="Nazwa środowiska (np. binance_paper)")
    parser.add_argument("--scheduler", default=None, help="Nazwa schedulera z sekcji multi_strategy_schedulers")
    parser.add_argument(
        "--adapter-factory",
        action="append",
        dest="adapter_factories",
        metavar="NAME=SPEC",
        help=(
            "Override fabryk adapterów przekazywane do bootstrapu runtime. "
            "Użyj '!remove', aby usunąć wpis – opcję można podawać wielokrotnie."
        ),
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Wykonaj pojedynczy cykl harmonogramu i zakończ (tryb smoke/audit)",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    cli_adapter_specs = parse_adapter_factory_cli_specs(
        cast(Sequence[str] | None, getattr(args, "adapter_factories", None))
    )
    adapter_factories: Mapping[str, object] | None = cli_adapter_specs if cli_adapter_specs else None

    scheduler = _build_scheduler(
        config_path=config_path,
        environment=args.environment,
        scheduler_name=args.scheduler,
        adapter_factories=adapter_factories,
    )

    runtime: MultiStrategyRuntime | None = getattr(scheduler, "_runtime", None)

    try:
        if args.run_once:
            asyncio.run(scheduler.run_once())
        else:
            asyncio.run(scheduler.run_forever())
    except KeyboardInterrupt:
        stop = getattr(scheduler, "stop", None)
        if callable(stop):
            stop()
    finally:
        if runtime is not None:
            runtime.shutdown()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())

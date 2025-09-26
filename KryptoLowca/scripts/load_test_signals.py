"""Symulacja obciążenia sygnałami do testów wydajnościowych."""
from __future__ import annotations

import argparse
import random
import time

from KryptoLowca.event_emitter_adapter import EventBus, EventType
from KryptoLowca.logging_utils import get_logger
from KryptoLowca.telemetry.prometheus_exporter import metrics

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500, help="Liczba generowanych sygnałów")
    parser.add_argument("--sleep", type=float, default=0.01, help="Opóźnienie między sygnałami (s)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol testowy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bus = EventBus()
    bus.start()

    logger.info("Rozpoczynam symulację %s sygnałów", args.iterations)
    states = ["ok", "warn", "lock"]
    modes = ["demo", "live"]

    for idx in range(args.iterations):
        state = random.choices(states, weights=[0.8, 0.15, 0.05])[0]
        fraction = max(0.0, min(1.0, random.gauss(0.3, 0.15)))
        mode = modes[0]
        metrics.observe_risk(args.symbol, state, fraction, mode)
        bus.emit(EventType.RISK_ALERT, {
            "symbol": args.symbol,
            "state": state,
            "fraction": fraction,
            "seq": idx,
        })
        time.sleep(args.sleep)

    logger.info("Symulacja zakończona")
    bus.stop()


if __name__ == "__main__":
    main()

"""CLI do przechwytywania snapshotu strumienia lokalnego."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bot_core.runtime.streaming_bridge import (
    capture_stream_snapshot,
    write_snapshot_to_file,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Adres bazowy serwisu strumieniowego")
    parser.add_argument("--path", required=True, help="Ścieżka endpointu (np. /stream)")
    parser.add_argument("--adapter", required=True, help="Identyfikator adaptera giełdy")
    parser.add_argument("--scope", required=True, help="Zakres strumienia (np. public/private)")
    parser.add_argument("--environment", required=True, help="Środowisko (paper/testnet/live)")
    parser.add_argument("--channels", required=True, help="Lista kanałów, rozdzielona przecinkami")
    parser.add_argument("--output", required=True, help="Ścieżka pliku wynikowego JSON")
    parser.add_argument("--limit", type=int, default=500, help="Limit pobranych zdarzeń")
    parser.add_argument("--poll-interval", type=float, default=0.25, help="Odstęp pomiędzy pollingami")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout żądania HTTP")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv or sys.argv[1:]))
    channels = [item.strip() for item in args.channels.split(",") if item.strip()]
    if not channels:
        raise SystemExit("Wymagane jest podanie co najmniej jednego kanału")

    events = capture_stream_snapshot(
        base_url=args.base_url,
        path=args.path,
        channels=channels,
        adapter=args.adapter,
        scope=args.scope,
        environment=args.environment,
        limit=args.limit,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_snapshot_to_file(events, str(output_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())


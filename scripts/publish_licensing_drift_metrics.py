"""Publikuje metryki dryfu licencyjnego do Pushgateway.

Publish licensing drift Prometheus metrics to a Pushgateway instance. This
helper reads the generated `licensing_drift.prom` file produced by
`scripts/aggregate_licensing_drift_reports.py` and uploads it to a configured
Pushgateway endpoint so Grafana dashboards can consume the latest CI results.

Basic authentication can be provided either via the `--basic-auth` flag or an
environment variable (defaults to `LICENSING_DRIFT_PUSHGATEWAY_AUTH`) to avoid
leaking credentials in workflow logs. A small timeout protects the job from
hanging when the Pushgateway endpoint is unreachable.
"""
from __future__ import annotations

import argparse
import base64
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Mapping


def parse_grouping_labels(raw_labels: list[str]) -> Dict[str, str]:
    grouping_labels: Dict[str, str] = {}
    for raw in raw_labels:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(
                f"Grouping label '{raw}' must be in key=value format."
            )
        key, value = raw.split("=", 1)
        if not key:
            raise argparse.ArgumentTypeError("Grouping label key cannot be empty.")
        grouping_labels[key] = value
    return grouping_labels


def build_target_url(base_url: str, job_name: str, grouping_labels: Dict[str, str]) -> str:
    base = base_url.rstrip("/")
    parts = [base, "metrics", "job", urllib.parse.quote(job_name, safe="")]
    for key, value in grouping_labels.items():
        parts.extend([urllib.parse.quote(key, safe=""), urllib.parse.quote(value, safe="")])
    return "/".join(parts)


def resolve_basic_auth(raw_value: str | None, env_var: str) -> str | None:
    if raw_value:
        return raw_value
    env_value = os.getenv(env_var)
    return env_value or None


def build_headers(basic_auth: str | None) -> Dict[str, str]:
    headers: Dict[str, str] = {"User-Agent": "licensing-drift-metrics/1.0"}
    if basic_auth is None:
        return headers

    if ":" not in basic_auth:
        raise argparse.ArgumentTypeError(
            "Basic auth credentials must be provided as username:password."
        )

    encoded = base64.b64encode(basic_auth.encode("utf-8")).decode("ascii")
    headers["Authorization"] = f"Basic {encoded}"
    return headers


def push_metrics(
    prom_path: Path, target_url: str, *, headers: Mapping[str, str], timeout: float
) -> None:
    if not prom_path.exists():
        raise FileNotFoundError(f"Prometheus file not found: {prom_path}")

    payload = prom_path.read_bytes()
    if not payload.strip():
        raise ValueError(f"Prometheus file is empty: {prom_path}")

    request = urllib.request.Request(target_url, data=payload, method="PUT", headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # type: ignore[call-arg]
            response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - handled in CLI
        raise RuntimeError(
            f"Pushgateway responded with HTTP {exc.code}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - handled in CLI
        raise RuntimeError(f"Failed to reach Pushgateway: {exc.reason}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prom-file",
        default="reports/ci/licensing_drift/dashboard/licensing_drift.prom",
        help="Ścieżka do pliku .prom wygenerowanego przez agregację dryfu licencji.",
    )
    parser.add_argument(
        "--pushgateway",
        required=True,
        help="Adres Pushgateway (np. https://pushgateway.example.com)",
    )
    parser.add_argument(
        "--job-name",
        default="licensing-drift",
        help="Nazwa joba w Pushgateway (domyślnie licensing-drift)",
    )
    parser.add_argument(
        "--grouping-label",
        action="append",
        default=[],
        help="Dodatkowe etykiety grupujące (wielokrotne, format klucz=wartość)",
    )
    parser.add_argument(
        "--basic-auth",
        help=(
            "Dane podstawowej autoryzacji w formacie user:pass (zamiast tego można"
            " ustawić zmienną środowiskową)."
        ),
    )
    parser.add_argument(
        "--basic-auth-env",
        default="LICENSING_DRIFT_PUSHGATEWAY_AUTH",
        help="Nazwa zmiennej środowiskowej z danymi basic auth (domyślnie LICENSING_DRIFT_PUSHGATEWAY_AUTH)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Limit czasu (s) na połączenie z Pushgateway (domyślnie 10s)",
    )

    args = parser.parse_args(argv)
    prom_path = Path(args.prom_file)
    grouping_labels = parse_grouping_labels(args.grouping_label)
    basic_auth = resolve_basic_auth(args.basic_auth, args.basic_auth_env)
    headers = build_headers(basic_auth)

    target_url = build_target_url(args.pushgateway, args.job_name, grouping_labels)
    try:
        push_metrics(prom_path, target_url, headers=headers, timeout=args.timeout)
    except Exception as exc:  # pragma: no cover - CLI handling
        parser.error(str(exc))
        return 1

    print(f"Published licensing drift metrics to {target_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Uruchom scenariusz sandboxowy Decision Engine z linii poleceń."""

from __future__ import annotations

import argparse
from pathlib import Path

from bot_core.ai.manager import AIManager
from bot_core.observability.metrics import get_global_metrics_registry
from bot_core.runtime.journal import JsonlTradingDecisionJournal


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI sandbox scenario")
    parser.add_argument("scenario", nargs="?", default="smoke", help="Scenario identifier to execute")
    parser.add_argument("--dataset", dest="dataset", help="Optional dataset name or path")
    parser.add_argument("--model", dest="model", help="Decision model name to use")
    parser.add_argument(
        "--dashboard-output",
        dest="dashboard_output",
        help="Override path for dashboard annotations JSON",
    )
    parser.add_argument(
        "--journal-dir",
        dest="journal_dir",
        help="Directory for decision journal JSONL output",
    )
    parser.add_argument(
        "--instrument",
        dest="instruments",
        action="append",
        help="Instrument symbol filter (can be repeated)",
    )
    parser.add_argument(
        "--event-type",
        dest="event_types",
        action="append",
        help="Event type filter (can be repeated)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    journal = None
    if args.journal_dir:
        journal_path = Path(args.journal_dir)
        journal = JsonlTradingDecisionJournal(journal_path)
        journal_path.mkdir(parents=True, exist_ok=True)
    manager = AIManager(decision_journal=journal)
    result = manager.run_sandbox_scenario(
        args.scenario,
        dataset=args.dataset,
        model_name=args.model,
        dashboard_output=args.dashboard_output,
        metrics_registry=get_global_metrics_registry(),
        decision_journal=journal,
        instruments=args.instruments,
        event_types=args.event_types,
    )
    print(
        "Scenario '%s' processed %d events from %s" %
        (result.scenario, result.processed_events, result.dataset)
    )
    if result.risk_limit_summary:
        print("Observed risk limit utilisation:")
        for instrument, summaries in result.risk_limit_summary.items():
            print(f"  {instrument}:")
            for summary in summaries:
                max_util = f"{summary.max_utilization:.3f}" if summary.max_utilization else "0.000"
                threshold_util = (
                    f"{summary.max_threshold_utilization:.3f}"
                    if summary.max_threshold_utilization
                    else "0.000"
                )
                print(
                    "    - {code}: max_util={max_util}, threshold_util={threshold_util}, hard_breaches={hard}, threshold_breaches={soft}".format(
                        code=summary.code,
                        max_util=max_util,
                        threshold_util=threshold_util,
                        hard=summary.hard_limit_breaches,
                        soft=summary.threshold_breaches,
                    )
                )
    quantile_specs = list(result.score_quantiles)

    def _format_score_summary(summary: dict[str, float | int]) -> str:
        mean = float(summary.get("mean", 0.0))
        count = int(float(summary.get("count", 0.0)))
        minimum = summary.get("min")
        maximum = summary.get("max")
        stddev = float(summary.get("stddev", 0.0))
        sample_stddev = float(summary.get("sample_stddev", 0.0))
        skewness = summary.get("skewness")
        kurtosis_excess = summary.get("kurtosis_excess")
        quantile_fields = []
        for name, _ in quantile_specs:
            value = summary.get(name)
            if value is not None:
                quantile_fields.append(f"{name}={float(value):.4f}")
        quantiles = ", ".join(quantile_fields) if quantile_fields else "none"
        return (
            "mean={mean:.4f}, stddev={stddev:.4f}, sample_stddev={sample_stddev:.4f}, "
            "skewness={skewness}, kurtosis_excess={kurtosis}, count={count}, "
            "min={min_value}, max={max_value}, quantiles=[{quantiles}]"
        ).format(
            mean=mean,
            stddev=stddev,
            sample_stddev=sample_stddev,
            skewness=f"{float(skewness):.4f}" if skewness is not None else "n/a",
            kurtosis=f"{float(kurtosis_excess):.4f}" if kurtosis_excess is not None else "n/a",
            count=count,
            min_value=f"{minimum:.4f}" if minimum is not None else "n/a",
            max_value=f"{maximum:.4f}" if maximum is not None else "n/a",
            quantiles=quantiles,
        )

    if result.decision_score_summary:
        print("Decision score summary:")
        for metric, summary in sorted(result.decision_score_summary.items()):
            print(f"  {metric}: {_format_score_summary(summary)}")
    if result.decision_score_summary_by_instrument:
        print("Decision score summary by instrument:")
        for instrument, metrics in sorted(result.decision_score_summary_by_instrument.items()):
            print(f"  {instrument}:")
            for metric, summary in sorted(metrics.items()):
                print(f"    {metric}: {_format_score_summary(summary)}")


if __name__ == "__main__":
    main()

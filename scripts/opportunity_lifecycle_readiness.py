"""Operator entrypoint for Opportunity AI persisted lifecycle readiness report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

if __package__ is None:  # pragma: no cover - uruchomienie jako skrypt
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.ai.opportunity_lifecycle import OpportunityLifecycleService
from bot_core.ai.repository import FilesystemModelRepository
from bot_core.ai.trading_opportunity_shadow import OpportunityShadowRepository


def build_opportunity_lifecycle_readiness_report(
    *,
    model_repository_path: str | Path,
    shadow_repository_path: str | Path,
    champion_version: str,
    challenger_version: str,
) -> Mapping[str, Any]:
    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=FilesystemModelRepository(Path(model_repository_path)),
        shadow_repository=OpportunityShadowRepository(Path(shadow_repository_path)),
        champion_version=champion_version,
        challenger_version=challenger_version,
    )
    return _json_safe(asdict(report))


def _json_safe(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", required=True, help="Path to model repository root")
    parser.add_argument(
        "--shadow-repo", required=True, help="Path to opportunity shadow repository root"
    )
    parser.add_argument("--champion-version", required=True, help="Champion model version")
    parser.add_argument("--challenger-version", required=True, help="Challenger model version")
    parser.add_argument("--output", help="Optional path to save JSON report")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_opportunity_lifecycle_readiness_report(
        model_repository_path=args.model_repo,
        shadow_repository_path=args.shadow_repo,
        champion_version=args.champion_version,
        challenger_version=args.challenger_version,
    )
    json_kwargs: dict[str, object] = {"ensure_ascii": False}
    if args.pretty:
        json_kwargs.update({"indent": 2, "sort_keys": True})
    payload = json.dumps(report, **json_kwargs)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Generate explicit marketing parity skip reports for CI fallback paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def render_markdown(reason: str, required_s3: str, required_git: str) -> str:
    return "\n".join(
        [
            "## Marketing parity check skipped (infrastructure precondition)",
            "",
            f"Reason: `{reason}`.",
            "",
            "Parity validation did not execute due to infrastructure precondition failure.",
            "",
            "This is an **infrastructure skip**, not a parity pass.",
            "Downstream consumers must treat `status=skipped` as non-passing.",
            "",
            f"Set either `{required_s3}` or `{required_git}` to enable full parity validation.",
            "",
        ]
    )


def build_payload(reason: str, required_s3: str, required_git: str) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
        "classification": "infra_skip",
        "parity_validated": False,
        "result": "non_passing",
        "required": [required_s3, required_git],
    }


def write_skip_reports(markdown_output: Path, json_output: Path, reason: str) -> None:
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    markdown_output.write_text(
        render_markdown(
            reason=reason,
            required_s3="MARKETING_PARITY_MIRROR_S3",
            required_git="MARKETING_PARITY_MIRROR_GIT",
        ),
        encoding="utf-8",
    )
    json_output.write_text(
        json.dumps(
            build_payload(
                reason=reason,
                required_s3="MARKETING_PARITY_MIRROR_S3",
                required_git="MARKETING_PARITY_MIRROR_GIT",
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CI skip report for marketing parity check")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("docs/audit/marketing_parity_report.md"),
        help="Path to markdown report",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("docs/audit/marketing_parity_report.json"),
        help="Path to JSON report",
    )
    parser.add_argument(
        "--reason",
        default="mirror_not_configured",
        help="Machine-readable reason code for skip",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    write_skip_reports(args.audit_output, args.json_output, reason=args.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

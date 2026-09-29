"""Offline TEST_ONLY enrollment administration; production is intentionally impossible."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.licensing.activation_request import ActivationRequestV1, EnrollmentDecisionV1
from bot_core.licensing.authority import OfflineEnrollmentAuthority, TestOnlyPDSAAuthority
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical

STOP = "STOP — PRODUCTION PDSA CEREMONY MATERIAL NOT ACTIVATED FOR LICENSING."


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--environment" in raw_argv:
        position = raw_argv.index("--environment")
        if position + 1 >= len(raw_argv) or raw_argv[position + 1].upper() != "TEST_ONLY":
            raise SystemExit(STOP)

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="TEST_ONLY")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-request")
    inspect.add_argument("request")
    issue = commands.add_parser("issue-test-license")
    issue.add_argument("--request", required=True)
    issue.add_argument("--decision", required=True)
    issue.add_argument("--output", required=True)
    args = parser.parse_args(raw_argv)
    if args.environment.upper() != "TEST_ONLY":
        parser.error(STOP)

    request = ActivationRequestV1.from_mapping(parse_canonical(Path(args.request).read_bytes()))
    if args.command == "inspect-request":
        print(json.dumps(request.document, indent=2, ensure_ascii=False))
        return 0
    decision = EnrollmentDecisionV1.from_mapping(parse_canonical(Path(args.decision).read_bytes()))
    package = OfflineEnrollmentAuthority(TestOnlyPDSAAuthority.deterministic_fixture()).issue(
        request, decision
    )
    Path(args.output).write_bytes(canonical_json_bytes(package))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

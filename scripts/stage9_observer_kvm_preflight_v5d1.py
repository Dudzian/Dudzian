"""Emit the fixed-policy operator-host read-only preflight as JSON.

Run from an independently reviewed protected installation. Python imports run
before policy inspection; this CLI cannot authenticate its own launch. Exit 2
always preserves V5d1's execution block, including after complete local reads.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from deployment.stage9_observer_kvm_preflight_v5d1 import collect_preflight_v5d1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = collect_preflight_v5d1()
    print(json.dumps(asdict(report), sort_keys=True, indent=2, allow_nan=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Utility CLI to refresh committed exchange presets against the current YAML specs.

The flow mirrors the CI expectation from ``tests/marketplace/test_exchange_presets_repository.py``:
exchange definitions in ``config/exchanges`` are rendered into canonical preset payloads,
those payloads are signed with the repository Ed25519 key, and the result is validated
to ensure signatures and payload hashes stay in sync with the specs (spec-hash strategy).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping

from bot_core.marketplace import reconcile_exchange_presets, validate_exchange_presets

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCHANGES_DIR = REPO_ROOT / "config" / "exchanges"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "config" / "marketplace" / "presets" / "exchanges"
DEFAULT_PRIVATE_KEY = REPO_ROOT / "config" / "marketplace" / "keys" / "dev-presets-ed25519.key"
DEFAULT_PUBLIC_KEY = REPO_ROOT / "config" / "marketplace" / "keys" / "dev-presets-ed25519.pub"


def _load_key_material(path: str | Path | None) -> bytes | None:
    if path is None:
        return None
    return Path(path).expanduser().read_bytes()


def _print_status(result) -> None:
    status_bits: list[str] = []
    if not result.exists:
        status_bits.append("missing")
    if not result.verified:
        status_bits.append("unverified")
    if not result.up_to_date:
        status_bits.append("stale")
    status = "OK" if not status_bits and not result.issues else ", ".join(status_bits or ["issues"])
    version_info = result.current_version or "n/a"
    print(
        f"- {result.preset_id} → {result.preset_path} | current={version_info} | expected={result.expected_version} | {status}"
    )
    for issue in result.issues:
        print(f"    • {issue}")


def _summarize(results) -> None:
    healthy = [r for r in results if r.exists and r.verified and r.up_to_date and not r.issues]
    print(
        f"\nSummary: {len(healthy)}/{len(results)} presets verified and current; {len(results) - len(healthy)} require attention."
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exchanges-dir",
        type=Path,
        default=DEFAULT_EXCHANGES_DIR,
        help="Directory with YAML exchange definitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for signed exchange presets.",
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        default=DEFAULT_PRIVATE_KEY,
        help="Ed25519 private key used to sign presets.",
    )
    parser.add_argument(
        "--public-key",
        type=Path,
        default=DEFAULT_PUBLIC_KEY,
        help="Optional Ed25519 public key to verify signatures after regeneration.",
    )
    parser.add_argument(
        "--key-id",
        default="dev-presets",
        help="Identifier written to the signature block.",
    )
    parser.add_argument(
        "--issuer",
        default="marketplace-ci",
        help="Optional issuer recorded in signature metadata.",
    )
    parser.add_argument(
        "--version",
        help="Base version used for the spec-hash strategy (default: reuse current or 1.0.0).",
    )
    parser.add_argument(
        "--version-strategy",
        choices=["spec-hash", "static"],
        default="spec-hash",
        help="Versioning strategy used during reconciliation.",
    )
    parser.add_argument(
        "--exchange",
        "-e",
        action="append",
        help="Limit reconciliation to selected exchanges (may be repeated).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate presets without writing changes.",
    )
    parser.add_argument(
        "--keep-orphans",
        action="store_true",
        help="Do not remove orphaned preset files during reconciliation.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    signing_keys: Mapping[str, bytes] | None = None
    try:
        pub_bytes = _load_key_material(args.public_key)
    except OSError as exc:
        raise SystemExit(f"Failed to read public key: {exc}") from exc
    if pub_bytes:
        signing_keys = {args.key_id: pub_bytes}

    if args.check_only:
        print("Validating existing exchange presets...")
        results = validate_exchange_presets(
            exchanges_dir=args.exchanges_dir,
            output_dir=args.output_dir,
            version=args.version,
            signing_keys=signing_keys,
            selected_exchanges=args.exchange,
            version_strategy=args.version_strategy,
        )
    else:
        print("Reconciling exchange presets against YAML specs...")
        # Flow: YAML specs -> canonical preset payload -> Ed25519 signature -> JSON file,
        # preserving auxiliary catalog fields if they already exist. Validation ensures the
        # payload matches the current spec-hash fingerprint and the signature verifies.
        try:
            private_key = _load_key_material(args.private_key)
        except OSError as exc:
            raise SystemExit(f"Failed to read private key: {exc}") from exc
        results = reconcile_exchange_presets(
            exchanges_dir=args.exchanges_dir,
            output_dir=args.output_dir,
            private_key=private_key,
            key_id=args.key_id,
            issuer=args.issuer,
            version=args.version,
            signing_keys=signing_keys,
            remove_orphans=not args.keep_orphans,
            selected_exchanges=args.exchange,
            version_strategy=args.version_strategy,
        )

    for result in results:
        _print_status(result)
    _summarize(results)
    unhealthy = [
        r for r in results if not (r.exists and r.verified and r.up_to_date and not r.issues)
    ]
    return 1 if unhealthy else 0


if __name__ == "__main__":  # pragma: no cover - manual maintenance entrypoint
    raise SystemExit(main())

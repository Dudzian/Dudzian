"""Utilities to compare marketing bundle hashes with mirror copies.

This script reads the Stress Lab artefacts (packaged as the
``stress-lab-report`` CI artifact) and the aggregated
``signal_quality/index.csv`` and compares their SHA-256 hashes against
mirror copies stored in S3 or a Git mirror that has been synced locally.

The script is intentionally file-system based so that CI steps can fetch
artefacts however they prefer (``aws s3 sync``, ``git clone`` or a plain
copy), then pass the resulting root directory through ``--mirror-dir``.

It exits with a non-zero status code when:
* any required source artefact is missing,
* the mirror is missing the counterpart file, or
* SHA-256 digests do not match.

The Markdown report is generated regardless of success/failure, allowing
CI to upload it as an audit artefact or copy selected snippets to
``docs/audit/paper_trading_log.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Tuple


@dataclass
class FileHash:
    key: str
    path: Path
    sha256: str


class ParityError(Exception):
    pass


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        rel = Path(path.name)
    return PurePosixPath(rel).as_posix()


def collect_files(
    stress_lab_dir: Path | None, signal_quality_index: Path, root: Path
) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []

    if stress_lab_dir:
        if not stress_lab_dir.exists():
            raise ParityError(f"Stress Lab directory missing: {stress_lab_dir}")
        if not stress_lab_dir.is_dir():
            raise ParityError(f"Stress Lab path is not a directory: {stress_lab_dir}")
        for file_path in sorted(stress_lab_dir.rglob("*")):
            if file_path.is_file():
                files.append((_relative_posix(file_path, root), file_path))

    if not signal_quality_index.exists():
        raise ParityError(f"Signal quality index missing: {signal_quality_index}")
    if not signal_quality_index.is_file():
        raise ParityError(f"Signal quality index is not a file: {signal_quality_index}")
    files.append((_relative_posix(signal_quality_index, root), signal_quality_index))

    return files


def hash_files(files: Iterable[Tuple[str, Path]]) -> List[FileHash]:
    hashed: List[FileHash] = []
    for key, path in files:
        hashed.append(FileHash(key=key, path=path, sha256=compute_sha256(path)))
    return hashed


def compare_against_mirror(local_hashes: List[FileHash], mirror_root: Path) -> Dict[str, List[str]]:
    missing_in_mirror: List[str] = []
    mismatched: List[str] = []

    for file_hash in local_hashes:
        mirror_path = mirror_root / file_hash.key
        if not mirror_path.exists():
            missing_in_mirror.append(str(file_hash.key))
            continue

        mirror_hash = compute_sha256(mirror_path)
        if mirror_hash != file_hash.sha256:
            mismatched.append(
                json.dumps(
                    {
                        "file": str(file_hash.key),
                        "local_sha256": file_hash.sha256,
                        "mirror_sha256": mirror_hash,
                    },
                    ensure_ascii=False,
                )
            )

    return {"missing_in_mirror": missing_in_mirror, "mismatched": mismatched}


def render_report(
    local_hashes: List[FileHash],
    mirror_root: Path,
    parity: Dict[str, List[str]],
    error: str | None = None,
) -> str:
    lines = ["# Marketing bundle parity report", ""]

    if error:
        lines.append("## Błąd walidacji")
        lines.append(error)
        lines.append("")
    lines.append(f"Mirror root: `{mirror_root}`")
    lines.append("")

    lines.append("## Local artefact hashes")
    lines.append("| Plik | SHA-256 | Lokalizacja |")
    lines.append("|------|---------|-------------|")
    for entry in local_hashes:
        lines.append(f"| `{entry.key}` | `{entry.sha256}` | `{entry.path}` |")
    lines.append("")

    lines.append("## Wynik porównania z lustrem")
    if not parity["missing_in_mirror"] and not parity["mismatched"]:
        lines.append("✅ Wszystkie pliki w lustrze posiadają identyczne sumy SHA-256.")
    else:
        if parity["missing_in_mirror"]:
            lines.append("❌ Brakujące w lustrze:")
            for item in parity["missing_in_mirror"]:
                lines.append(f"- {item}")
        if parity["mismatched"]:
            lines.append("❌ Rozbieżności hashy:")
            for item in parity["mismatched"]:
                lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines)


def write_report(report_path: Path, content: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def render_json_report(
    local_hashes: List[FileHash],
    mirror_root: Path,
    parity: Dict[str, List[str]],
    error: str | None = None,
) -> Dict[str, object]:
    missing = parity.get("missing_in_mirror", [])
    mismatched = parity.get("mismatched", [])
    status = "failed" if error or missing or mismatched else "ok"
    return {
        "status": status,
        "error": error,
        "mirror_root": str(mirror_root),
        "generated_at": int(time.time()),
        "missing_in_mirror": missing,
        "mismatched": mismatched,
        "local_hashes": [
            {"file": entry.key, "sha256": entry.sha256, "source": str(entry.path)}
            for entry in local_hashes
        ],
    }


def write_json_report(report_path: Path, payload: Dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_parity_check(
    stress_lab_dir: Path | None,
    signal_quality_index: Path,
    mirror_dir: Path,
    report_output: Path,
    json_output: Path | None = None,
) -> None:
    if not mirror_dir.exists():
        raise ParityError(f"Mirror directory not found: {mirror_dir}")
    if not mirror_dir.is_dir():
        raise ParityError(f"Mirror path is not a directory: {mirror_dir}")
    root = (
        Path(stress_lab_dir).resolve().parent
        if stress_lab_dir
        else signal_quality_index.resolve().parent.parent
    )
    files = collect_files(stress_lab_dir, signal_quality_index, root)
    local_hashes = hash_files(files)
    parity = compare_against_mirror(local_hashes, mirror_dir)
    parity_error = None
    if parity["missing_in_mirror"] or parity["mismatched"]:
        details: List[str] = []
        if parity["missing_in_mirror"]:
            details.append(f"missing in mirror ({len(parity['missing_in_mirror'])})")
        if parity["mismatched"]:
            details.append(f"mismatched hashes ({len(parity['mismatched'])})")
        suffix = f": {', '.join(details)}" if details else ""
        parity_error = f"Mirror parity check failed{suffix}"

    report = render_report(local_hashes, mirror_dir, parity, error=parity_error)
    write_report(report_output, report)
    if json_output:
        write_json_report(
            json_output, render_json_report(local_hashes, mirror_dir, parity, error=parity_error)
        )

    if parity_error:
        raise ParityError(parity_error)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate marketing bundle hashes against mirror")
    parser.add_argument(
        "--stress-lab-dir",
        type=Path,
        default=None,
        help="Directory containing Stress Lab artefacts (from the stress-lab-report artifact)",
    )
    parser.add_argument(
        "--signal-quality-index",
        type=Path,
        required=True,
        help="Path to signal_quality/index.csv from the exchange-report artifact",
    )
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        required=True,
        help="Root directory of the marketing mirror (S3 sync or Git clone)",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("docs/audit/marketing_parity_report.md"),
        help="Where to write the Markdown audit report",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON audit file mirroring the Markdown report",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    previous_mtime = args.audit_output.stat().st_mtime if args.audit_output.exists() else None
    json_previous_mtime = (
        args.json_output.stat().st_mtime if args.json_output and args.json_output.exists() else None
    )

    try:
        run_parity_check(
            stress_lab_dir=args.stress_lab_dir,
            signal_quality_index=args.signal_quality_index,
            mirror_dir=args.mirror_dir,
            report_output=args.audit_output,
            json_output=args.json_output,
        )
    except ParityError as exc:
        # Still exit with non-zero status but keep the report for debugging.
        report_updated = False
        if args.audit_output.exists():
            current_mtime = args.audit_output.stat().st_mtime
            report_updated = previous_mtime is None or current_mtime > previous_mtime

        json_updated = False
        if args.json_output and args.json_output.exists():
            current_mtime = args.json_output.stat().st_mtime
            json_updated = json_previous_mtime is None or current_mtime > json_previous_mtime

        if not report_updated:
            write_report(
                args.audit_output,
                "\n".join(
                    [
                        "# Marketing bundle parity report (failed)",
                        "",
                        f"Mirror root: `{args.mirror_dir}`",
                        "",
                        "## Błąd walidacji",
                        f"- {exc}",
                        "",
                        "Raport wygenerowany po niepowodzeniu walidacji parytetu.",
                    ]
                ),
            )
        if args.json_output and not json_updated:
            write_json_report(
                args.json_output,
                render_json_report(
                    local_hashes=[], mirror_root=args.mirror_dir, parity={}, error=str(exc)
                ),
            )
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

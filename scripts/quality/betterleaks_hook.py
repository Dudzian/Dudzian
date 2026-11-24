from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

# Keep comfortably below Windows' 32,767 character command-line limit and the
# usual POSIX ARG_MAX.  Splitting the targets would change Betterleaks' scan
# scope/fingerprints, so an oversized scope is rejected rather than weakened.
_MAX_COMMAND_BYTES = 30_000 if os.name == "nt" else 128_000
# Resource limits refuse the commit instead of silently omitting large blobs.
_MAX_BLOB_BYTES = 32 * 1024 * 1024
_MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
_WINDOWS_DEVICES = {"con", "prn", "aux", "nul", "conin$", "conout$"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in "123456789¹²³"
}


def _find_betterleaks() -> str | None:
    configured = os.environ.get("BETTERLEAKS_BIN")
    candidates = [configured, shutil.which("betterleaks")]
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        candidates.append(
            str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "betterleaks" / "betterleaks.exe")
        )
    return next(
        (
            str(Path(candidate).resolve())
            for candidate in candidates
            if candidate and Path(candidate).is_file()
        ),
        None,
    )


def _staged_entries(repository: Path) -> list[tuple[str, str, str]] | None:
    # --no-renames leaves one destination record per new blob; lowercase d
    # excludes only deletions, including type changes and unresolved entries.
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "--no-ext-diff",
            "--no-relative",
            "--ignore-submodules=none",
            "--diff-filter=d",
            "-z",
            "--",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        print("Unable to read staged paths; refusing to skip the secret scan.", file=sys.stderr)
        return None
    records = completed.stdout.split(b"\0")
    if records.pop() != b"" or len(records) % 2:
        print("Incomplete staged index report; refusing the secret scan.", file=sys.stderr)
        return None
    entries: list[tuple[str, str, str]] = []
    for metadata, raw_path in zip(records[::2], records[1::2], strict=True):
        fields = metadata.split()
        if len(fields) != 5 or not fields[0].startswith(b":"):
            print("Invalid staged index report; refusing the secret scan.", file=sys.stderr)
            return None
        _, mode, _, oid, status = fields
        path = os.fsdecode(raw_path)
        if mode not in {b"100644", b"100755", b"120000"} or status not in {b"A", b"M", b"T"}:
            print(
                "Unresolved or unsupported staged file type; refusing the secret scan.",
                file=sys.stderr,
            )
            return None
        if (
            not _safe_index_path(path)
            or len(oid) not in {40, 64}
            or any(value not in b"0123456789abcdef" for value in oid)
        ):
            print("Invalid staged path or object ID; refusing the secret scan.", file=sys.stderr)
            return None
        entries.append((path, mode.decode("ascii"), oid.decode("ascii")))
    return entries


def _safe_index_path(path: str) -> bool:
    # Reject lexical traversal and platform-dependent drive/alternate-stream
    # names before writing anything to the private snapshot directory.
    return bool(path) and not (
        PurePosixPath(path).is_absolute()
        or PureWindowsPath(path).drive
        or "\\" in path
        or ":" in path
        or any(
            part in {"", ".", ".."}
            or part.casefold() == ".git"
            or part.endswith((".", " "))
            or part.split(".", 1)[0].rstrip(" ").casefold() in _WINDOWS_DEVICES
            or any(character in '<>"|?*' or ord(character) < 32 for character in part)
            or any(0xD800 <= ord(character) <= 0xDFFF for character in part)
            for part in path.split("/")
        )
    )


def _unambiguous_snapshot_paths(entries: list[tuple[str, str, str]]) -> bool:
    # Refuse aliases on every platform before any blob is materialized. Prefix
    # checks also cover a file aliasing a directory on case-insensitive systems.
    prefixes: dict[str, str] = {}
    files = {path.casefold() for path, _mode, _oid in entries}
    if len(files) != len(entries):
        return False
    for path, _mode, _oid in entries:
        if not _safe_index_path(path):
            return False
        parts = path.split("/")
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            key = prefix.casefold()
            if key in prefixes and prefixes[key] != prefix:
                return False
            if length < len(parts) and key in files:
                return False
            prefixes[key] = prefix
    return True


def _scan_paths(repository: Path, supplied: list[str] | None) -> list[str] | None:
    targets: list[str] = []
    for item in supplied or []:
        candidate = Path(item)
        absolute = candidate if candidate.is_absolute() else repository / candidate
        try:
            relative = absolute.absolute().relative_to(repository.resolve())
            if not _safe_index_path(relative.as_posix()):
                raise ValueError("unsafe relative path")
            absolute.resolve(strict=True).relative_to(repository.resolve())
        except ValueError:
            print("Secret-scan path is outside the repository or unsafe.", file=sys.stderr)
            return None
        try:
            mode = absolute.lstat().st_mode
        except OSError:
            print("Unable to read explicit secret-scan path; refusing to skip it.", file=sys.stderr)
            return None
        if not stat.S_ISREG(mode):
            print("Explicit secret-scan path must be a regular file.", file=sys.stderr)
            return None
        targets.append(relative.as_posix())
    return targets


def _configuration_arguments(repository: Path, default_config: Path) -> list[str]:
    # Keep the original repository's config/ignore context when the scan cwd is
    # the index snapshot. Absolute paths also preserve relative env overrides.
    arguments = ["--gitleaks-ignore-path", str(repository)]
    configured = os.environ.get("BETTERLEAKS_CONFIG") or os.environ.get("GITLEAKS_CONFIG")
    if configured:
        return [*arguments, "--config", str((repository / configured).absolute())]
    if os.environ.get("BETTERLEAKS_CONFIG_TOML") or os.environ.get("GITLEAKS_CONFIG_TOML"):
        return arguments
    for name in (".betterleaks.toml", ".gitleaks.toml"):
        candidate = repository / name
        if candidate.exists() or candidate.is_symlink():
            return [*arguments, "--config", str(candidate)]
    # Explicit defaults prevent a staged-only TOML file from silently becoming
    # the scanner's configuration merely because it exists in the snapshot.
    default_config.write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
    return [*arguments, "--config", str(default_config)]


def _run_scan(
    repository: Path, targets: list[str], baseline: Path, configuration: list[str] | None = None
) -> int:
    executable = _find_betterleaks()
    if executable is None:
        print(
            "Betterleaks is required for the staged secret scan. "
            "Install stable v1.9.x or set BETTERLEAKS_BIN.",
            file=sys.stderr,
        )
        return 2

    baseline_args = ["--baseline-path", str(baseline)] if baseline.is_file() else []
    command = [
        executable,
        "dir",
        *baseline_args,
        *(configuration or []),
        "--redact=100",
        "--no-banner",
        "--no-color",
        "--log-level=debug",
        "--exit-code=1",
        "--",
        *targets,
    ]
    command_bytes = sum(len(os.fsencode(argument)) + 1 for argument in command)
    if command_bytes > _MAX_COMMAND_BYTES:
        print(
            "Staged secret-scan scope exceeds the safe command-line limit; "
            "refusing to split or skip it.",
            file=sys.stderr,
        )
        return 2

    # Betterleaks 1.9 reports some skipped binary/archive inputs only in its
    # diagnostics. Capture these instead of interpreting exit zero as coverage.
    completed = subprocess.run(command, cwd=repository, check=False, capture_output=True)
    diagnostics = completed.stderr or b""
    incomplete = any(
        marker in diagnostics.lower()
        for marker in (
            b"skipping binary file",
            b"skipping archive:",
            b"skipping unknown archive type",
            b"skipping compressed file:",
            b"partial scan",
            b"could not ",
            b" err ",
            b" ftl ",
        )
    )
    if incomplete:
        # Debug output can include config values and sensitive paths. Emit only
        # a category; never forward the captured diagnostics to CI artifacts.
        print("Betterleaks scan was incomplete or reported errors; refusing.", file=sys.stderr)
        return 2
    if completed.returncode:
        print("Betterleaks reported findings or failed; refusing the secret scan.", file=sys.stderr)
    return completed.returncode


def _snapshot_entries(
    repository: Path, snapshot: Path, entries: list[tuple[str, str, str]]
) -> list[str] | None:
    if not _unambiguous_snapshot_paths(entries):
        print("Unsafe or colliding staged paths; refusing the secret scan.", file=sys.stderr)
        return None
    targets: list[str] = []
    total = 0
    for path, _mode, oid in entries:
        size = subprocess.run(
            ["git", "cat-file", "-s", oid], cwd=repository, check=False, capture_output=True
        )
        try:
            count = int(size.stdout)
        except ValueError:
            count = -1
        if (
            size.returncode
            or count < 0
            or count > _MAX_BLOB_BYTES
            or total + count > _MAX_SNAPSHOT_BYTES
        ):
            print(
                "Staged object unreadable or exceeds the scan resource limit; refusing.",
                file=sys.stderr,
            )
            return None
        blob = subprocess.run(
            ["git", "cat-file", "blob", oid], cwd=repository, check=False, capture_output=True
        )
        if blob.returncode or len(blob.stdout) != count:
            print("Unable to read complete staged blob; refusing the secret scan.", file=sys.stderr)
            return None
        # Binary/encoded data cannot receive a complete text-scan claim. The
        # caller must resolve it instead of receiving a silent scanner skip.
        try:
            blob.stdout.decode("utf-8")
        except UnicodeDecodeError:
            print("Non-UTF-8 staged blob requires separate review; refusing.", file=sys.stderr)
            return None
        if b"\0" in blob.stdout:
            print("Binary staged blob requires separate review; refusing.", file=sys.stderr)
            return None
        destination = snapshot / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A Git symlink is a blob containing its target text. Materialize that
        # text as a regular file; never follow or recreate the link itself.
        with destination.open("xb") as output:
            output.write(blob.stdout)
        targets.append(path)
        total += count
    return targets


def main(argv: list[str] | None = None) -> int:
    repository = Path.cwd().resolve()
    baseline = repository / ".betterleaks-baseline.json"
    try:
        if argv:
            targets = _scan_paths(repository, argv)
            if targets is None:
                return 2
            return _run_scan(repository, targets, baseline) if targets else 0
        entries = _staged_entries(repository)
        if entries is None:
            return 2
        if not entries:
            return 0
        with tempfile.TemporaryDirectory(prefix="betterleaks-index-") as directory:
            temporary = Path(directory)
            snapshot = temporary / "index"
            snapshot.mkdir()
            targets = _snapshot_entries(repository, snapshot, entries)
            if targets is None:
                return 2
            configuration = _configuration_arguments(repository, temporary / "scanner-default.toml")
            result = _run_scan(snapshot, targets, baseline, configuration)
            if _staged_entries(repository) != entries:
                print("Staged index changed during the scan; refusing success.", file=sys.stderr)
                return 2
            return result
    except (OSError, ValueError):
        # Do not emit exception text: Git paths and file content may be sensitive.
        print("Unable to complete the secret scan; refusing success.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

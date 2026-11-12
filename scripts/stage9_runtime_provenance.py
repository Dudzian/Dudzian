"""Read-only observations for an already signature-verified runtime manifest.

The verified digest is supplied from the independent runtime verifier. This
collector never authenticates Ubuntu packages and never returns PASS. It reads
files and dpkg's database; no import, bytecode execution, ldd, apt or repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

MAX_FILE = 512 * 1024 * 1024
SAFE_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


class ProvenanceError(ValueError):
    """A fixed refusal code, without host file contents."""


def _json(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ProvenanceError("DUPLICATE_JSON_KEY")
            value[key] = item
        return value

    value = json.loads(raw, object_pairs_hook=unique)
    if type(value) is not dict:
        raise ProvenanceError("MANIFEST_OBJECT_REQUIRED")
    return value


def _read(path: Path, limit: int) -> tuple[bytes, os.stat_result]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise ProvenanceError("REGULAR_BOUNDED_FILE_REQUIRED")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
        )
        if len(raw) > limit or any(
            getattr(before, field) != getattr(info, field)
            for info in (after, path.lstat())
            for field in fields
        ):
            raise ProvenanceError("OBJECT_CHANGED_DURING_READ")
        return raw, before
    finally:
        os.close(fd)


def load_inventory(path: Path, verified_digest: str) -> dict[str, dict[str, Any]]:
    if re.fullmatch(r"[0-9a-f]{64}", verified_digest) is None:
        raise ProvenanceError("VERIFIED_DIGEST_REQUIRED")
    raw, _ = _read(path, 32 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != verified_digest:
        raise ProvenanceError("VERIFIED_MANIFEST_DIGEST_MISMATCH")
    inventory = _json(raw).get("external_files")
    if type(inventory) is not dict or len(inventory) > 100000:
        raise ProvenanceError("EXTERNAL_INVENTORY_REQUIRED")
    for name, record in inventory.items():
        if (
            type(name) is not str
            or re.fullmatch(r"/[A-Za-z0-9_./+@-]+", name) is None
            or str(PurePosixPath(name)) != name
            or ".." in PurePosixPath(name).parts
            or type(record) is not dict
            or record.get("kind") not in {"file", "directory", "symlink"}
        ):
            raise ProvenanceError("EXTERNAL_RECORD_INVALID")
        fields = {"kind", "mode"}
        if record["kind"] == "file":
            fields |= {"size", "sha256"}
            if (
                type(record.get("size")) is not int
                or not 0 <= record["size"] <= MAX_FILE
                or type(record.get("sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            ):
                raise ProvenanceError("EXTERNAL_RECORD_INVALID")
        elif record["kind"] == "symlink":
            fields.add("target")
            if type(record.get("target")) is not str or not record["target"]:
                raise ProvenanceError("EXTERNAL_RECORD_INVALID")
        if (
            set(record) != fields
            or type(record.get("mode")) is not int
            or not 0 <= record["mode"] <= 0o777
        ):
            raise ProvenanceError("EXTERNAL_RECORD_INVALID")
    return inventory


def pyc_source(name: str) -> str | None:
    match = re.fullmatch(r"(.+)/__pycache__/(.+)\.cpython-312(?:\.opt-[12])?\.pyc", name)
    return f"{match[1]}/{match[2]}.py" if match else None


def observe(path: Path) -> dict[str, Any]:
    info = path.lstat()
    result: dict[str, Any] = {
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
    }
    if stat.S_ISLNK(info.st_mode):
        result.update(kind="symlink", target=os.readlink(path))
    elif stat.S_ISDIR(info.st_mode):
        result.update(kind="directory")
    elif stat.S_ISREG(info.st_mode):
        raw, info = _read(path, MAX_FILE)
        result.update(uid=info.st_uid, gid=info.st_gid, mode=stat.S_IMODE(info.st_mode))
        result.update(kind="file", size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        result["native_elf"] = raw.startswith(b"\x7fELF")
        if path.suffix == ".pyc":
            result["pyc_header_hex"] = raw[:16].hex()
            result["pyc_source_candidate"] = pyc_source(str(path))
            result["pyc_provenance"] = "UNRESOLVED_GENERATED_CODE"
    else:
        result.update(kind="special")
    final = path.lstat()
    if any(
        getattr(info, field) != getattr(final, field)
        for field in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_ctime_ns")
    ):
        raise ProvenanceError("OBJECT_CHANGED_DURING_READ")
    return result


def _dpkg(arguments: list[str], allowed: set[int]) -> str:
    result = subprocess.run(
        ["/usr/bin/dpkg-query", *arguments],
        env=SAFE_ENV,
        cwd="/",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if result.returncode not in allowed or len(result.stdout) > 16 * 1024 * 1024:
        raise ProvenanceError("DPKG_QUERY_FAILED")
    return result.stdout.decode("utf-8", errors="strict")


def package_observations(paths: list[str]) -> dict[str, Any]:
    # dpkg is an observation, not an authenticated package origin. In particular,
    # a local .list or status database is not a substitute for signed InRelease.
    versions = _dpkg(
        [
            "--show",
            "--showformat=${binary:Package}\t${Version}\t${Architecture}\t${db:Status-Abbrev}\n",
        ],
        {0},
    )
    search = []
    for start in range(0, len(paths), 128):
        search.append(_dpkg(["--search", "--", *paths[start : start + 128]], {0, 1}))
    return {"versions_tsv": versions, "ownership_observations": search, "authenticated": False}


def collect(inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    unresolved = []
    for name, expected in sorted(inventory.items()):
        try:
            actual = observe(Path(name))
            matches = all(actual.get(key) == value for key, value in expected.items())
            row = {"path": name, "expected": expected, "observed": actual, "matches": matches}
        except (OSError, ValueError):
            row = {"path": name, "expected": expected, "matches": False, "error": "READ_FAILED"}
        rows.append(row)
        if not row["matches"]:
            unresolved.append(name)
    pyc = [name for name in inventory if name.endswith(".pyc")]
    package_paths = sorted(set(inventory) | {source for p in pyc if (source := pyc_source(p))})
    packages = package_observations(package_paths)
    return {
        "schema": 1,
        "status": "FAIL" if unresolved else "BLOCKED",
        "external_count": len(rows),
        "pyc_count": len(pyc),
        "mismatched_or_unreadable": unresolved,
        "rows": rows,
        "dpkg": packages,
        "remaining": [
            "Authenticated Ubuntu InRelease -> Packages -> exact .deb -> every payload object",
            "Byte provenance of generated pyc and the audit's 19 unresolved files",
            "Observed native loader/libc/OpenSSL/provider/NSS/dlopen dependency closure",
        ],
        "s3": "BLOCKED",
        "production": "NO_GO",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verified-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        inventory = load_inventory(args.manifest, args.verified_manifest_sha256)
        report = collect(inventory)
        if report["external_count"] != 1352 or report["pyc_count"] != 573:
            report["status"] = "FAIL"
            report["audit_count_mismatch"] = True
    except (OSError, ValueError, subprocess.SubprocessError):
        print(json.dumps({"status": "FAIL", "reason": "COLLECTION_REFUSED"}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 1 if report["status"] == "FAIL" else 2


if __name__ == "__main__":
    sys.exit(main())

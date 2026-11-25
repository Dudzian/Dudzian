"""Collect supplemental static evidence for the Stage 9 V5d8 physical host.

Run with /usr/bin/python3.12 -I -S -B. The archive contains measurements only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import runpy
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, cast

REVISION = "e097a4de25fedb09ba47706914dfc332b8a1f03e"
BASELINE = "8c51cdbe12613b42ba7bc8de56eb9f5d19b9fb19f4e9800e6715d148bcb31b8e"
PROPOSAL = "7f5f6e7f885a368ccdbb844b11534ad7d09871eeb2d8b45484a6bf8fa95aacc4"
ABI = "stage9_v5d4_abi.py"
CLOSURE = "stage9_v5d4_closure.py"
HELPER_PINS = {
    ABI: "d684845d3802624e13356b55a47b301e46b6bdd7db9c56c8235948224d7fd89a",
    CLOSURE: "28f421e2bb5151f436cd0c26f71cc59b4c1f1aa5a4a8e5263cee97ff1c744b6f",
}
PYTHON = "/usr/bin/python3.12"
QEMU_IMG = "/usr/bin/qemu-img"
HEADER = "/usr/include/python3.12/patchlevel.h"
SYSCONFIG = "/usr/lib/python3.12/_sysconfigdata__x86_64-linux-gnu.py"
CACHE = Path("/etc/ld.so.cache")
READELF = "/usr/bin/readelf"
LDCONFIG = "/usr/sbin/ldconfig"
DPKG_QUERY = "/usr/bin/dpkg-query"
GIT = "/usr/bin/git"
ENV = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
    "LANG": "C",
    "DEBUGINFOD_URLS": "",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
}
DEFAULT_DIRS = tuple(
    Path(directory)
    for directory in (
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/lib64",
        "/usr/lib64",
        "/lib",
        "/usr/lib",
    )
)
SYSTEM_ROOTS = tuple(Path(root) for root in ("/usr", "/lib", "/lib64", "/bin", "/sbin"))
FILE_LIMIT = 512 * 1024 * 1024
OUTPUT_LIMIT = 64 * 1024 * 1024
GRAPH_LIMIT = 4096
CACHE_HEADER = re.compile(r"^(\d+) libs? found in cache ")
CACHE_ROW = re.compile(r"^\s*(\S+)\s+\(([^()]*)\)\s+=>\s+(/.+)$")
ORIGIN = re.compile(r"\$\{ORIGIN\}|\$ORIGIN(?![A-Za-z0-9_])")
TAG = re.compile(r"\((RPATH|RUNPATH)\).*\[([^\]]*)\]")


class Refusal(ValueError):
    """A static evidence input or collection step could not be verified."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def regular(path: Path, limit: int = FILE_LIMIT) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= limit:
        raise Refusal(f"REGULAR_FILE_REQUIRED {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if identity(opened) != identity(before):
            raise Refusal(f"FILE_CHANGED {path}")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (
        len(raw) != before.st_size
        or identity(after) != identity(before)
        or identity(path.lstat()) != identity(before)
    ):
        raise Refusal(f"FILE_CHANGED {path}")
    return raw, before


def identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def read_baseline(archive: Path) -> dict[str, Any]:
    raw, _ = regular(archive, 16 * 1024 * 1024)
    if sha(raw) != BASELINE:
        raise Refusal("PHYSICAL_V5D7_ARCHIVE_MISMATCH")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as source:
        report = source.extractfile("wheelhouse-evidence.json")
        proposal = source.extractfile("v5d7-proposal.json")
        if report is None or proposal is None or sha(proposal.read()) != PROPOSAL:
            raise Refusal("PHYSICAL_V5D7_PROPOSAL_MISMATCH")
        evidence: dict[str, Any] = json.load(report)["native_inventory"]
    return evidence


def parse_cache(stdout: str) -> dict[str, list[dict[str, str]]]:
    headers = [int(match[1]) for line in stdout.splitlines() if (match := CACHE_HEADER.match(line))]
    decoded: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in stdout.splitlines():
        if match := CACHE_ROW.fullmatch(line):
            name, abi, path = match.groups()
            decoded[name].append({"path": path, "abi_as_listed": abi})
    if headers != [sum(len(rows) for rows in decoded.values())]:
        raise Refusal("LOADER_CACHE_COUNT_MISMATCH")
    return dict(decoded)


def directories(search: list[str], aliases: set[str]) -> tuple[set[Path], set[str]]:
    supported: set[Path] = set()
    unresolved: set[str] = set()
    for entry in search:
        for alias in aliases:
            expanded = ORIGIN.sub(lambda _: str(Path(alias).parent), entry)
            if expanded.startswith("/") and "$" not in expanded:
                supported.add(Path(os.path.normpath(expanded)))
            else:
                unresolved.add(entry)
    return supported, unresolved


class Collection:
    def __init__(self, contents: Path, scratch: Path) -> None:
        self.contents = contents
        self.scratch = scratch
        self.steps: dict[str, dict[str, Any]] = {}
        self.issues: list[str] = []
        self.stderr: list[str] = []
        self.measured: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, str] = {}
        self.fingerprints: dict[str, tuple[int, ...]] = {}
        self.cache: dict[str, list[dict[str, str]]] = {}
        self.report: dict[str, Any] = {}

    def store(self, name: str, raw: bytes) -> None:
        path = self.contents / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)

    def json(self, name: str, value: Any) -> None:
        self.store(name, canonical(value) + b"\n")

    def problem(self, message: str) -> None:
        self.issues.append(message)
        self.stderr.append(message)

    def command(
        self,
        name: str,
        args: list[str],
        *,
        timeout: int = 45,
        outfile: Path | None = None,
        save_stdout: str | None = None,
    ) -> tuple[int, bytes]:
        stdout = b""
        stderr = b""
        with (
            tempfile.TemporaryFile(dir=self.scratch) as captured,
            tempfile.TemporaryFile(dir=self.scratch) as errors,
        ):
            try:
                result = subprocess.run(
                    args,
                    cwd=self.scratch,
                    env=ENV,
                    stdin=subprocess.DEVNULL,
                    stdout=captured,
                    stderr=errors,
                    timeout=timeout,
                    check=False,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
                stderr = b"COLLECTION_COMMAND_TIMEOUT\n"
            except OSError as exc:
                code = 125
                stderr = f"COLLECTION_COMMAND_FAILED {exc}\n".encode()
            output_limit = FILE_LIMIT if outfile else OUTPUT_LIMIT
            output_size = captured.tell()
            if output_size > output_limit or errors.tell() > OUTPUT_LIMIT:
                code = 125
                stderr += b"COLLECTION_COMMAND_OUTPUT_LIMIT\n"
            else:
                errors.seek(0)
                stderr += errors.read()
                captured.seek(0)
                if outfile is not None and code == 0:
                    with outfile.open("xb") as stream:
                        while chunk := captured.read(1024 * 1024):
                            stream.write(chunk)
                else:
                    stdout = captured.read(OUTPUT_LIMIT)
        self.steps[name] = {"argv": args, "exit_code": code, "stdout_bytes": output_size}
        if save_stdout is not None:
            self.store(save_stdout, stdout)
        if stderr:
            self.store(f"commands/{name}.stderr.txt", stderr)
            self.stderr.append(f"[{name}] {stderr.decode(errors='replace').rstrip()}")
        return code, stdout

    def collect_source(
        self, repository: Path, expected_head: str
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        git = [GIT, "-C", str(repository), "--no-optional-locks"]
        code, stdout = self.command("git-head", [*git, "rev-parse", "--verify", "HEAD^{commit}"])
        head = stdout.decode().strip()
        self.store("revision.txt", (head + "\n").encode())
        self.report["expected_head"] = expected_head
        self.report["observed_head"] = head
        if code or head != expected_head:
            raise Refusal("SOURCE_HEAD_MISMATCH")
        export = self.scratch / "source.tar"
        code, _ = self.command(
            "git-source",
            [*git, "archive", "--format=tar", head],
            timeout=120,
            outfile=export,
        )
        if code:
            raise Refusal("SOURCE_ARCHIVE_FAILED")
        snapshot = self.scratch / "source"
        snapshot.mkdir()
        with tarfile.open(export, mode="r:") as source:
            source.extractall(snapshot, filter="data")
        helpers = {}
        for name, expected in HELPER_PINS.items():
            path = snapshot / "scripts" / name
            if sha(regular(path)[0]) != expected:
                raise Refusal(f"STATIC_HELPER_SHA256_MISMATCH {name}")
            helpers[name] = runpy.run_path(str(path))
        closure: dict[str, Any] = helpers[CLOSURE]["inventory"](snapshot)
        self.json("source-closure.json", closure)
        expected_files = {item["path"]: item["sha256"] for item in closure["sources"]}
        expected_files["pyproject.toml"] = closure["installation_contract"]["pyproject_sha256"]
        observed = {}
        for relative, expected in expected_files.items():
            path = repository / relative
            try:
                if path.resolve(strict=True) != path or sha(regular(path)[0]) != expected:
                    raise Refusal(f"CHECKOUT_SOURCE_CHANGED {relative}")
                observed[relative] = expected
            except (OSError, RuntimeError, Refusal) as exc:
                self.problem(str(exc))
        self.json(
            "source-candidates.json",
            {
                "code": {item["path"]: item["sha256"] for item in closure["sources"]},
                "checkout_matches_snapshot": observed,
                "dynamic_review_payload_sha256": sha(canonical(closure)),
                "helper_sha256": HELPER_PINS,
                "source_head": head,
            },
        )
        if not closure["static_graph_closed"]:
            self.problem("SOURCE_CLOSURE_INCOMPLETE")
        return head, helpers[CLOSURE], helpers[ABI]

    def measure(self, alias: str, abi: dict[str, Any]) -> str:
        path = Path(alias)
        if not path.is_absolute() or str(path) != os.path.normpath(alias):
            raise Refusal(f"NON_ABSOLUTE_OR_NONCANONICAL_FILE {alias}")
        real = path.resolve(strict=True)
        if not any(real.is_relative_to(root) for root in SYSTEM_ROOTS):
            raise Refusal(f"SYSTEM_FILE_ROOT_REJECTED {alias}")
        key = str(real)
        if key in self.measured:
            if (
                identity(real.lstat()) != self.fingerprints[key]
                or path.resolve(strict=True) != real
            ):
                raise Refusal(f"FILE_CHANGED {alias}")
            self.aliases[alias] = key
            return key
        if len(self.measured) >= GRAPH_LIMIT:
            raise Refusal("HOST_FILE_LIMIT")
        data, metadata = regular(real)
        if path.resolve(strict=True) != real:
            raise Refusal(f"ALIAS_CHANGED {alias}")
        digest = sha(data)
        row: dict[str, Any] = {
            "sha256": digest,
            "size": len(data),
            "mode": metadata.st_mode,
            "link_count": metadata.st_nlink,
            "format": "ELF" if data.startswith(b"\x7fELF") else "REGULAR_FILE",
        }
        if row["format"] == "ELF":
            self.measure_elf(data, digest, row, abi)
        self.measured[key] = row
        self.aliases[alias] = key
        self.fingerprints[key] = identity(metadata)
        return key

    def measure_elf(
        self, data: bytes, digest: str, row: dict[str, Any], abi: dict[str, Any]
    ) -> None:
        name = f"elf/{digest}.readelf.txt"
        if not (self.contents / name).exists():
            with tempfile.TemporaryDirectory(prefix="elf-", dir=self.scratch) as temporary:
                specimen = Path(temporary) / "input.elf"
                specimen.write_bytes(data)
                code, output = self.command(
                    f"readelf-{digest}",
                    [
                        READELF,
                        "--file-header",
                        "--program-headers",
                        "--dynamic",
                        "--version-info",
                        "--dyn-syms",
                        "--relocs",
                        "--notes",
                        "--wide",
                        "--",
                        str(specimen),
                    ],
                    save_stdout=name,
                )
            if code:
                raise Refusal(f"READELF_FAILED {digest}")
        else:
            if self.steps[f"readelf-{digest}"]["exit_code"]:
                raise Refusal(f"READELF_FAILED {digest}")
            output = (self.contents / name).read_bytes()
        text = output.decode("utf-8")
        record = abi["parse_elf"](text)
        tags: dict[str, list[str]] = defaultdict(list)
        for tag, search in TAG.findall(text):
            tags[tag].extend(search.split(":"))
        row["elf"] = {
            "class": record.elf_class,
            "endianness": record.endian,
            "machine": record.machine,
            "type": record.kind,
            "interpreter": record.interpreter,
            "needed": record.needed,
            "soname": record.soname,
            "rpath": tags["RPATH"],
            "runpath": tags["RUNPATH"],
            "version_needs": record.version_needs,
            "version_definitions": record.version_definitions,
            "defined_python_symbols": record.defined_python_symbols,
            "undefined_python_symbols": record.undefined_python_symbols,
            "raw_static_evidence": name,
        }

    def load_cache(self) -> None:
        try:
            before = sha(regular(CACHE, OUTPUT_LIMIT)[0])
            code, output = self.command(
                "ldconfig-print", [LDCONFIG, "-p"], save_stdout="ldconfig.txt"
            )
            if code or sha(regular(CACHE, OUTPUT_LIMIT)[0]) != before:
                raise Refusal("LOADER_CACHE_CAPTURE_FAILED_OR_CHANGED")
            self.report["ld_so_cache_sha256"] = before
            self.cache = parse_cache(output.decode("utf-8"))
        except (OSError, UnicodeError, Refusal) as exc:
            self.problem(str(exc))

    def candidates(self, name: str, paths: set[Path]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = defaultdict(list)
        if name.startswith("/"):
            return {name: ["ABSOLUTE_REFERENCE"]}
        if "/" in name or "$" in name or name in {"", ".", ".."}:
            return {}
        for item in self.cache.get(name, []):
            found[item["path"]].append(f"CACHE {item['abi_as_listed']}")
        for directory in sorted(paths):
            path = directory / name
            if path.is_symlink() or path.exists():
                found[str(path)].append("DIRECTORY_CANDIDATE")
        return dict(found)

    def collect_graph(
        self,
        baseline: dict[str, Any],
        abi: dict[str, Any],
        seeds: set[str],
    ) -> None:
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        seen: set[tuple[str, frozenset[Path]]] = set()
        historic = baseline["host_abi_inputs"]
        aliases = {row["path"]: row["resolved_path"] for row in historic["paths"]}
        seeds.update(aliases)
        synthesis = baseline["dependency_synthesis"]
        unmatched = set(synthesis["unmatched_needed_names"])
        wheel_names = unmatched | set(synthesis["bundled_candidates_by_needed_name"])
        wheel_loaders = set(synthesis["program_interpreters"])
        wheels: dict[str, Any] = {}
        for name in sorted(wheel_names | wheel_loaders):
            found = self.candidates(name, set(DEFAULT_DIRS))
            wheels[name] = found
            seeds.update(found)
        pending: deque[tuple[str, frozenset[Path]]] = deque(
            (file, frozenset()) for file in sorted(seeds)
        )
        visited_paths: set[str] = set()
        unresolved_tokens: dict[str, list[str]] = {}
        while pending:
            alias, inherited = pending.popleft()
            context = (alias, inherited)
            if context in seen:
                continue
            if len(seen) >= GRAPH_LIMIT:
                self.problem("HOST_GRAPH_CONTEXT_LIMIT")
                break
            seen.add(context)
            visited_paths.add(alias)
            try:
                key = self.measure(alias, abi)
            except (OSError, RuntimeError, ValueError) as exc:
                self.problem(f"HOST_FILE_MEASUREMENT_FAILED {alias} {exc}")
                continue
            elf = self.measured[key].get("elf")
            if not elf:
                continue
            direct, unknown = directories(elf["runpath"] or elf["rpath"], {alias, key})
            if unknown:
                unresolved_tokens[alias] = sorted(unknown)
            inherited_next = (
                inherited | direct if elf["rpath"] and not elf["runpath"] else inherited
            )
            paths = set(DEFAULT_DIRS) | direct | inherited
            references = [(name, "DT_NEEDED") for name in elf["needed"]]
            if elf["interpreter"]:
                references.append((elf["interpreter"], "PT_INTERP"))
            for name, kind in references:
                found = self.candidates(name, paths)
                edge = edges.setdefault((alias, name, kind), {"candidates": {}})
                for candidate, origins in found.items():
                    current = edge["candidates"].get(candidate, [])
                    edge["candidates"][candidate] = sorted(set(current) | set(origins))
                    if (candidate, inherited_next) not in seen:
                        pending.append((candidate, inherited_next))
        freshness = {}
        for alias, expected_path in sorted(aliases.items()):
            measured_path = self.aliases.get(alias)
            expected_sha = historic["files_by_resolved_path"][expected_path]["sha256"]
            observed_sha = self.measured.get(measured_path or "", {}).get("sha256")
            match = measured_path == expected_path and observed_sha == expected_sha
            freshness[alias] = {
                "baseline_path": expected_path,
                "observed_path": measured_path,
                "baseline_sha256": expected_sha,
                "observed_sha256": observed_sha,
                "matches": match,
            }
            if not match:
                self.problem(f"PHYSICAL_BASELINE_CHANGED {alias}")
        self.json("v5d7-freshness.json", freshness)
        for alias, key in sorted(self.aliases.items()):
            try:
                if (
                    str(Path(alias).resolve(strict=True)) != key
                    or identity(Path(key).lstat()) != self.fingerprints[key]
                ):
                    raise Refusal(f"ALIAS_OR_FILE_CHANGED {alias}")
            except (OSError, RuntimeError, Refusal) as exc:
                self.problem(str(exc))
        graph = []
        for (consumer, name, kind), edge in sorted(edges.items()):
            decoded = []
            for candidate, origins in sorted(edge["candidates"].items()):
                provider = self.aliases.get(candidate)
                elf = self.measured.get(provider or "", {}).get("elf", {})
                decoded.append(
                    {
                        "alias": candidate,
                        "resolved_path": provider,
                        "origins": origins,
                        "elf64_x86_64": elf.get("class") == "ELF64"
                        and elf.get("machine") == "Advanced Micro Devices X86-64",
                        "soname": elf.get("soname"),
                        "soname_matches": elf.get("soname") == name
                        if kind == "DT_NEEDED"
                        else None,
                    }
                )
            if not any(item["elf64_x86_64"] for item in decoded):
                self.problem(f"HOST_DEPENDENCY_NO_MEASURED_X86_64_CANDIDATE {consumer} {name}")
            graph.append(
                {"consumer": consumer, "lookup": name, "kind": kind, "candidates": decoded}
            )
        for alias, tokens in sorted(unresolved_tokens.items()):
            self.problem(f"HOST_SEARCH_PATH_UNRESOLVED {alias} {tokens}")
        self.json(
            "system-graph.json",
            {
                "scope": "STATIC_CANDIDATE_SUPERSET_NOT_LOADER_ORDER",
                "wheel_system_queries": wheels,
                "wheel_system_queries_without_candidate": sorted(
                    name for name, rows in wheels.items() if name in unmatched and not rows
                ),
                "visited_paths": sorted(visited_paths),
                "aliases": self.aliases,
                "files_by_resolved_path": self.measured,
                "edges": graph,
                "unresolved_search_tokens": unresolved_tokens,
                "loader_resolution": "NOT_TESTED",
                "native_load": "NOT_EXECUTED",
            },
        )

    def collect_host(self, closure: dict[str, Any], abi: dict[str, Any]) -> set[str]:
        seeds = {PYTHON, QEMU_IMG}
        choices: dict[str, bool] = {}
        for entry in sys.path:
            path = Path(entry)
            if not path.is_absolute() or not any(
                path.is_relative_to(root) for root in SYSTEM_ROOTS
            ):
                self.problem(f"BASE_IMPORT_PATH_REJECTED {entry}")
                continue
            choices[entry] = path.exists()
        roots: list[Path] = []
        for entry in sorted(key for key, exists in choices.items() if exists):
            path = Path(entry)
            if not any(path.is_relative_to(root) for root in roots):
                roots.append(path)
        if roots:
            try:
                base: dict[str, Any] = closure["base_runtime_inventory"](roots)
                self.json("base-runtime.json", base)
                seeds.update(base["native"])
            except (OSError, RuntimeError, ValueError) as exc:
                self.problem(f"BASE_RUNTIME_INVENTORY_FAILED {exc}")
        else:
            self.problem("BASE_RUNTIME_ROOTS_MISSING")
        self.json("base-import-paths.json", choices)
        inputs = {}
        for name, alias in {
            "interpreter": PYTHON,
            "patchlevel_header": HEADER,
            "sysconfigdata": SYSCONFIG,
            "qemu_img": QEMU_IMG,
        }.items():
            try:
                if name in {"patchlevel_header", "sysconfigdata"} and Path(alias).resolve(
                    strict=True
                ) != Path(alias):
                    raise Refusal("ABI_INPUT_SYMLINK_REJECTED")
                selected = self.measure(alias, abi)
                row = self.measured[selected]
                if name in {"interpreter", "qemu_img"} and row["format"] != "ELF":
                    raise Refusal("ABI_ELF_REQUIRED")
                if name == "qemu_img" and (alias != selected or row["link_count"] != 1):
                    self.problem("QEMU_IMG_PATH_NOT_COMPATIBLE_WITH_BUILDER")
                inputs[name] = {"path": selected, "sha256": row["sha256"], "size": row["size"]}
            except (OSError, RuntimeError, ValueError) as exc:
                self.problem(f"HOST_INPUT_UNAVAILABLE {name} {exc}")
        python_contract = "MISSING_INPUT"
        if {"patchlevel_header", "sysconfigdata"} <= inputs.keys():
            try:
                abi["_python_contract"](
                    {name: row["path"] for name, row in inputs.items()},
                    {row["path"]: row["sha256"] for row in inputs.values()},
                    {},
                )
                python_contract = "STATIC_FIELDS_MATCH"
            except (OSError, ValueError, SyntaxError) as exc:
                python_contract = "STATIC_FIELDS_REJECTED"
                self.problem(f"HOST_PYTHON_ABI_INPUTS_REJECTED {exc}")
        release = platform.freedesktop_os_release()
        host = {
            "python_full_version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "system": platform.system(),
            "soabi": "cpython-312-x86_64-linux-gnu"
            if python_contract == "STATIC_FIELDS_MATCH"
            else None,
            "interpreter_sha256": inputs.get("interpreter", {}).get("sha256"),
            "distribution": release.get("ID"),
            "distribution_version": release.get("VERSION_ID"),
            "glibc": os.confstr("CS_GNU_LIBC_VERSION"),
        }
        if (
            host["python_full_version"] != "3.12.3"
            or host["implementation"] != "CPython"
            or host["machine"] != "x86_64"
            or host["system"] != "Linux"
            or host["distribution"] != "ubuntu"
            or host["distribution_version"] != "24.04"
            or host["glibc"] != "glibc 2.39"
        ):
            self.problem("PHYSICAL_HOST_PROFILE_MISMATCH")
        self.json(
            "host-candidates.json",
            {"host": host, "abi_inputs": inputs, "python_contract": python_contract},
        )
        return seeds

    def collect_provenance(self, abi: dict[str, Any]) -> None:
        tools = {}
        for alias in (GIT, READELF, LDCONFIG, DPKG_QUERY):
            try:
                path = self.measure(alias, abi)
                tools[alias] = {"path": path, "sha256": self.measured[path]["sha256"]}
            except (OSError, RuntimeError, ValueError) as exc:
                self.problem(f"COLLECTION_TOOL_UNAVAILABLE {alias} {exc}")
        self.json("collection-tools.json", tools)
        names = sorted(self.aliases.keys() | self.measured.keys())
        owner_lookup: dict[str, list[str]] = defaultdict(list)
        package_pattern = re.compile(r"[a-z0-9][a-z0-9+.-]*(?::[a-z0-9-]+)?")
        # A missing owner (for example /usr-merge aliases) makes dpkg-query -S exit 1.
        # Preserve that output and stderr even when other paths in the batch have owners.
        for start in range(0, len(names), 64):
            group = names[start : start + 64]
            label = f"dpkg-owners-{start // 64}"
            code, output = self.command(
                label, [DPKG_QUERY, "--search", "--", *group], save_stdout=f"commands/{label}.txt"
            )
            if code not in {0, 1}:
                self.problem(f"DPKG_OWNER_LOOKUP_FAILED {label}")
            for line in output.decode(errors="replace").splitlines():
                owners, sep, path = line.partition(": ")
                if sep and path in group:
                    owner_lookup[path].extend(
                        name for name in owners.split(", ") if package_pattern.fullmatch(name)
                    )
        installed = sorted({name for owners in owner_lookup.values() for name in owners})
        if installed:
            code, _ = self.command(
                "dpkg-packages",
                [
                    DPKG_QUERY,
                    "--show",
                    "--showformat=${binary:Package}\t${Version}\t${Architecture}\t${db:Status-Abbrev}\n",
                    "--",
                    *installed,
                ],
                save_stdout="dpkg-packages.tsv",
            )
            if code:
                self.problem("DPKG_PACKAGE_LOOKUP_FAILED")
        aliases_by_file: dict[str, set[str]] = defaultdict(set)
        for alias, path in self.aliases.items():
            aliases_by_file[path].add(alias)
        provenance = {}
        for path in self.measured:
            aliases = aliases_by_file[path] | {path}
            provenance[path] = {alias: owner_lookup.get(alias, []) for alias in sorted(aliases)}
        self.json("dpkg-file-owners.json", provenance)

    def finish(self) -> int:
        code = 2 if self.issues else 0
        self.report.update(
            profile="stage9-v5d8-physical-static-evidence-v1",
            authorization="NOT_AUTHORIZED",
            STAGE_9="IN_PROGRESS",
            OBSERVER_3A="BLOCKED",
            STAGE_3B="NOT_AUTHORIZED",
            PRODUCTION="NO-GO",
            physical_v5d7_sha256=BASELINE,
            physical_proposal_sha256=PROPOSAL,
            collector_sha256=sha(regular(Path(__file__).resolve())[0]),
            collection_status="PARTIAL" if code else "MEASURED",
            abi_compatibility="NOT_TESTED",
            loader_resolution="NOT_TESTED",
            native_load="NOT_EXECUTED",
            issues=self.issues,
        )
        self.json("collection.json", self.report)
        self.json("exit-codes.json", self.steps)
        self.store("exit-code.txt", f"{code}\n".encode())
        self.store("stderr.txt", ("\n".join(self.stderr) + "\n").encode() if self.stderr else b"")
        return code


def archive(contents: Path, temporary: Path, output: Path) -> str:
    paths = sorted(
        path.relative_to(contents).as_posix() for path in contents.rglob("*") if path.is_file()
    )
    expected = {name: sha(regular(contents / name)[0]) for name in paths}
    checksums = "".join(f"{digest}  {name}\n" for name, digest in expected.items()).encode()
    with (
        temporary.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as compressed,
    ):
        os.fchmod(raw.fileno(), 0o600)
        with tarfile.open(fileobj=cast(BinaryIO, compressed), mode="w|") as target:
            for name in ["SHA256SUMS", *paths]:
                data = checksums if name == "SHA256SUMS" else regular(contents / name)[0]
                info = tarfile.TarInfo(name)
                info.mode = 0o600
                info.size = len(data)
                target.addfile(info, io.BytesIO(data))
    with tarfile.open(temporary, mode="r|gz") as packaged:
        seen = set()
        for info in packaged:
            if info.name in seen or not info.isfile():
                raise Refusal("ARCHIVE_MEMBER_MISMATCH")
            seen.add(info.name)
            stream = packaged.extractfile(info)
            if stream is None:
                raise Refusal("ARCHIVE_MEMBER_MISMATCH")
            verified = sha(stream.read())
            digest = sha(checksums) if info.name == "SHA256SUMS" else expected.get(info.name)
            if verified != digest:
                raise Refusal("ARCHIVE_SHA256_MISMATCH")
        if seen != {"SHA256SUMS", *paths}:
            raise Refusal("ARCHIVE_MEMBER_SET_MISMATCH")
    with temporary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    os.link(temporary, output)
    return digest


def collect(repository: Path, baseline: Path, output: Path, expected_head: str) -> int:
    with tempfile.TemporaryDirectory(prefix="stage9-v5d8-", dir=output.parent) as directory:
        scratch = Path(directory)
        contents = scratch / "contents"
        contents.mkdir()
        result = Collection(contents, scratch)
        try:
            head, closure, abi = result.collect_source(repository, expected_head)
            result.report["source_head"] = head
            physical = read_baseline(baseline)
            result.load_cache()
            seeds = result.collect_host(closure, abi)
            result.collect_graph(physical, abi, seeds)
            result.collect_provenance(abi)
        except (OSError, RuntimeError, ValueError, SyntaxError, KeyError, tarfile.TarError) as exc:
            result.problem(f"COLLECTION_REFUSED {exc}")
        code = result.finish()
        digest = archive(contents, scratch / "verified.tar.gz", output)
    print(f"{digest}  {output}")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--physical-v5d7", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-head", default=REVISION)
    args = parser.parse_args()
    try:
        if (
            sys.platform != "linux"
            or not sys.flags.isolated
            or not sys.flags.no_site
            or not sys.flags.dont_write_bytecode
            or Path(sys.executable) != Path(PYTHON)
            or os.geteuid() == 0
        ):
            raise Refusal("UNPRIVILEGED_SYSTEM_CPYTHON_REQUIRED_USE_I_S_B")
        if not re.fullmatch(r"[0-9a-f]{40}", args.expected_head):
            raise Refusal("EXPECTED_HEAD_SHA1_REQUIRED")
        repository = args.repository.resolve(strict=True)
        baseline = args.physical_v5d7.resolve(strict=True)
        output = args.output.absolute()
        if (
            output.name in {"", ".", ".."}
            or output.suffixes[-2:] != [".tar", ".gz"]
            or output.parent.resolve(strict=True) != output.parent
            or output.is_relative_to(repository)
            or output.exists()
            or output.is_symlink()
        ):
            raise Refusal("NEW_ARCHIVE_OUTSIDE_REPOSITORY_REQUIRED")
        return collect(repository, baseline, output, args.expected_head)
    except (OSError, RuntimeError, Refusal, tarfile.TarError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

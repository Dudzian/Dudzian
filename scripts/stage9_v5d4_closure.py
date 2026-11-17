"""Statyczny graf źródeł kontrolera; nie importuje analizowanego kodu.

Uruchom z przejrzanej kopii: python3 -I -S -B ten_plik --repository KATALOG.
Wynik opisuje krawędzie i materiał do review, nie przyznaje aprobaty instalacji.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tomllib
from typing import Any

DEFAULT_ROOTS = ("scripts/stage9_observer_kvm_preflight_v5d1.py",)
LIMIT = 8 * 1024 * 1024
BASE_FILE_LIMIT = 512 * 1024 * 1024
MECHANISMS = {
    "__import__",
    "exec",
    "eval",
    "compile",
    "import_module",
    "run_path",
    "run_module",
    "exec_module",
    "load_module",
    "spec_from_file_location",
    "spec_from_loader",
    "CDLL",
    "PyDLL",
    "LoadLibrary",
    "dlopen",
}


class Refusal(ValueError):
    """Niekompletny lub niebezpieczny materiał wejściowy."""


def base_runtime_inventory(roots: list[Path]) -> dict[str, Any]:
    """Konserwatywny wykaz bajtów base CPython, bez wykonywania jego kodu.

    Obejmuje cache bytecode, bo -B wyłącza tylko zapis. Katalogi site-packages
    i dist-packages są poza zakresem startu -S. Łańcuch do 16 dowiązań do
    regularnego pliku jest zapisywany ogniwo po ogniwie; pętle i symlink dirs
    odmawiają. Każdy cel poza roots również wymaga niezależnej aprobaty.
    """
    if not roots:
        raise Refusal("Brak korzeni bazowego runtime.")
    selected = sorted({Path(os.path.abspath(root)) for root in roots})
    files: dict[str, str] = {}
    native: set[str] = set()
    symlinks: dict[str, dict[str, str]] = {}
    excluded: set[str] = set()
    directories: dict[Path, tuple[int, ...]] = {}

    def fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def ancestors(path: Path) -> None:
        if any(parent.is_symlink() for parent in path.parents):
            raise Refusal("Dowiązanie w przodkach bazowego runtime.")

    def regular(path: Path) -> str:
        ancestors(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > BASE_FILE_LIMIT:
            raise Refusal("Niezwykły plik lub przekroczony limit runtime.")
        if not before.st_mode & 0o444:
            raise Refusal("Nieczytelny plik bazowego runtime.")
        if path.suffix == ".pth":
            raise Refusal("Plik .pth w bazowym runtime wymaga osobnego profilu.")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        checksum = hashlib.sha256()
        length = 0
        prefix = b""
        with os.fdopen(descriptor, "rb") as stream:
            if fingerprint(os.fstat(stream.fileno())) != fingerprint(before):
                raise Refusal("Zmiana pliku bazowego runtime przed odczytem.")
            while chunk := stream.read(1024 * 1024):
                if not prefix:
                    prefix = chunk[:4]
                checksum.update(chunk)
                length += len(chunk)
                if length > BASE_FILE_LIMIT:
                    raise Refusal("Przekroczony limit pliku runtime.")
            if fingerprint(os.fstat(stream.fileno())) != fingerprint(before):
                raise Refusal("Zmiana pliku bazowego runtime podczas odczytu.")
        if length != before.st_size or fingerprint(path.lstat()) != fingerprint(before):
            raise Refusal("Zmiana ścieżki bazowego runtime.")
        value = checksum.hexdigest()
        previous = files.get(str(path))
        if previous is not None and previous != value:
            raise Refusal("Niespójny ponowny odczyt bazowego runtime.")
        files[str(path)] = value
        if prefix == b"\x7fELF":
            native.add(str(path))
        return value

    def linked(path: Path) -> None:
        chain: list[tuple[Path, str, tuple[int, ...]]] = []
        seen: set[Path] = set()
        target = path
        while True:
            ancestors(target)
            metadata = target.lstat()
            if not stat.S_ISLNK(metadata.st_mode):
                if not stat.S_ISREG(metadata.st_mode):
                    raise Refusal("Dowiązanie do katalogu lub pliku specjalnego runtime.")
                break
            if target in seen:
                raise Refusal("Pętla dowiązań bazowego runtime.")
            if len(chain) == 16:
                raise Refusal("Przekroczony limit 16 dowiązań bazowego runtime.")
            if target.suffix == ".pth":
                raise Refusal("Dowiązanie .pth w bazowym runtime.")
            seen.add(target)
            raw_target = os.readlink(target)
            chain.append((target, raw_target, fingerprint(metadata)))
            target = Path(os.path.abspath(target.parent / raw_target))
        checksum = regular(target)
        for alias, raw_target, before in chain:
            if os.readlink(alias) != raw_target or fingerprint(alias.lstat()) != before:
                raise Refusal("Zmiana dowiązania bazowego runtime.")
            record = {"target": raw_target, "resolved": str(target), "sha256": checksum}
            previous = symlinks.get(str(alias))
            if previous is not None and previous != record:
                raise Refusal("Niespójny ponowny odczyt dowiązania runtime.")
            symlinks[str(alias)] = record

    pending = list(reversed(selected))
    while pending:
        path = pending.pop()
        ancestors(path)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            linked(path)
        elif stat.S_ISDIR(metadata.st_mode):
            if path.name in {"site-packages", "dist-packages"}:
                if path in selected:
                    raise Refusal("Korzeń site-packages nie jest bazowym runtime -S.")
                excluded.add(str(path))
                continue
            if not metadata.st_mode & 0o444 or not metadata.st_mode & 0o111:
                raise Refusal("Nieczytelny katalog bazowego runtime.")
            state = fingerprint(metadata)
            if path in directories and directories[path] != state:
                raise Refusal("Zmiana katalogu bazowego runtime.")
            directories[path] = state
            pending.extend(sorted(path.iterdir(), reverse=True))
        elif stat.S_ISREG(metadata.st_mode):
            regular(path)
        else:
            raise Refusal("Plik specjalny w bazowym runtime.")
        if len(files) + len(symlinks) > 100_000:
            raise Refusal("Przekroczony limit wykazu bazowego runtime.")
    if not files:
        raise Refusal("Brak bajtów bazowego runtime.")
    for path, before in directories.items():
        if fingerprint(path.lstat()) != before:
            raise Refusal("Zmiana drzewa bazowego runtime.")
    return {
        "profile": "stage9-v5d4-base-runtime-inventory-v1",
        "roots": [str(path) for path in selected],
        "files": dict(sorted(files.items())),
        "symlinks": dict(sorted(symlinks.items())),
        "native": sorted(native),
        "excluded": sorted(excluded),
        "scope": "STATIC_BYTES_SUPERSET_NOT_DYNAMIC_COMPLETENESS",
        "installation_approved": False,
    }


def read_source(repository: Path, relative: str) -> bytes:
    """Odczyt zwykłego pliku bez dowiązań w ścieżce i bez zapisu cache."""
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise Refusal("Niedozwolona ścieżka źródła.")
    candidate = repository
    for part in parts:
        candidate /= part
        if candidate.is_symlink():
            raise Refusal("Dowiązanie w źródłach.")
    if not candidate.is_file() or candidate.stat().st_size > LIMIT:
        raise Refusal("Brak źródła lub przekroczony limit.")
    return candidate.read_bytes()


def module_for(path: str) -> str:
    value = path.removesuffix(".py").replace("/", ".")
    return value.removesuffix(".__init__")


def candidate(repository: Path, module: str) -> str | None:
    if not module or any(not part.isidentifier() for part in module.split(".")):
        return None
    stem = module.replace(".", "/")
    for relative in (stem + ".py", stem + "/__init__.py"):
        path = repository / relative
        if path.is_symlink():
            raise Refusal("Dowiązanie modułu.")
        if path.is_file():
            return relative
    return None


def initializers(repository: Path, relative: str) -> list[str]:
    result = []
    for parent in reversed(Path(relative).parents):
        if parent == Path("."):
            continue
        value = (parent / "__init__.py").as_posix()
        if (repository / value).exists():
            read_source(repository, value)
            result.append(value)
    return result


def expression_name(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return expression_name(node.value, aliases) + "." + node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        if isinstance(node.slice.value, str):
            return expression_name(node.value, aliases) + "." + node.slice.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
        if len(node.args) >= 2:
            base = expression_name(node.args[0], aliases)
            attr = node.args[1]
            if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
                return base + "." + attr.value
            return base + ".<dynamic>"
    return "<expression>"


class SourceVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.module = module_for(path)
        self.package = (
            self.module if path.endswith("/__init__.py") else self.module.rpartition(".")[0]
        )
        self.deferred = 0
        self.aliases: dict[str, str] = {}
        self.imports: list[dict[str, Any]] = []
        self.dynamic: list[dict[str, Any]] = []

    def edge(self, node: ast.AST, module: str, members: list[str]) -> None:
        self.imports.append(
            {
                "source": self.path,
                "line": node.lineno if isinstance(node, (ast.Import, ast.ImportFrom)) else 0,
                "module": module,
                "members": members,
                "phase": "deferred" if self.deferred else "module_initialization",
            }
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name.split(".")[0]] = (
                alias.name if alias.asname else alias.name.split(".")[0]
            )
            self.edge(node, alias.name, [])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level:
            package = self.package.split(".") if self.package else []
            if node.level > len(package):
                raise Refusal("Nierozwiązany import względny.")
            module = ".".join(
                package[: len(package) - node.level + 1] + ([module] if module else [])
            )
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = module + "." + alias.name
        self.edge(node, module, [alias.name for alias in node.names])

    def visit_Assign(self, node: ast.Assign) -> None:
        name = expression_name(node.value, self.aliases)
        for target in node.targets:
            if isinstance(target, ast.Name) and name != "<expression>":
                self.aliases[target.id] = name
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Dekoratory, wartości domyślne i adnotacje mogą działać przy imporcie.
        for value in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
            if value is not None:
                self.visit(value)
        for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        self.deferred += 1
        for statement in node.body:
            self.visit(statement)
        self.deferred -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        name = expression_name(node.func, self.aliases)
        leaf = name.rsplit(".", 1)[-1]
        reflective = name.endswith(".<dynamic>") or (
            name in {"getattr", "builtins.getattr"}
            and node.args
            and expression_name(node.args[0], self.aliases).split(".")[0]
            in {"importlib", "builtins", "__builtins__", "ctypes", "runpy"}
        )
        mechanism = leaf in MECHANISMS
        if leaf == "compile":
            mechanism = name in {"compile", "builtins.compile", "__builtins__.compile"}
        if mechanism or reflective:
            target: str | None = None
            if leaf in {"__import__", "import_module"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    target = first.value
                    if target.startswith("."):
                        target = None  # import_module package= wymaga osobnej analizy.
            self.dynamic.append(
                {
                    "source": self.path,
                    "line": node.lineno,
                    "mechanism": name,
                    "target": target,
                    "resolved_literal_import": target is not None,
                    "phase": "deferred" if self.deferred else "module_initialization",
                }
            )
        self.generic_visit(node)


def inventory(repository: Path, roots: tuple[str, ...] = DEFAULT_ROOTS) -> dict[str, Any]:
    repository = Path(os.path.abspath(repository))
    if any(part.is_symlink() for part in (repository, *repository.parents)):
        raise Refusal("Dowiązanie w katalogu repozytorium.")
    pending = set(roots)
    collected: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    dynamic: list[dict[str, Any]] = []
    missing: list[str] = []
    local_roots = {
        path.name
        for path in repository.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    } | {Path(relative).parts[0] for relative in roots if "/" in relative}
    for relative in roots:
        pending.update(initializers(repository, relative))
    while pending:
        relative = min(pending)
        pending.remove(relative)
        if relative in collected:
            continue
        raw = read_source(repository, relative)
        try:
            syntax = ast.parse(raw, filename=relative)
        except (SyntaxError, ValueError) as exc:
            raise Refusal("Nieprawidłowe źródło Python.") from exc
        visitor = SourceVisitor(relative)
        visitor.visit(syntax)
        collected[relative] = {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
            "initializer": relative.endswith("/__init__.py"),
        }
        edges.extend(visitor.imports)
        dynamic.extend(visitor.dynamic)
        requests = [(edge["module"], edge["members"]) for edge in visitor.imports]
        requests += [(item["target"], []) for item in visitor.dynamic if item["target"]]
        for module, members in requests:
            found = candidate(repository, module)
            if found:
                pending.add(found)
                pending.update(initializers(repository, found))
                for member in members:
                    child = candidate(repository, module + "." + member)
                    if child:
                        pending.add(child)
                        pending.update(initializers(repository, child))
            elif module.split(".")[0] in local_roots:
                missing.append(module)
    project_raw = read_source(repository, "pyproject.toml")
    project = tomllib.loads(project_raw.decode("utf-8"))["project"]
    stdlib_roots = set(sys.stdlib_module_names) | {"__future__"}
    external = sorted(
        {
            edge["module"].split(".")[0]
            for edge in edges
            if not candidate(repository, edge["module"])
            and edge["module"].split(".")[0] not in stdlib_roots | local_roots
        }
    )
    unresolved = [item for item in dynamic if not item["resolved_literal_import"]]
    return {
        "profile": "stage9-v5d4-source-closure-v1",
        "roots": list(roots),
        "sources": [collected[key] for key in sorted(collected)],
        "static_edges": sorted(
            edges, key=lambda item: (item["source"], item["line"], item["module"])
        ),
        "dynamic_mechanisms": sorted(dynamic, key=lambda item: (item["source"], item["line"])),
        "missing_local_modules": sorted(set(missing)),
        "unresolved_dynamic_mechanisms": unresolved,
        "preflight_external_static_roots": external,
        "preflight_dynamic_targets": sorted({item["target"] for item in dynamic if item["target"]}),
        "stdlib_catalog": {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
        },
        "installation_contract": {
            "pyproject_sha256": hashlib.sha256(project_raw).hexdigest(),
            "requires_python": project["requires-python"],
            "project_dependencies": project["dependencies"],
            "observer_extra": project["optional-dependencies"]["observer-kvm-preflight"],
            "reduced_preflight_installation_authorized": False,
        },
        "static_graph_closed": not missing and not unresolved,
        "installation_approved": False,
        "remaining_evidence": [
            "Bajty zależności Python i ich importy przechodnie, wheel METADATA oraz native members.",
            "CPython 3.12, stdlib i rozszerzenia natywne oraz rekurencyjny graf ELF/loader.",
            "Jawne roots importów, site, .pth, customizers, bytecode i zewnętrzna akceptacja środowiska.",
            "AST nie dowodzi pełnej semantyki refleksji, aliasów przekazanych jako wartości ani warunków wykonania.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--repository", type=Path)
    modes.add_argument("--base-runtime-root", action="append", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = (
            base_runtime_inventory(args.base_runtime_root)
            if args.base_runtime_root
            else inventory(args.repository)
        )
        text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
        return 0 if result.get("static_graph_closed", True) else 2
    except (Refusal, OSError, KeyError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        print(f"Odmowa analizy źródeł: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

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
    To profil dowodowy POSIX; Windows wymaga osobnego projektu kontroli ACL
    i punktów ponownej analizy. Brak flag no-follow nigdy nie włącza fallbacku.
    """
    if os.name != "posix" or not all(
        isinstance(getattr(os, flag, None), int) and getattr(os, flag) > 0
        for flag in ("O_NOFOLLOW", "O_NONBLOCK")
    ):
        raise Refusal("BASE_RUNTIME_POSIX_CAPABILITY_REQUIRED")
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
        if node.id in aliases:
            return aliases[node.id]
        if node.id in {"exec", "eval", "compile", "__import__", "getattr", "__builtins__"}:
            return node.id
        return "<unresolved>." + node.id
    if isinstance(node, ast.Attribute):
        qualified = expression_name(node.value, aliases) + "." + node.attr
        return aliases.get("@" + qualified, qualified)
    if isinstance(node, ast.Subscript):
        base = expression_name(node.value, aliases)
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            return base + "." + node.slice.value
        if base.split(".")[0] in {"builtins", "__builtins__", "importlib", "ctypes", "runpy"}:
            return base + ".<dynamic>"
    if isinstance(node, ast.NamedExpr):
        return expression_name(node.value, aliases)
    if isinstance(node, ast.Call) and expression_name(node.func, aliases) in {
        "getattr",
        "builtins.getattr",
        "__builtins__.getattr",
    }:
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
        self.class_outer_aliases: dict[str, str] | None = None
        self.scope_rebound_names: set[str] = set()
        self.imports: list[dict[str, Any]] = []
        self.dynamic: list[dict[str, Any]] = []

    @staticmethod
    def rebound_names(node: ast.AST) -> set[str]:
        class Bindings(ast.NodeVisitor):
            def __init__(self) -> None:
                self.counts: dict[str, int] = {}

            def add(self, name: str) -> None:
                self.counts[name] = self.counts.get(name, 0) + 1

            def visit_Name(self, item: ast.Name) -> None:
                if isinstance(item.ctx, (ast.Store, ast.Del)):
                    self.add(item.id)

            def visit_Import(self, item: ast.Import) -> None:
                for alias in item.names:
                    self.add(alias.asname or alias.name.split(".")[0])

            def visit_ImportFrom(self, item: ast.ImportFrom) -> None:
                for alias in item.names:
                    self.add(alias.asname or alias.name)

            def visit_FunctionDef(self, item: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                self.add(item.name)

            def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
                self.visit_FunctionDef(item)

            def visit_ClassDef(self, item: ast.ClassDef) -> None:
                self.add(item.name)

            def visit_Attribute(self, item: ast.Attribute) -> None:
                if isinstance(item.ctx, (ast.Store, ast.Del)):
                    value: ast.AST = item.value
                    while isinstance(value, ast.Attribute):
                        value = value.value
                    if isinstance(value, ast.Name):
                        self.add(value.id)

        bindings = Bindings()
        bindings.visit(node)
        return {name for name, count in bindings.counts.items() if count > 1}

    def visit_Module(self, node: ast.Module) -> None:
        self.scope_rebound_names = self.rebound_names(node)
        self.generic_visit(node)

    def deferred_aliases(self) -> dict[str, str]:
        outer = self.class_outer_aliases if self.class_outer_aliases is not None else self.aliases
        result = dict(outer)
        for name in self.scope_rebound_names:
            previous = result.get(name, name)
            if previous == "re" or previous.rsplit(".", 1)[-1] in MECHANISMS:
                result[name] = "<unresolved>." + name + "." + previous.rsplit(".", 1)[-1]
        return result

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
        # RHS jest obliczane przed zastąpieniem poprzedniej wartości symbolu.
        self.visit(node.value)
        name = expression_name(node.value, self.aliases)
        for target in node.targets:
            self.bind(target, name)

    def bind(self, target: ast.AST, name: str = "<expression>") -> None:
        def binding(symbol: str, previous: str) -> str:
            if name == "<expression>" or name.startswith("<unresolved>."):
                leaf = previous.rsplit(".", 1)[-1]
                sensitive = symbol if symbol in MECHANISMS else leaf
                return (
                    "<unresolved>." + symbol + ("." + sensitive if sensitive in MECHANISMS else "")
                )
            return name

        if isinstance(target, ast.Name):
            self.aliases[target.id] = binding(target.id, self.aliases.get(target.id, target.id))
        elif isinstance(target, ast.Attribute):
            qualified = expression_name(target.value, self.aliases) + "." + target.attr
            self.aliases["@" + qualified] = binding(qualified, qualified)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self.bind(item)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
            self.bind(node.target, expression_name(node.value, self.aliases))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self.bind(node.target)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self.bind(target)

    def merge_aliases(self, states: list[dict[str, str]]) -> None:
        merged: dict[str, str] = {}
        for name in set().union(*(state.keys() for state in states)):
            values = {state.get(name, "<unresolved>." + name) for state in states}
            if len(values) == 1:
                merged[name] = values.pop()
            else:
                sensitive = sorted(
                    value.rsplit(".", 1)[-1]
                    for value in values
                    if value.rsplit(".", 1)[-1] in MECHANISMS
                )
                merged[name] = "<unresolved>." + name + ("." + sensitive[0] if sensitive else "")
        self.aliases = merged

    def branch(self, statements: list[ast.stmt], initial: dict[str, str]) -> dict[str, str]:
        self.aliases = dict(initial)
        for statement in statements:
            self.visit(statement)
        return dict(self.aliases)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        initial = dict(self.aliases)
        self.merge_aliases([self.branch(node.body, initial), self.branch(node.orelse, initial)])

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        initial = dict(self.aliases)
        self.visit(node.body)
        body = dict(self.aliases)
        self.aliases = initial
        self.visit(node.orelse)
        self.merge_aliases([body, dict(self.aliases)])

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        for index, value in enumerate(node.values):
            initial = dict(self.aliases)
            self.visit(value)
            if index:
                self.merge_aliases([initial, dict(self.aliases)])

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        initial = dict(self.aliases)
        body = self.branch(node.body, initial)
        self.merge_aliases([initial, body, self.branch(node.orelse, body)])

    def visit_Try(self, node: ast.Try | ast.TryStar) -> None:
        initial = dict(self.aliases)
        body = self.branch(node.body, initial)
        states = [initial, body, self.branch(node.orelse, body)]
        for handler in node.handlers:
            self.aliases = dict(initial)
            self.visit(handler)
            states.append(dict(self.aliases))
        self.merge_aliases(states)
        for statement in node.finalbody:
            self.visit(statement)

    visit_TryStar = visit_Try

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.bind(node.target, expression_name(node.value, self.aliases))

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        initial = dict(self.aliases)
        self.bind(node.target)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)
        self.merge_aliases([initial, dict(self.aliases)])

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.bind(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self.bind(ast.Name(id=node.name, ctx=ast.Store()))
        for statement in node.body:
            self.visit(statement)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for value in [*node.args.defaults, *node.args.kw_defaults]:
            if value is not None:
                self.visit(value)
        outer, class_outer = self.aliases, self.class_outer_aliases
        self.aliases = self.deferred_aliases()
        self.class_outer_aliases = None
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments += [argument for argument in (node.args.vararg, node.args.kwarg) if argument]
        for argument in arguments:
            self.bind(ast.Name(id=argument.arg, ctx=ast.Store()))
        self.deferred += 1
        try:
            self.visit(node.body)
        finally:
            self.deferred -= 1
            self.aliases, self.class_outer_aliases = outer, class_outer

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Dekoratory, wartości domyślne i adnotacje mogą działać przy imporcie.
        for value in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
            if value is not None:
                self.visit(value)
        type_bindings = self.type_parameters(node)
        annotation_outer = self.aliases
        self.aliases = {**annotation_outer, **type_bindings}
        annotated_arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        annotated_arguments += [
            argument for argument in (node.args.vararg, node.args.kwarg) if argument
        ]
        for argument in annotated_arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        self.aliases = annotation_outer
        outer, class_outer = self.aliases, self.class_outer_aliases
        self.aliases = self.deferred_aliases()
        self.aliases.update(type_bindings)
        self.class_outer_aliases = None
        outer_rebound = self.scope_rebound_names
        self.scope_rebound_names = outer_rebound | self.rebound_names(
            ast.Module(body=node.body, type_ignores=[])
        )
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments += [argument for argument in (node.args.vararg, node.args.kwarg) if argument]
        for argument in arguments:
            self.bind(ast.Name(id=argument.arg, ctx=ast.Store()))
        self.deferred += 1
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self.deferred -= 1
            self.aliases, self.class_outer_aliases = outer, class_outer
            self.scope_rebound_names = outer_rebound
        self.aliases[node.name] = "<unresolved>." + node.name

    visit_AsyncFunctionDef = visit_FunctionDef

    def type_parameters(self, node: ast.AST) -> dict[str, str]:
        # getattr zachowuje działanie narzędzia na Pythonie 3.11. Analiza AST
        # obejmuje leniwe bounds CPython 3.12 bez uruchamiania ich kodu.
        parameters: list[ast.AST] = getattr(node, "type_params", [])
        outer = self.aliases
        self.aliases = dict(outer)
        names = [str(getattr(parameter, "name")) for parameter in parameters]
        try:
            for name in names:
                self.bind(ast.Name(id=name, ctx=ast.Store()))
            for parameter in parameters:
                self.visit(parameter)
            return {name: self.aliases[name] for name in names}
        finally:
            self.aliases = outer

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for value in node.decorator_list:
            self.visit(value)
        type_bindings = self.type_parameters(node)
        outer, class_outer = self.aliases, self.class_outer_aliases
        self.aliases = {**outer, **type_bindings}
        for value in node.bases:
            self.visit(value)
        for keyword in node.keywords:
            self.visit(keyword.value)
        enclosing = class_outer if class_outer is not None else outer
        self.class_outer_aliases = {**enclosing, **type_bindings}
        self.aliases = dict(self.class_outer_aliases)
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self.aliases, self.class_outer_aliases = outer, class_outer
        self.aliases[node.name] = "<unresolved>." + node.name

    def visit_comprehension_scope(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        outer, class_outer = self.aliases, self.class_outer_aliases
        self.aliases = dict(class_outer if class_outer is not None else outer)
        self.class_outer_aliases = None
        try:
            for generator in node.generators:
                self.visit(generator.iter)
                self.bind(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
        finally:
            self.aliases, self.class_outer_aliases = outer, class_outer

    visit_ListComp = visit_comprehension_scope
    visit_SetComp = visit_comprehension_scope
    visit_DictComp = visit_comprehension_scope
    visit_GeneratorExp = visit_comprehension_scope

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
            # Wyjątek ma znane pochodzenie symbolu. Nieznane obj.compile może
            # wskazywać builtin; sam suffix compile nie uzasadnia pominięcia.
            mechanism = name != "re.compile"
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

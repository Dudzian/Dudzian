"""Sprawdza dane pakietu; nie importuje ani nie wykonuje jego źródeł."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import sys
import tomllib
from pathlib import PurePosixPath
from typing import Any, cast

MAX_FILES = 4096
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
EXPECTED_BASE = "17e03a9cb42a84302401415f96ac90e756926915"
SAFETY = {
    "Stage 9": "IN_PROGRESS",
    "Observer 3A": "BLOCKED",
    "Stage 3B": "NOT AUTHORIZED",
    "Production": "NO-GO",
    "HOST_PREFLIGHT_VERIFIED": "NOT_CLAIMED",
    "EXECUTION_AUTHORIZATION": "NOT_AUTHORIZED",
    "PYC_RECONSTRUCTION": "UNVERIFIED",
}
STATUSES = {"PASS", "FAIL", "BLOCKED", "UNVERIFIED", "NOT_TESTED"}
REQUIRED_AREAS = {
    "package",
    "installation_environment",
    "host",
    "policy",
    "vm",
    "qualification",
    "execution_authorization",
}
REQUIRED_SOURCES = {
    "deployment/canonical_json.py",
    "deployment/stage9_observer_collector_identity_v5d1.py",
    "deployment/stage9_observer_kvm_domain_v5d1.py",
    "deployment/stage9_observer_kvm_image_v5d1.py",
    "deployment/stage9_observer_kvm_preflight_v5d1.py",
    "scripts/stage9_observer_kvm_preflight_v5d1.py",
}
REQUIRED_FILES = {
    "RELEASE.json",
    "DEPENDENCIES.json",
    "IMPORTS.json",
    "INTERPRETER.json",
    "NATIVE.json",
    "MISSING.json",
    "SUMMARY.json",
    "SUMMARY.tsv",
    "COMPLETE.json",
    "sources/pyproject.toml",
    "review/stage9_operator_v5d3_release_preparation.md",
    *(f"logs/{stage}.log" for stage in "ABCDEFGHI"),
}
CONTROL_FILES = {"MANIFEST.json", "SHA256SUMS"}
DEPENDENCY_KEYS = {
    "profile",
    "requires_python",
    "project_dependencies",
    "observer_extra",
    "libvirt_binding",
    "approved_package_set",
    "transitive_closure",
}


class Refusal(ValueError):
    """Stały kod odmowy; nie ujawnia badanej zawartości."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise Refusal(code)


def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def json_object(data: bytes) -> Any:
    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(Refusal("NONFINITE_JSON")),
    )


def keys(value: Any, expected: set[str], code: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == expected, code)
    return cast(dict[str, Any], value)


def relative_path(value: Any) -> str:
    require(
        type(value) is str
        and 0 < len(value) <= 1024
        and all(32 < ord(char) < 127 and char != "\\" for char in value)
        and not value.startswith("/")
        and all(part not in {"", ".", ".."} for part in value.split("/")),
        "UNSAFE_PATH",
    )
    return cast(str, value)


def identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def open_root(path: str) -> int:
    """Trzyma deskryptory podczas przejścia; nie podąża za symlinkami."""
    require(sys.platform == "linux", "VERIFIER_HOST_UNSUPPORTED")
    require(
        0 < len(os.fsencode(path)) <= 4096 and ".." not in path.split("/") and "\x00" not in path,
        "UNSAFE_BUNDLE_PATH",
    )
    absolute = os.path.abspath(path)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in PurePosixPath(absolute).parts[1:]:
            following = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def read_tree(descriptor: int) -> dict[str, tuple[bytes, str]]:
    result: dict[str, tuple[bytes, str]] = {}
    directories: set[str] = set()
    inodes: set[tuple[int, int]] = set()
    total = 0

    def names_in(directory: int) -> list[str]:
        names: list[str] = []
        with os.scandir(directory) as entries:
            for entry in entries:
                require(len(names) < MAX_FILES, "TREE_LIMIT_EXCEEDED")
                names.append(entry.name)
        return sorted(names)

    def walk(directory: int, prefix: str, depth: int) -> None:
        nonlocal total
        require(depth <= 16, "TREE_LIMIT_EXCEEDED")
        before = os.fstat(directory)
        require(before.st_mode & 0o7022 == 0, "UNSAFE_MODE")
        names = names_in(directory)
        for name in names:
            path = relative_path(prefix + name)
            named = os.stat(name, dir_fd=directory, follow_symlinks=False)
            require(not stat.S_ISLNK(named.st_mode), "SYMLINK_REJECTED")
            require(named.st_mode & 0o7022 == 0, "UNSAFE_MODE")
            if stat.S_ISDIR(named.st_mode):
                require(
                    path in {"sources", "sources/deployment", "sources/scripts", "logs", "review"}
                    or path.startswith(("sources/deployment/", "sources/scripts/")),
                    "UNEXPECTED_DIRECTORY",
                )
                require(len(directories) < MAX_FILES, "TREE_LIMIT_EXCEEDED")
                directories.add(path)
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory,
                )
                try:
                    require(identity(named) == identity(os.fstat(child)), "TREE_CHANGED")
                    walk(child, path + "/", depth + 1)
                    require(
                        identity(os.stat(name, dir_fd=directory, follow_symlinks=False))
                        == identity(os.fstat(child)),
                        "TREE_CHANGED",
                    )
                finally:
                    os.close(child)
                continue
            require(stat.S_ISREG(named.st_mode), "SPECIAL_OBJECT_REJECTED")
            require(permitted_file(path), "UNEXPECTED_FILE")
            check_secrets(path, b"")
            require(named.st_nlink == 1, "HARDLINK_REJECTED")
            inode = (named.st_dev, named.st_ino)
            require(inode not in inodes, "DUPLICATE_OBJECT")
            inodes.add(inode)
            require(
                len(result) < MAX_FILES and named.st_size <= MAX_FILE_BYTES, "TREE_LIMIT_EXCEEDED"
            )
            total += named.st_size
            require(total <= MAX_TOTAL_BYTES, "TREE_LIMIT_EXCEEDED")
            child = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=directory,
            )
            try:
                require(identity(named) == identity(os.fstat(child)), "TREE_CHANGED")
                chunks: list[bytes] = []
                size = 0
                while chunk := os.read(child, 65536):
                    size += len(chunk)
                    require(size <= MAX_FILE_BYTES and size <= named.st_size, "TREE_CHANGED")
                    chunks.append(chunk)
                require(size == named.st_size, "TREE_CHANGED")
                require(identity(named) == identity(os.fstat(child)), "TREE_CHANGED")
                require(
                    identity(named)
                    == identity(os.stat(name, dir_fd=directory, follow_symlinks=False)),
                    "TREE_CHANGED",
                )
                result[path] = (b"".join(chunks), f"{stat.S_IMODE(named.st_mode):04o}")
            finally:
                os.close(child)
        require(names == names_in(directory), "TREE_CHANGED")
        require(identity(before) == identity(os.fstat(directory)), "TREE_CHANGED")

    walk(descriptor, "", 0)
    expected_directories = {
        str(parent)
        for path in result
        for parent in PurePosixPath(path).parents
        if str(parent) != "."
    }
    require(directories == expected_directories, "UNEXPECTED_DIRECTORY")
    return result


def compile_calls(tree: ast.AST) -> set[ast.Call]:
    """Wyłącza wyłącznie re.compile ustalone przez jawne importy i aliasy.

    Nieznane pochodzenie, przesłonięcie lub refleksja nie stanowią dowodu,
    że kompilowane jest wyrażenie regularne. Analiza nie wykonuje źródeł.
    """

    class CompileVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.aliases: dict[str, str] = {}
            self.result: set[ast.Call] = set()
            self.class_outer: dict[str, str] | None = None
            self.scope_statements: list[ast.stmt] = []

        def name(self, node: ast.AST) -> str:
            if isinstance(node, ast.Name):
                return self.aliases.get(
                    node.id, node.id if node.id == "compile" else f"?.{node.id}"
                )
            if isinstance(node, ast.Attribute):
                return self.name(node.value) + "." + node.attr
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    base = self.name(node.value).removesuffix(".__dict__")
                    return base + "." + node.slice.value
                base = self.name(node.value).removesuffix(".__dict__")
                if base in {"builtins", "?.__builtins__"} or self.name(node.value).endswith(
                    ".__dict__"
                ):
                    return "?.compile"
            if isinstance(node, ast.Call) and self.name(node.func) in {
                "?.getattr",
                "getattr",
                "builtins.getattr",
                "?.__builtins__.getattr",
            }:
                if len(node.args) >= 2:
                    attribute = node.args[1]
                    if isinstance(attribute, ast.Constant) and isinstance(attribute.value, str):
                        return self.name(node.args[0]) + "." + attribute.value
                    return "?.compile"
            return "?"

        def compiler(self, node: ast.AST) -> bool:
            name = self.name(node)
            if name == "re.compile":
                return False
            if name == "compile" or name.endswith(".compile"):
                return True
            return any(self.compiler(child) for child in ast.iter_child_nodes(node))

        def bind(self, target: ast.AST, name: str = "?") -> None:
            if isinstance(target, ast.Name):
                previous = self.aliases.get(target.id, "")
                if name.startswith("?") and (
                    previous.endswith(".compile") or previous == "compile" or target.id == "compile"
                ):
                    name = "?.compile"
                self.aliases[target.id] = name
            elif isinstance(target, (ast.Tuple, ast.List)):
                for item in target.elts:
                    self.bind(item)
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                # Przypisanie re.compile = ... odbiera dowód znanego pochodzenia.
                if target.attr == "compile":
                    self.bind(target.value)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                self.aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                self.aliases[alias.asname or alias.name] = (
                    ((node.module or "?") if not node.level else "?.relative") + "." + alias.name
                )

        def visit_Assign(self, node: ast.Assign) -> None:
            self.visit(node.value)
            name = self.name(node.value)
            for target in node.targets:
                self.bind(target, name)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self.visit(node.annotation)
            if node.value is not None:
                self.visit(node.value)
            self.bind(node.target, self.name(node.value) if node.value is not None else "?")

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self.visit(node.value)
            self.bind(node.target)

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
            self.visit(node.value)
            self.bind(node.target, self.name(node.value))

        def local_bindings(self, nodes: list[ast.stmt]) -> set[str]:
            names: set[str] = set()
            pending: list[ast.AST] = list(nodes)
            while pending:
                node = pending.pop()
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                    continue
                if isinstance(node, ast.Lambda):
                    continue
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.add(node.id)
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Store)
                    and node.attr == "compile"
                    and isinstance(node.value, ast.Name)
                ):
                    names.add(node.value.id)
                if isinstance(node, ast.Import):
                    names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
                if isinstance(node, ast.ImportFrom):
                    names.update(alias.asname or alias.name for alias in node.names)
                if isinstance(node, ast.ExceptHandler) and node.name:
                    names.add(node.name)
                pending.extend(ast.iter_child_nodes(node))
            return names

        def visit_Module(self, node: ast.Module) -> None:
            self.scope_statements = node.body
            for statement in node.body:
                self.visit(statement)

        def rebound_names(self) -> set[str]:
            seen: set[str] = set()
            repeated: set[str] = set()
            for statement in self.scope_statements:
                names = self.local_bindings([statement])
                repeated.update(seen & names)
                seen.update(names)
            return repeated

        def visit_type_parameters(self, node: ast.AST) -> set[str]:
            # Pole istnieje od 3.12; narzędzia przygotowania wspierają także 3.11.
            parameters: list[ast.AST] = list(getattr(node, "type_params", ()))
            if not parameters:
                return set()
            names = {getattr(parameter, "name") for parameter in parameters}
            previous = self.aliases.copy()
            # Bounds są odroczone i mogą być odczytane po ponownym związaniu aliasu.
            for name in self.rebound_names():
                self.bind(ast.Name(id=name, ctx=ast.Store()))
            for name in names:
                self.aliases[name] = "?." + name
            for parameter in parameters:
                self.visit(parameter)
            self.aliases = previous
            return names

        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            for expression in [
                *node.decorator_list,
                *node.args.defaults,
                *node.args.kw_defaults,
            ]:
                if expression is not None:
                    self.visit(expression)
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            arguments += [
                argument for argument in (node.args.vararg, node.args.kwarg) if argument is not None
            ]
            type_names = self.visit_type_parameters(node)
            annotation_aliases = self.aliases.copy()
            for name in type_names:
                self.aliases[name] = "?." + name
            for annotation in [*(argument.annotation for argument in arguments), node.returns]:
                if annotation is not None:
                    self.visit(annotation)
            self.aliases = annotation_aliases
            self.aliases[node.name] = "?"
            previous = self.aliases.copy()
            previous_class = self.class_outer
            if self.class_outer is not None:
                self.aliases = self.class_outer.copy()
            self.class_outer = None
            # Ciało funkcji może działać po późniejszym przypisaniu w outer scope.
            # Zmienny alias traci wtedy dowód re.compile.
            for name in self.rebound_names():
                self.bind(ast.Name(id=name, ctx=ast.Store()))
            previous_statements = self.scope_statements
            self.scope_statements = node.body
            for name in self.local_bindings(node.body):
                self.aliases[name] = "?." + name
            for name in type_names:
                self.aliases[name] = "?." + name
            for argument in arguments:
                self.aliases[argument.arg] = "?." + argument.arg
            for statement in node.body:
                self.visit(statement)
            self.aliases = previous
            self.class_outer = previous_class
            self.scope_statements = previous_statements

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node: ast.Lambda) -> None:
            for expression in [*node.args.defaults, *node.args.kw_defaults]:
                if expression is not None:
                    self.visit(expression)
            previous = self.aliases.copy()
            previous_class = self.class_outer
            if self.class_outer is not None:
                self.aliases = self.class_outer.copy()
            self.class_outer = None
            for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                self.aliases[argument.arg] = "?." + argument.arg
            for variadic in (node.args.vararg, node.args.kwarg):
                if variadic is not None:
                    self.aliases[variadic.arg] = "?." + variadic.arg
            self.visit(node.body)
            self.aliases = previous
            self.class_outer = previous_class

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            for expression in node.decorator_list:
                self.visit(expression)
            type_names = self.visit_type_parameters(node)
            base_aliases = self.aliases.copy()
            for name in type_names:
                self.aliases[name] = "?." + name
            for expression in [*node.bases, *(keyword.value for keyword in node.keywords)]:
                self.visit(expression)
            self.aliases = base_aliases
            previous = self.aliases.copy()
            previous_class = self.class_outer
            self.class_outer = (
                previous_class.copy() if previous_class is not None else previous.copy()
            )
            for name in type_names:
                self.class_outer[name] = "?." + name
            self.aliases = self.class_outer.copy()
            for statement in node.body:
                self.visit(statement)
            self.aliases = previous
            self.class_outer = previous_class
            self.aliases[node.name] = "?"

        def visit_Delete(self, node: ast.Delete) -> None:
            for target in node.targets:
                self.bind(target)

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
            if node.name:
                self.bind(ast.Name(id=node.name, ctx=ast.Store()))
            for statement in node.body:
                self.visit(statement)
            if node.name:
                self.bind(ast.Name(id=node.name, ctx=ast.Store()))

        def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
            self.visit(node.iter)
            self.bind(node.target)
            for name in self.local_bindings(node.body):
                self.bind(ast.Name(id=name, ctx=ast.Store()))
            for statement in [*node.body, *node.orelse]:
                self.visit(statement)
            for name in self.local_bindings(node.body):
                self.bind(ast.Name(id=name, ctx=ast.Store()))

        visit_AsyncFor = visit_For

        def visit_If(self, node: ast.If) -> None:
            self.visit(node.test)
            previous = self.aliases.copy()
            names = self.local_bindings([*node.body, *node.orelse])
            for branch in (node.body, node.orelse):
                self.aliases = previous.copy()
                for name in names:
                    self.bind(ast.Name(id=name, ctx=ast.Store()))
                for statement in branch:
                    self.visit(statement)
            for name in names:
                self.bind(ast.Name(id=name, ctx=ast.Store()))

        def visit_While(self, node: ast.While) -> None:
            self.visit(node.test)
            names = self.local_bindings([*node.body, *node.orelse])
            for name in names:
                self.bind(ast.Name(id=name, ctx=ast.Store()))
            for statement in [*node.body, *node.orelse]:
                self.visit(statement)
            for name in names:
                self.bind(ast.Name(id=name, ctx=ast.Store()))

        def visit_Try(self, node: ast.Try | ast.TryStar) -> None:
            previous = self.aliases.copy()
            names = self.local_bindings([node])
            branches: list[list[ast.AST]] = [
                list(node.body),
                list(node.handlers),
                list(node.orelse),
                list(node.finalbody),
            ]
            for branch in branches:
                self.aliases = previous.copy()
                for name in names:
                    self.bind(ast.Name(id=name, ctx=ast.Store()))
                for statement in branch:
                    self.visit(statement)
            for name in names:
                self.bind(ast.Name(id=name, ctx=ast.Store()))

        visit_TryStar = visit_Try

        def visit_BoolOp(self, node: ast.BoolOp) -> None:
            self.generic_visit(node)
            for name in self.local_bindings([ast.Expr(value=node)]):
                self.bind(ast.Name(id=name, ctx=ast.Store()))

        def visit_IfExp(self, node: ast.IfExp) -> None:
            self.visit(node.test)
            previous = self.aliases.copy()
            names = self.local_bindings([ast.Expr(value=node.body), ast.Expr(value=node.orelse)])
            for branch in (node.body, node.orelse):
                self.aliases = previous.copy()
                for name in names:
                    self.bind(ast.Name(id=name, ctx=ast.Store()))
                self.visit(branch)
            for name in names:
                self.bind(ast.Name(id=name, ctx=ast.Store()))

        def visit_comprehension_expression(
            self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
        ) -> None:
            previous = self.aliases.copy()
            previous_class = self.class_outer
            for index, generator in enumerate(node.generators):
                self.visit(generator.iter)
                if index == 0 and self.class_outer is not None:
                    self.aliases = self.class_outer.copy()
                    self.class_outer = None
                self.bind(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
            self.aliases = previous
            self.class_outer = previous_class

        visit_ListComp = visit_comprehension_expression
        visit_SetComp = visit_comprehension_expression
        visit_DictComp = visit_comprehension_expression
        visit_GeneratorExp = visit_comprehension_expression

        def visit_Call(self, node: ast.Call) -> None:
            if self.compiler(node.func):
                self.result.add(node)
            self.generic_visit(node)

    visitor = CompileVisitor()
    visitor.visit(tree)
    return visitor.result


def dynamic_calls(tree: ast.AST) -> list[ast.Call]:
    """Rozpoznaje mechanizmy po nazwie AST, bez dopasowania podciągu execute.

    Alias nie stanowi dowodu wykonania; pozostaje wskazaniem do przeglądu.
    Nierozstrzygalne przepływy danych wymagają osobnego przeglądu domknięcia.
    """
    mechanisms = {
        "__import__",
        "import_module",
        "exec",
        "eval",
        "run_path",
        "run_module",
        "CDLL",
        "PyDLL",
        "LoadLibrary",
        "spec_from_file_location",
        "spec_from_loader",
        "dlopen",
        "load_dynamic",
        "module_from_spec",
        "exec_module",
        "load_module",
    }
    aliases = set(mechanisms)
    # Proste aliasy importu i przypisania są domykane przed analizą wywołań.
    changed = True
    while changed:
        previous = set(aliases)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in aliases:
                        aliases.add(alias.asname or alias.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                value = node.value
                name = (
                    value.id
                    if isinstance(value, ast.Name)
                    else (value.attr if isinstance(value, ast.Attribute) else "")
                )
                if (isinstance(value, ast.Name) and name in aliases) or (
                    isinstance(value, ast.Attribute) and name in mechanisms
                ):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    aliases.update(target.id for target in targets if isinstance(target, ast.Name))
        changed = aliases != previous
    compilers = compile_calls(tree)
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Zachowujemy getattr(..., "exec") i słownikowe wyszukiwanie
            # __builtins__["eval"], które wcześniejszy skan tekstowy widział.
            names = {
                part.id
                if isinstance(part, ast.Name)
                else part.attr
                if isinstance(part, ast.Attribute) and part.attr in mechanisms
                else part.value
                if isinstance(part, ast.Constant)
                else ""
                for part in ast.walk(node.func)
                if isinstance(part, (ast.Name, ast.Attribute))
                or (isinstance(part, ast.Constant) and isinstance(part.value, str))
            }
            if names & aliases or node in compilers:
                result.append(node)
    return result


def check_source_closure(files: dict[str, tuple[bytes, str]]) -> int:
    pending = list(REQUIRED_SOURCES)
    visited: set[str] = set()
    dynamic_expected: list[dict[str, Any]] = []
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        require("sources/" + path in files, "REQUIRED_SOURCE_MISSING")
        data = files["sources/" + path][0]
        tree = ast.parse(data, filename=path)
        package = path.split("/")[:-1]
        for count in range(1, len(package) + 1):
            initializer = "/".join(package[:count]) + "/__init__.py"
            if initializer != path:
                pending.append(initializer)
        for dynamic_node in dynamic_calls(tree):
            dynamic_expected.append(
                {
                    "path": path,
                    "line": dynamic_node.lineno,
                    "expression": ast.unparse(dynamic_node)[:1000],
                }
            )
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    require(node.level <= len(package), "IMPORT_PATH_UNRESOLVED")
                    parts = package[: len(package) - node.level + 1]
                    if node.module:
                        parts += node.module.split(".")
                    modules = [".".join(parts)]
                elif node.module:
                    modules = [node.module]
            for module in modules:
                if module.split(".")[0] not in {"deployment", "scripts"}:
                    continue
                base = module.replace(".", "/")
                candidate = base + ".py"
                if "sources/" + candidate not in files:
                    candidate = base + "/__init__.py"
                require("sources/" + candidate in files, "IMPORTED_SOURCE_MISSING")
                pending.append(candidate)
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        nested = base + "/" + alias.name
                        if "sources/" + nested + ".py" in files:
                            pending.append(nested + ".py")
                        elif "sources/" + nested + "/__init__.py" in files:
                            pending.append(nested + "/__init__.py")
                        elif base in {"deployment", "scripts"}:
                            raise Refusal("IMPORTED_SOURCE_MISSING")
    bundled = {
        path.removeprefix("sources/")
        for path in files
        if path.startswith("sources/") and path.endswith(".py")
    }
    require(bundled == visited, "SOURCE_FILESET_MISMATCH")
    imports = keys(
        json_object(files["IMPORTS.json"][0]),
        {"profile", "static_sources", "dynamic_imports", "status", "detail"},
        "IMPORT_SCHEMA",
    )
    require(imports["profile"] == "stage9-v5d3-imports-v1", "IMPORT_SCHEMA")
    require(imports["static_sources"] == sorted(visited), "IMPORT_CLOSURE_MISMATCH")
    require(imports["status"] == "UNVERIFIED", "IMPORT_APPROVAL_UNSUPPORTED")
    require(type(imports["detail"]) is str, "IMPORT_SCHEMA")
    require(type(imports["dynamic_imports"]) is list, "IMPORT_SCHEMA")
    for dynamic in imports["dynamic_imports"]:
        keys(dynamic, {"path", "line", "expression"}, "IMPORT_SCHEMA")
        require(dynamic["path"] in visited, "IMPORT_SCHEMA")
        require(type(dynamic["line"]) is int and dynamic["line"] > 0, "IMPORT_SCHEMA")
        require(type(dynamic["expression"]) is str, "IMPORT_SCHEMA")
    require(
        imports["dynamic_imports"]
        == sorted(dynamic_expected, key=lambda row: (row["path"], row["line"])),
        "DYNAMIC_IMPORT_INVENTORY_MISMATCH",
    )
    return len(visited)


def check_dependencies(files: dict[str, tuple[bytes, str]]) -> None:
    dependencies = keys(
        json_object(files["DEPENDENCIES.json"][0]), DEPENDENCY_KEYS, "DEPENDENCY_SCHEMA"
    )
    require(dependencies["profile"] == "stage9-v5d3-dependencies-v1", "DEPENDENCY_SCHEMA")
    project = tomllib.loads(files["sources/pyproject.toml"][0].decode("utf-8"))["project"]
    require(
        type(dependencies["requires_python"]) is str
        and bool(dependencies["requires_python"])
        and type(dependencies["observer_extra"]) is list
        and all(type(value) is str for value in dependencies["observer_extra"]),
        "DEPENDENCY_SCHEMA",
    )
    require(dependencies["requires_python"] == project["requires-python"], "DEPENDENCY_MISMATCH")
    require(
        dependencies["project_dependencies"] == project["dependencies"]
        and dependencies["observer_extra"]
        == project["optional-dependencies"]["observer-kvm-preflight"],
        "DEPENDENCY_MISMATCH",
    )
    require(
        type(project["dependencies"]) is list
        and all(type(item) is str for item in project["dependencies"])
        and "cryptography>=50.0.1,<51" in project["dependencies"]
        and "defusedxml==0.7.1" in dependencies["observer_extra"],
        "REQUIRED_DEPENDENCY_MISSING",
    )
    require(
        dependencies["libvirt_binding"] == "REVIEWED_SYSTEM_BINDING_REQUIRED"
        and dependencies["approved_package_set"] == "UNVERIFIED"
        and dependencies["transitive_closure"] == "UNVERIFIED",
        "DEPENDENCY_APPROVAL_UNSUPPORTED",
    )


def check_summary(files: dict[str, tuple[bytes, str]]) -> None:
    summary = keys(
        json_object(files["SUMMARY.json"][0]), {"profile", "areas", "safety"}, "SUMMARY_SCHEMA"
    )
    require(summary["profile"] == "stage9-v5d3-summary-v1", "SUMMARY_SCHEMA")
    require(summary["safety"] == SAFETY, "SAFETY_BOUNDARY_CHANGED")
    require(type(summary["areas"]) is list, "SUMMARY_SCHEMA")
    seen: set[str] = set()
    for row in summary["areas"]:
        keys(row, {"area", "status", "detail"}, "SUMMARY_SCHEMA")
        require(type(row["area"]) is str and row["area"] not in seen, "SUMMARY_SCHEMA")
        seen.add(row["area"])
        require(type(row["status"]) is str and row["status"] in STATUSES, "SUMMARY_SCHEMA")
        require(
            type(row["detail"]) is str and not any(char in row["detail"] for char in "\r\n\t"),
            "SUMMARY_SCHEMA",
        )
        if row["area"] in REQUIRED_AREAS - {"package"}:
            require(row["status"] != "PASS", "READINESS_CLAIM_REJECTED")
    require(REQUIRED_AREAS <= seen, "SUMMARY_AREA_MISSING")
    expected_areas = {
        **dict.fromkeys("ABCDEFGHI", "PASS"),
        "package": "PASS",
        "installation_environment": "BLOCKED",
        "host": "NOT_TESTED",
        "policy": "BLOCKED",
        "vm": "BLOCKED",
        "qualification": "BLOCKED",
        "execution_authorization": "BLOCKED",
        "exact_cpython_execution": "NOT_TESTED",
        "import_completeness": "UNVERIFIED",
        "dependency_resolution": "BLOCKED",
        "credential_scan": "UNVERIFIED",
    }
    require(
        {row["area"]: row["status"] for row in summary["areas"]} == expected_areas,
        "SUMMARY_READINESS_MISMATCH",
    )
    expected_tsv = "AREA\tSTATUS\tDETAIL\n"
    for row in summary["areas"]:
        expected_tsv += "\t".join(row[key] for key in ("area", "status", "detail")) + "\n"
    for key, value in SAFETY.items():
        expected_tsv += f"{key}\t{value}\tGranica kontraktu; bez zmiany.\n"
    require(files["SUMMARY.tsv"][0] == expected_tsv.encode("utf-8"), "SUMMARY_FORMAT_MISMATCH")


def check_metadata(files: dict[str, tuple[bytes, str]], commit: str) -> None:
    release = keys(
        json_object(files["RELEASE.json"][0]),
        {
            "profile",
            "expected_base",
            "release_commit",
            "repository_clean",
            "tools",
            "previous_bundle",
            "source_approval",
            "detail",
        },
        "RELEASE_SCHEMA",
    )
    require(
        release["profile"] == "stage9-v5d3-release-v1"
        and release["expected_base"] == EXPECTED_BASE
        and release["release_commit"] == commit
        and release["repository_clean"] is True
        and release["source_approval"] == "UNVERIFIED"
        and type(release["previous_bundle"]) is str
        and type(release["detail"]) is str,
        "RELEASE_IDENTITY_MISMATCH",
    )
    tools = keys(
        release["tools"], {"bash", "git", "python3", "sha256sum", "readelf"}, "RELEASE_SCHEMA"
    )
    require(
        all(
            type(value) is str and value.startswith("/")
            for name, value in tools.items()
            if name != "readelf"
        )
        and (tools["readelf"] is None or type(tools["readelf"]) is str),
        "RELEASE_SCHEMA",
    )
    interpreter = keys(
        json_object(files["INTERPRETER.json"][0]),
        {
            "profile",
            "scope",
            "executable",
            "actual_executable",
            "version",
            "implementation",
            "cache_tag",
            "soabi",
            "multiarch",
            "sys_path",
            "isolated",
            "no_site",
            "deployment_interpreter",
            "startup_files",
            "detail",
        },
        "INTERPRETER_SCHEMA",
    )
    require(
        interpreter["profile"] == "stage9-v5d3-interpreter-v1"
        and interpreter["scope"] == "PREPARATION_TOOL_ONLY"
        and interpreter["deployment_interpreter"] == "UNVERIFIED"
        and interpreter["startup_files"] == "UNVERIFIED"
        and type(interpreter["isolated"]) is int
        and interpreter["isolated"] == 1
        and type(interpreter["no_site"]) is int
        and interpreter["no_site"] == 1,
        "INTERPRETER_CLAIM_REJECTED",
    )
    require(
        all(
            type(interpreter[key]) is str
            for key in ("executable", "actual_executable", "version", "implementation", "detail")
        )
        and all(
            interpreter[key] is None or type(interpreter[key]) is str
            for key in ("cache_tag", "soabi", "multiarch")
        )
        and type(interpreter["sys_path"]) is list
        and all(type(value) is str for value in interpreter["sys_path"]),
        "INTERPRETER_SCHEMA",
    )
    native = keys(
        json_object(files["NATIVE.json"][0]),
        {"profile", "status", "required_components", "libraries", "detail"},
        "NATIVE_SCHEMA",
    )
    require(
        native["profile"] == "stage9-v5d3-native-v1"
        and native["status"] == "UNVERIFIED"
        and native["libraries"] == []
        and type(native["required_components"]) is list
        and all(type(value) is str for value in native["required_components"])
        and {"libvirt-python", "libvirtmod", "/usr/bin/qemu-img", "/usr/bin/qemu-system-x86_64"}
        <= set(native["required_components"])
        and type(native["detail"]) is str,
        "NATIVE_CLOSURE_UNSUPPORTED",
    )
    missing = keys(
        json_object(files["MISSING.json"][0]),
        {"profile", "items", "execution_authorization"},
        "MISSING_SCHEMA",
    )
    require(
        missing["profile"] == "stage9-v5d3-missing-v1"
        and missing["execution_authorization"] == "NOT_AUTHORIZED"
        and type(missing["items"]) is list
        and 1 <= len(missing["items"]) <= 1024
        and all(type(item) is str and bool(item) for item in missing["items"]),
        "MISSING_EVIDENCE_MISMATCH",
    )


def check_secrets(path: str, data: bytes) -> None:
    name = PurePosixPath(path).name.lower()
    require(
        not name.startswith(".env")
        and name
        not in {
            "id_rsa",
            "id_ed25519",
            "credentials",
            "operator-anchors.json",
            "approved-policy.json",
        }
        and not name.endswith((".pem", ".key", ".p12", ".pfx", ".pyc", ".pyo")),
        "CREDENTIAL_OR_BYTECODE_FILE_REJECTED",
    )
    # Zakres musi obejmować każdą odmowę scan_bytes generatora, także
    # dopasowania wewnątrz słowa; reguły pochodzą z zaufanego narzędzia.
    require(
        re.search(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----", data) is None
        and re.search(
            rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}|AKIA[A-Z0-9]{16})", data
        )
        is None,
        "POTENTIAL_CREDENTIAL_REJECTED",
    )


def permitted_file(path: str) -> bool:
    return (
        path in REQUIRED_FILES | CONTROL_FILES
        or path
        in {f"logs/previous-verification.{suffix}" for suffix in ("stdout", "stderr", "exit")}
        or (path.startswith("sources/deployment/") and path.endswith(".py"))
        or (path.startswith("sources/scripts/") and path.endswith(".py"))
    )


def verify(path: str) -> dict[str, Any]:
    descriptor = open_root(path)
    try:
        files = read_tree(descriptor)
    finally:
        os.close(descriptor)
    return verify_files(files)


def verify_files(files: dict[str, tuple[bytes, str]]) -> dict[str, Any]:
    require(CONTROL_FILES | REQUIRED_FILES <= set(files), "BUNDLE_INCOMPLETE")
    previous = {f"logs/previous-verification.{suffix}" for suffix in ("stdout", "stderr", "exit")}
    require(not (previous & files.keys()) or previous <= files.keys(), "PREVIOUS_REPORT_INCOMPLETE")
    if previous <= files.keys():
        require(files["logs/previous-verification.exit"][0] == b"0\n", "PREVIOUS_VERIFIER_FAILED")
        previous_report = json_object(files["logs/previous-verification.stdout"][0])
        require(
            type(previous_report) is dict and previous_report.get("local_consistency") == "PASS",
            "PREVIOUS_VERIFIER_FAILED",
        )
    for name in REQUIRED_FILES:
        if name.endswith(".json"):
            json_object(files[name][0])
    for path, (data, _) in files.items():
        require(permitted_file(path), "UNEXPECTED_FILE")
        check_secrets(path, data)
    manifest = keys(
        json_object(files["MANIFEST.json"][0]),
        {"profile", "release_commit", "files"},
        "MANIFEST_SCHEMA",
    )
    require(manifest["profile"] == "stage9-v5d3-bundle-v1", "MANIFEST_SCHEMA")
    commit = manifest["release_commit"]
    require(
        type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
        "RELEASE_COMMIT_INVALID",
    )
    require(
        type(manifest["files"]) is list and len(manifest["files"]) <= MAX_FILES, "MANIFEST_SCHEMA"
    )
    declared: set[str] = set()
    for row in manifest["files"]:
        keys(row, {"path", "sha256", "size", "mode"}, "MANIFEST_SCHEMA")
        entry = relative_path(row["path"])
        require(entry not in declared and entry not in CONTROL_FILES, "DUPLICATE_MANIFEST_PATH")
        declared.add(entry)
        require(entry in files, "DECLARED_FILE_MISSING")
        data, mode = files[entry]
        require(type(row["size"]) is int and row["size"] == len(data), "SIZE_MISMATCH")
        require(type(row["mode"]) is str and row["mode"] == mode, "MODE_MISMATCH")
        require(
            type(row["sha256"]) is str and row["sha256"] == hashlib.sha256(data).hexdigest(),
            "SHA256_MISMATCH",
        )
    require(declared == set(files) - CONTROL_FILES, "MANIFEST_FILESET_MISMATCH")
    sums: dict[str, str] = {}
    for line in files["SHA256SUMS"][0].decode("ascii").splitlines():
        require(re.fullmatch(r"[0-9a-f]{64}  [!-~]+", line) is not None, "CHECKSUM_MANIFEST_SCHEMA")
        digest, entry = line.split("  ", 1)
        relative_path(entry)
        require(entry not in sums and entry != "SHA256SUMS", "DUPLICATE_CHECKSUM_PATH")
        require(entry in files, "CHECKSUM_FILE_MISSING")
        require(hashlib.sha256(files[entry][0]).hexdigest() == digest, "SHA256_MISMATCH")
        sums[entry] = digest
    require(set(sums) == set(files) - {"SHA256SUMS"}, "CHECKSUM_FILESET_MISMATCH")
    complete = keys(
        json_object(files["COMPLETE.json"][0]), {"profile", "release_commit"}, "COMPLETE_SCHEMA"
    )
    require(
        complete == {"profile": "stage9-v5d3-complete-v1", "release_commit": commit},
        "PREPARATION_INCOMPLETE",
    )
    check_dependencies(files)
    check_summary(files)
    check_metadata(files, commit)
    source_count = check_source_closure(files)
    return {
        "profile": "stage9-v5d3-verification-v1",
        "local_consistency": "PASS",
        "file_count": len(files),
        "static_source_count": source_count,
        "release_commit": commit,
        "independent_approval": "UNVERIFIED",
        "import_runtime_closure": "UNVERIFIED",
        "dependency_completeness": "UNVERIFIED",
        "credential_scan": "UNVERIFIED",
        "detail": "Spójność lokalna potwierdzona; sumy nie dowodzą pochodzenia, braku wszystkich sekretów ani niezależnej aprobaty.",
        "safety": SAFETY,
    }


def read_receipt(directory: int, name: str) -> bytes:
    """Odczyt pojedynczego pliku koperty bez symlinków i bez wykonania jego treści."""
    named = os.stat(name, dir_fd=directory, follow_symlinks=False)
    require(stat.S_ISREG(named.st_mode) and named.st_nlink == 1, "HANDOFF_OBJECT_REJECTED")
    require(
        named.st_mode & 0o7022 == 0 and named.st_size <= MAX_FILE_BYTES, "HANDOFF_OBJECT_REJECTED"
    )
    child = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=directory
    )
    try:
        require(identity(named) == identity(os.fstat(child)), "TREE_CHANGED")
        chunks = []
        size = 0
        while chunk := os.read(child, 65536):
            size += len(chunk)
            require(size <= named.st_size, "TREE_CHANGED")
            chunks.append(chunk)
        require(
            size == named.st_size and identity(named) == identity(os.fstat(child)), "TREE_CHANGED"
        )
        require(
            identity(named) == identity(os.stat(name, dir_fd=directory, follow_symlinks=False)),
            "TREE_CHANGED",
        )
        return b"".join(chunks)
    finally:
        os.close(child)


def verify_handoff(path: str) -> dict[str, Any]:
    descriptor = open_root(path)
    try:
        before = os.fstat(descriptor)
        require(before.st_mode & 0o7022 == 0, "UNSAFE_MODE")
        manifest = keys(
            json_object(read_receipt(descriptor, "HANDOFF.json")),
            {
                "profile",
                "files",
                "independent_approval",
                "safety",
            },
            "HANDOFF_SCHEMA",
        )
        require(manifest["profile"] == "stage9-v5d4-handoff-v1", "HANDOFF_SCHEMA")
        require(
            manifest["independent_approval"] == "UNVERIFIED" and manifest["safety"] == SAFETY,
            "HANDOFF_AUTHORITY_REJECTED",
        )
        require(
            type(manifest["files"]) is list and len(manifest["files"]) == 5, "HANDOFF_INCOMPLETE"
        )
        bundle_descriptor = os.open(
            "bundle", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=descriptor
        )
        try:
            files = read_tree(bundle_descriptor)
            bundle_identity = identity(os.fstat(bundle_descriptor))
        finally:
            os.close(bundle_descriptor)
        report = verify_files(files)
        payloads = {
            "bundle/MANIFEST.json": files["MANIFEST.json"][0],
            "bundle/SHA256SUMS": files["SHA256SUMS"][0],
            **{
                name: read_receipt(descriptor, name)
                for name in ("verification.json", "verification.stderr", "verification.exit")
            },
        }
        declared = set()
        for entry in manifest["files"]:
            keys(entry, {"path", "sha256", "size"}, "HANDOFF_SCHEMA")
            name = relative_path(entry["path"])
            require(name in payloads and name not in declared, "HANDOFF_FILESET_MISMATCH")
            declared.add(name)
            data = payloads[name]
            require(
                type(entry["size"]) is int and entry["size"] == len(data), "HANDOFF_SIZE_MISMATCH"
            )
            require(entry["sha256"] == hashlib.sha256(data).hexdigest(), "HANDOFF_SHA256_MISMATCH")
            check_secrets(name, data)
        require(declared == set(payloads), "HANDOFF_INCOMPLETE")
        require(payloads["verification.exit"] == b"0\n", "HANDOFF_VERIFIER_FAILED")
        require(json_object(payloads["verification.json"]) == report, "HANDOFF_REPORT_MISMATCH")
        require(not payloads["verification.stderr"], "HANDOFF_VERIFIER_STDERR")
        require(
            bundle_identity
            == identity(os.stat("bundle", dir_fd=descriptor, follow_symlinks=False)),
            "TREE_CHANGED",
        )
        require(identity(before) == identity(os.fstat(descriptor)), "TREE_CHANGED")
        return {
            "profile": "stage9-v5d4-handoff-verification-v1",
            "local_consistency": "PASS",
            "release_commit": report["release_commit"],
            "independent_approval": "UNVERIFIED",
            "safety": SAFETY,
        }
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handoff",
        action="store_true",
        help="Wymagaj koperty i oryginalnych raportów obok bundle/",
    )
    parser.add_argument("bundle", help="Katalog bundle lub katalog przekazania z --handoff")
    args = parser.parse_args(argv)
    try:
        report = verify_handoff(args.bundle) if args.handoff else verify(args.bundle)
        exit_code = 0
    except (OSError, ValueError, TypeError, KeyError, SyntaxError, RecursionError) as error:
        report = {
            "profile": "stage9-v5d3-verification-v1",
            "local_consistency": "FAIL",
            "reason_code": str(error)
            if isinstance(error, Refusal)
            else "UNREADABLE_OR_MALFORMED_BUNDLE",
            "detail": "Odmowa weryfikacji; zachowaj pakiet do analizy i nie instaluj go.",
            "independent_approval": "UNVERIFIED",
            "safety": SAFETY,
        }
        exit_code = 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

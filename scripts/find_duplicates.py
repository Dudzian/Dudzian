"""Utility for detecting duplicated Python modules and definitions.

The migration from the legacy ``KryptoLowca`` package to the modernised
"bot_core"/"core" layout left a substantial amount of code that may look
identical even if it now lives in different directories.  This helper analyses
the repository, normalises the Python AST (removing docstrings and comments)
and reports groups of duplicated files as well as duplicated class/function
definitions.

The script focuses on the directories that matter for the runtime
implementation (``bot_core/``, ``core/``, ``scripts/``, ``tests/``) and marks
preferred (canonical) locations using a deterministic priority order.

The output is a JSON object containing two sections:

``duplicate_files``
    Mapping of hash → information about the canonical module and its
    duplicates.

``duplicate_definitions``
    Mapping of hash → information about duplicated class/function definitions
    including their qualified names within the modules.

Example usage::

    $ python scripts/find_duplicates.py --json

"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that constitute the new code base – we prefer keeping these files
# and treat other locations as legacy fallbacks.
CANONICAL_ROOTS = [
    REPO_ROOT / "bot_core",
    REPO_ROOT / "core",
    REPO_ROOT / "ui",
    REPO_ROOT / "proto",
    REPO_ROOT / "scripts",
    REPO_ROOT / "tests",
]

# Legacy sources have been removed – we only analyse canonical roots.
LEGACY_ROOTS: list[Path] = []

SCAN_ROOTS = CANONICAL_ROOTS


class _DocstringStripper(ast.NodeTransformer):
    """Remove docstrings from modules, classes and (async) functions."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, (ast.Str, ast.Constant)) and isinstance(
                getattr(value, "value", getattr(value, "s", None)), str
            ):
                node.body.pop(0)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node = self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, (ast.Str, ast.Constant)) and isinstance(
                getattr(value, "value", getattr(value, "s", None)), str
            ):
                node.body.pop(0)
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node = self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, (ast.Str, ast.Constant)) and isinstance(
                getattr(value, "value", getattr(value, "s", None)), str
            ):
                node.body.pop(0)
        return node


@dataclasses.dataclass(slots=True)
class FileInfo:
    path: Path
    root: Path

    @property
    def relative_path(self) -> Path:
        return self.path.relative_to(REPO_ROOT)

    @property
    def priority(self) -> Tuple[int, str]:
        for index, base in enumerate(CANONICAL_ROOTS):
            if base in self.path.parents or self.path == base:
                return (0, f"{index:02d}:{self.relative_path.as_posix()}")
        for index, base in enumerate(LEGACY_ROOTS):
            if base in self.path.parents or self.path == base:
                return (index + 1, self.relative_path.as_posix())
        # Everything else is considered the lowest priority (archival).
        return (len(LEGACY_ROOTS) + 1, self.relative_path.as_posix())


@dataclasses.dataclass(slots=True)
class DefinitionInfo:
    name: str
    file: FileInfo


def _iter_python_files() -> Iterator[FileInfo]:
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.name == "__pycache__":
                continue
            yield FileInfo(path=path, root=root)


def _normalise_tree(tree: ast.AST) -> ast.AST:
    tree = deepcopy(tree)
    _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def _hash_tree(tree: ast.AST) -> str:
    data = ast.dump(tree, include_attributes=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _collect_definitions(tree: ast.Module, file_info: FileInfo) -> Iterable[DefinitionInfo]:
    stack: List[str] = []

    def walk(node: ast.AST) -> Iterator[DefinitionInfo]:
        for child in getattr(node, "body", []) or []:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stack.append(child.name)
                qualname = ".".join(stack)
                yield DefinitionInfo(name=qualname, file=file_info)
                yield from walk(child)
                stack.pop()
            else:
                yield from walk(child)

    return walk(tree)


def _canonical_entry(entries: Iterable[FileInfo]) -> FileInfo:
    return min(entries, key=lambda info: info.priority)


def analyse_repository() -> Dict[str, Dict[str, object]]:
    file_hashes: Dict[str, List[FileInfo]] = defaultdict(list)
    definition_hashes: Dict[str, List[DefinitionInfo]] = defaultdict(list)

    for file_info in _iter_python_files():
        try:
            source = file_info.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Syntax errors are handled separately (compileall), ignore here.
            continue

        normalised_module = _normalise_tree(tree)
        module_hash = _hash_tree(normalised_module)
        file_hashes[module_hash].append(file_info)

        for definition in _collect_definitions(normalised_module, file_info):
            node = _find_definition_node(normalised_module, definition.name.split("."))
            if node is None:
                continue
            definition_hashes[_hash_tree(node)].append(definition)

    duplicate_files = {}
    for digest, infos in sorted(file_hashes.items(), key=lambda item: item[0]):
        if len(infos) < 2:
            continue
        canonical = _canonical_entry(infos)
        duplicate_files[digest] = {
            "canonical": canonical.relative_path.as_posix(),
            "duplicates": [info.relative_path.as_posix() for info in infos],
        }

    duplicate_definitions = {}
    for digest, definitions in sorted(definition_hashes.items(), key=lambda item: item[0]):
        if len(definitions) < 2:
            continue
        canonical = _canonical_entry([info.file for info in definitions])
        duplicate_definitions[digest] = {
            "canonical": canonical.relative_path.as_posix(),
            "definitions": [
                {
                    "name": info.name,
                    "file": info.file.relative_path.as_posix(),
                }
                for info in definitions
            ],
        }

    return {
        "duplicate_files": duplicate_files,
        "duplicate_definitions": duplicate_definitions,
    }


def _find_definition_node(tree: ast.AST, path: List[str]) -> ast.AST | None:
    if not path:
        return tree
    current, *rest = path
    for child in getattr(tree, "body", []) or []:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == current:
            if rest:
                return _find_definition_node(child, rest)
            return child
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit result as JSON (default: pretty printed JSON)",
    )
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Exit with status 1 if any duplicates are detected.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = analyse_repository()
    output = json.dumps(report, indent=None if args.json else 2, sort_keys=True)
    print(output)
    if args.fail_on_duplicates and (report["duplicate_files"] or report["duplicate_definitions"]):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

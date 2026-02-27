from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _collect_qarg_aliases(tree: ast.AST) -> set[str]:
    aliases = {"Q_ARG"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                if imported.name == "Q_ARG":
                    aliases.add(imported.asname or imported.name)
    return aliases

def _is_qarg_call(node: ast.Call, qarg_aliases: set[str]) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id in qarg_aliases:
        return True
    if isinstance(func, ast.Attribute) and func.attr == "Q_ARG":
        return True
    return False


def _is_qvariant_type(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in {
        "QVariant",
        "QVariantMap",
    }


def _is_forbidden_source(node: ast.AST) -> bool:
    if isinstance(node, ast.Dict):
        return True
    if isinstance(node, ast.DictComp):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"dict", "_as_py"}:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_as_py":
            return True
    return False


def _collect_qvariant_alias_violations(
    tree: ast.AST,
    qarg_aliases: set[str],
) -> list[tuple[int, str]]:
    """Znajdź Q_ARG("QVariant"|"QVariantMap", <zmienna>) z aliasem dict-like."""

    forbidden_names: set[str] = set()
    assignments: list[tuple[str, ast.AST]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.append((target.id, node.value))
            if _is_forbidden_source(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        forbidden_names.add(target.id)
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                assignments.append((node.target.id, node.value))
                if _is_forbidden_source(node.value):
                    forbidden_names.add(node.target.id)

    # Minimal flow tracking (1-2+ hops): tmp=dict(...); arg=tmp; x=arg; Q_ARG("QVariant", x)
    changed = True
    while changed:
        changed = False
        for target_name, value in assignments:
            if isinstance(value, ast.Name) and value.id in forbidden_names and target_name not in forbidden_names:
                forbidden_names.add(target_name)
                changed = True
                continue
            # Propagacja tylko dla dict-like aliasu: copied = payload.copy().
            # Celowo nie obejmujemy call-site'ów typu get_payload().copy(), bo bez
            # analizy międzyproceduralnej dawałoby to dużo false-positive.
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "copy"
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id in forbidden_names
                and target_name not in forbidden_names
            ):
                forbidden_names.add(target_name)
                changed = True

    bad_lines: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_qarg_call(node, qarg_aliases):
            continue
        if len(node.args) < 2:
            continue
        variant_type = node.args[0]
        if not (
            isinstance(variant_type, ast.Constant)
            and isinstance(variant_type.value, str)
            and variant_type.value in {"QVariant", "QVariantMap"}
        ):
            continue
        value_arg = node.args[1]
        if isinstance(value_arg, ast.Name) and value_arg.id in forbidden_names:
            reason = f"{value_arg.id} aliases dict-like source"
            bad_lines.append((node.lineno, reason))

    return bad_lines


def test_no_direct_dict_sources_in_qvariant_qarg_calls() -> None:
    targets = [
        *ROOT.joinpath("tests").rglob("*.py"),
        *ROOT.joinpath("ui").rglob("*.py"),
    ]
    violations: list[str] = []
    for path in targets:
        if path.name == Path(__file__).name:
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue

        qarg_aliases = _collect_qarg_aliases(tree)
        bad_lines: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_qarg_call(node, qarg_aliases):
                continue
            if len(node.args) < 2 or not _is_qvariant_type(node.args[0]):
                continue
            if _is_forbidden_source(node.args[1]):
                bad_lines.append((node.lineno, "direct dict-like source in Q_ARG"))

        bad_lines.extend(_collect_qvariant_alias_violations(tree, qarg_aliases))

        if bad_lines:
            uniq = sorted(set(bad_lines), key=lambda item: item[0])
            joined = ", ".join(f"L{line} ({reason})" for line, reason in uniq)
            violations.append(f"{path.relative_to(ROOT)}: {joined}")

    assert not violations, (
        "Windows-safe invokeMethod rule violated: avoid dict-like sources directly in "
        "Q_ARG(\"QVariant\"|\"QVariantMap\", ...). "
        f"Violations: {violations}"
    )


def test_collect_qvariant_alias_violations_tracks_two_step_aliases() -> None:
    source = """
from PySide6.QtCore import Q_ARG

def f():
    payload = dict(a=1)
    tmp = payload
    arg = tmp
    Q_ARG("QVariant", arg)
"""
    tree = ast.parse(source)
    aliases = _collect_qarg_aliases(tree)
    lines = _collect_qvariant_alias_violations(tree, aliases)
    assert lines == [(8, "arg aliases dict-like source")]


def test_collect_qvariant_alias_violations_covers_qvariantmap_alias() -> None:
    source = """
from PySide6.QtCore import Q_ARG

def f():
    payload = {"a": 1}
    alias = payload
    Q_ARG("QVariantMap", alias)
"""
    tree = ast.parse(source)
    aliases = _collect_qarg_aliases(tree)
    lines = _collect_qvariant_alias_violations(tree, aliases)
    assert lines == [(7, "alias aliases dict-like source")]


def test_collect_qvariant_alias_violations_detects_copy_and_unpack_alias() -> None:
    source = """
from PySide6.QtCore import Q_ARG

def f(payload):
    copied = payload.copy()
    unpacked = {**copied}
    Q_ARG("QVariant", unpacked)
"""
    tree = ast.parse(source)
    aliases = _collect_qarg_aliases(tree)
    lines = _collect_qvariant_alias_violations(tree, aliases)
    assert lines == [(7, "unpacked aliases dict-like source")]


def test_collect_qvariant_alias_violations_detects_copy_alias_from_dict_source() -> None:
    source = """
from PySide6.QtCore import Q_ARG

def f():
    payload = {"a": 1}
    copied = payload.copy()
    Q_ARG("QVariant", copied)
"""
    tree = ast.parse(source)
    aliases = _collect_qarg_aliases(tree)
    lines = _collect_qvariant_alias_violations(tree, aliases)
    assert lines == [(7, "copied aliases dict-like source")]


def test_collect_qvariant_alias_violations_ignores_unrelated_copy_calls() -> None:
    source = """
from PySide6.QtCore import Q_ARG

def f(image):
    copied = image.copy()
    Q_ARG("QVariant", copied)
"""
    tree = ast.parse(source)
    aliases = _collect_qarg_aliases(tree)
    lines = _collect_qvariant_alias_violations(tree, aliases)
    assert lines == []


def test_collect_qvariant_alias_violations_ignores_copy_from_call_expression() -> None:
    source = """
from PySide6.QtCore import Q_ARG

def get_payload():
    return {"a": 1}

def f():
    copied = get_payload().copy()
    Q_ARG("QVariant", copied)
"""
    tree = ast.parse(source)
    aliases = _collect_qarg_aliases(tree)
    lines = _collect_qvariant_alias_violations(tree, aliases)
    assert lines == []

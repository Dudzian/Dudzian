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
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"dict", "_as_py"}:
            return True
    return False


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
        bad_lines: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_qarg_call(node, qarg_aliases):
                continue
            if len(node.args) < 2 or not _is_qvariant_type(node.args[0]):
                continue
            if _is_forbidden_source(node.args[1]):
                bad_lines.append(node.lineno)

        if bad_lines:
            violations.append(f"{path.relative_to(ROOT)}:{','.join(map(str, bad_lines))}")

    assert not violations, (
        "Windows-safe invokeMethod rule violated: avoid dict-like sources directly in "
        "Q_ARG(\"QVariant\"|\"QVariantMap\", ...). "
        f"Violations: {violations}"
    )

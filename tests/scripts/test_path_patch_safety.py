from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_STRING_TARGETS = {
    "os.name",
    "pathlib.Path",
    "pathlib.PosixPath",
    "pathlib.WindowsPath",
}
FORBIDDEN_PATHLIB_ATTRS = {"Path", "PosixPath", "WindowsPath"}
PATCH_IMPORT_SOURCES = {"unittest.mock", "mock"}


def _name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


def _is_os_target_name(name: str | None, os_aliases: set[str]) -> bool:
    return bool(name) and (name in os_aliases or name.endswith(".os"))


def _is_pathlib_type_target_name(name: str | None, pathlib_type_aliases: set[str]) -> bool:
    return bool(name) and name in pathlib_type_aliases


def _is_forbidden_string_target(target: str) -> bool:
    if target in FORBIDDEN_STRING_TARGETS:
        return True
    return any(
        target.startswith(prefix)
        for prefix in ("pathlib.Path.", "pathlib.PosixPath.", "pathlib.WindowsPath.")
    )


def _is_pathlib_global_target_name(
    name: str | None, pathlib_type_aliases: set[str], pathlib_aliases: set[str]
) -> bool:
    if _is_pathlib_type_target_name(name, pathlib_type_aliases):
        return True
    if not name:
        return False
    return any(
        name in {f"{alias}.Path", f"{alias}.PosixPath", f"{alias}.WindowsPath"}
        for alias in pathlib_aliases
    )


def _resolve_target_and_attr(
    node: ast.Call, *, attr_keywords: tuple[str, ...]
) -> tuple[str | None, str | None]:
    keyword_map = {
        keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None
    }
    target_node = node.args[0] if node.args else keyword_map.get("target")
    attr_node = node.args[1] if len(node.args) >= 2 else None
    if attr_node is None:
        for key in attr_keywords:
            if key in keyword_map:
                attr_node = keyword_map[key]
                break
    target_name = _name(target_node) if target_node is not None else None
    attr_name: str | None = None
    if isinstance(attr_node, ast.Constant) and isinstance(attr_node.value, str):
        attr_name = attr_node.value
    return target_name, attr_name


def _find_forbidden_monkeypatches(test_file: Path) -> list[str]:
    module = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
    os_aliases = {"os"}
    pathlib_aliases = {"pathlib"}
    pathlib_type_aliases: set[str] = set()
    patch_aliases = {"patch"}

    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    os_aliases.add(alias.asname or alias.name)
                if alias.name == "pathlib":
                    pathlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "pathlib":
                for alias in node.names:
                    if alias.name in FORBIDDEN_PATHLIB_ATTRS:
                        pathlib_type_aliases.add(alias.asname or alias.name)
            if node.module in PATCH_IMPORT_SOURCES:
                for alias in node.names:
                    if alias.name == "patch":
                        patch_aliases.add(alias.asname or alias.name)

    offenders: list[str] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func_name = _name(node.func) or ""
        if func_name.endswith("setattr"):
            target_name, attr_name = _resolve_target_and_attr(node, attr_keywords=("name",))
            keyword_map = {
                keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None
            }
            target_node = node.args[0] if node.args else keyword_map.get("target")
            if isinstance(target_node, ast.Constant) and isinstance(target_node.value, str):
                if _is_forbidden_string_target(target_node.value):
                    offenders.append(
                        f"L{node.lineno}: monkeypatch.setattr('{target_node.value}', ...)"
                    )
            if attr_name == "name" and _is_os_target_name(target_name, os_aliases):
                offenders.append(f"L{node.lineno}: monkeypatch.setattr({target_name}, 'name', ...)")
            if attr_name in FORBIDDEN_PATHLIB_ATTRS and target_name in pathlib_aliases:
                offenders.append(
                    f"L{node.lineno}: monkeypatch.setattr({target_name}, '{attr_name}', ...)"
                )
            if _is_pathlib_global_target_name(target_name, pathlib_type_aliases, pathlib_aliases):
                offenders.append(f"L{node.lineno}: monkeypatch.setattr({target_name}, ...)")

        is_patch_call = func_name.endswith("patch") or func_name in patch_aliases
        is_patch_object_call = func_name.endswith("patch.object") or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "object"
            and _name(node.func.value) in patch_aliases
        )
        if is_patch_call or is_patch_object_call:
            keyword_map = {
                keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None
            }
            patch_target = node.args[0] if node.args else keyword_map.get("target")
            if isinstance(patch_target, ast.Constant) and isinstance(patch_target.value, str):
                if _is_forbidden_string_target(patch_target.value):
                    offenders.append(f"L{node.lineno}: {func_name}('{patch_target.value}', ...)")
            if is_patch_object_call:
                target_name, attr_name = _resolve_target_and_attr(
                    node, attr_keywords=("attribute", "name")
                )
                if attr_name == "name" and _is_os_target_name(target_name, os_aliases):
                    offenders.append(f"L{node.lineno}: {func_name}({target_name}, 'name', ...)")
                if attr_name in FORBIDDEN_PATHLIB_ATTRS and target_name in pathlib_aliases:
                    offenders.append(
                        f"L{node.lineno}: {func_name}({target_name}, '{attr_name}', ...)"
                    )
                if _is_pathlib_global_target_name(target_name, pathlib_type_aliases, pathlib_aliases):
                    offenders.append(f"L{node.lineno}: {func_name}({target_name}, ...)")
    return offenders


def test_scripts_tests_do_not_patch_global_path_or_os_runtime() -> None:
    scripts_tests_dir = Path(__file__).resolve().parent
    offenders: list[str] = []
    for test_file in sorted(scripts_tests_dir.glob("test_*.py")):
        if test_file.name == "test_path_patch_safety.py":
            continue
        for offender in _find_forbidden_monkeypatches(test_file):
            offenders.append(f"{test_file.name}:{offender}")
    assert not offenders, "Niedozwolone globalne monkeypatche:\n" + "\n".join(offenders)

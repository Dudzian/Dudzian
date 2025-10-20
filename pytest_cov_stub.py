"""Minimalny zamiennik ``pytest-cov`` oparty na :mod:`trace`."""

from __future__ import annotations

from dataclasses import dataclass, field
import ast
import importlib
import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, MutableMapping, Optional
from xml.etree import ElementTree as ET

import trace


_REPO_ROOT = Path(__file__).resolve().parent


def _pytest_cov_available() -> bool:
    return importlib.util.find_spec("pytest_cov") is not None


@dataclass
class _CoverageFile:
    path: Path
    executable_lines: set[int] = field(default_factory=set)


@dataclass
class _CoverageState:
    tracer: Optional[trace.Trace] = None
    files_by_module: dict[str, list[_CoverageFile]] = field(default_factory=dict)
    executed: MutableMapping[Path, set[int]] = field(default_factory=dict)
    reports: list[tuple[str, Optional[str]]] = field(default_factory=list)
    fail_under: Optional[float] = None
    term_missing: bool = False
    overall: float = 100.0
    module_summaries: list[tuple[str, float, int, int, list[tuple[_CoverageFile, int, int]]]] = field(
        default_factory=list
    )
    failed: bool = False


_STATE = _CoverageState()


def _reset_state(*, keep_tracer: bool = True) -> None:
    """Przygotuj świeży stan pomiaru pokrycia dla nowej sesji."""

    global _STATE

    previous_tracer = _STATE.tracer
    if not keep_tracer and previous_tracer is not None:
        sys.settrace(None)
        threading.settrace(None)
    tracer = previous_tracer if keep_tracer else None
    _STATE = _CoverageState()
    _STATE.tracer = tracer


def _maybe_start_tracer_early() -> None:
    if _pytest_cov_available():
        return
    if _STATE.tracer is not None:
        return
    if not any(arg.startswith("--cov") for arg in sys.argv):
        return
    tracer = trace.Trace(count=True, trace=False, ignoredirs=[str(Path(sys.prefix).resolve())])
    _STATE.tracer = tracer
    sys.settrace(tracer.globaltrace)
    threading.settrace(tracer.globaltrace)


_maybe_start_tracer_early()


def _iter_py_files(module: Any) -> Iterable[Path]:
    module_file = getattr(module, "__file__", None)
    if module_file:
        path = Path(module_file).resolve()
        if path.suffix == ".py":
            yield path
    module_path = getattr(module, "__path__", None)
    if module_path is None:
        return
    for entry in module_path:
        base = Path(entry).resolve()
        if not base.exists():
            continue
        for file_path in base.rglob("*.py"):
            if "__pycache__" in file_path.parts:
                continue
            yield file_path.resolve()


def _collect_module_files(module_name: str) -> list[_CoverageFile]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - chcemy raportować błąd użytkownikowi
        raise RuntimeError(f"Nie udało się zaimportować modułu '{module_name}': {exc}") from exc
    files: list[_CoverageFile] = []
    seen: set[Path] = set()
    for file_path in _iter_py_files(module):
        if not file_path.is_file():
            continue
        if file_path in seen:
            continue
        seen.add(file_path)
        files.append(_CoverageFile(path=file_path))
    return files


def _load_executable_lines(file: _CoverageFile) -> set[int]:
    if file.executable_lines:
        return file.executable_lines
    try:
        source = file.path.read_text(encoding="utf-8")
    except OSError:
        file.executable_lines = set()
        return file.executable_lines
    try:
        tree = ast.parse(source, filename=str(file.path))
    except SyntaxError:
        file.executable_lines = set()
        return file.executable_lines
    stmt_lines: set[int] = set()
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if isinstance(node, (ast.stmt, ast.ExceptHandler)):
            stmt_lines.add(int(lineno))
        for decorator in getattr(node, "decorator_list", ()):
            deco_lineno = getattr(decorator, "lineno", None)
            if deco_lineno is not None:
                stmt_lines.add(int(deco_lineno))
    lowered_source = source.splitlines()
    for lineno, text in enumerate(lowered_source, 1):
        if "pragma: no cover" in text.lower():
            stmt_lines.discard(lineno)
    file.executable_lines = stmt_lines
    return stmt_lines


def pytest_addoption(parser: Any) -> None:  # pragma: no cover - hook wywoływany przez pytest
    if _pytest_cov_available():
        return
    group = parser.getgroup("cov_stub")
    group.addoption(
        "--cov",
        action="append",
        default=[],
        metavar="MODULE",
        help="nazwy modułów/pakietów objętych pomiarem pokrycia",
    )
    group.addoption(
        "--cov-report",
        action="append",
        default=[],
        metavar="TYPE",
        help="typ raportu (obsługiwane: term, term-missing, xml[:ścieżka])",
    )
    group.addoption(
        "--cov-config",
        action="store",
        default=None,
        metavar="PATH",
        help="zachowane dla zgodności, ignorowane",
    )
    group.addoption(
        "--cov-append",
        action="store_true",
        default=False,
        help="zachowane dla zgodności, ignorowane",
    )
    group.addoption(
        "--cov-branch",
        action="store_true",
        default=False,
        help="zachowane dla zgodności, ignorowane",
    )
    group.addoption(
        "--cov-fail-under",
        action="store",
        default=None,
        type=float,
        metavar="MIN",
        help="minimalne pokrycie wymagane do uznania testów",
    )
    group.addoption(
        "--no-cov",
        action="store_true",
        default=False,
        help="wyłącza pomiar pokrycia",
    )


def _parse_reports(raw_reports: Iterable[str]) -> list[tuple[str, Optional[str]]]:
    reports: list[tuple[str, Optional[str]]] = []
    for entry in raw_reports:
        if not entry:
            continue
        if ":" in entry:
            kind, _, destination = entry.partition(":")
            reports.append((kind.strip(), destination.strip() or None))
        else:
            reports.append((entry.strip(), None))
    return reports


def pytest_configure(config: Any) -> None:  # pragma: no cover - hook wywoływany przez pytest
    if _pytest_cov_available():
        return
    options = config.option
    _reset_state()
    if getattr(options, "no_cov", False):
        _reset_state(keep_tracer=False)
        return
    modules = [mod for mod in getattr(options, "cov", []) if mod]
    if not modules:
        _reset_state(keep_tracer=False)
        return
    _STATE.files_by_module = {mod: _collect_module_files(mod) for mod in modules}
    _STATE.reports = _parse_reports(getattr(options, "cov_report", []))
    _STATE.fail_under = getattr(options, "cov_fail_under", None)
    _STATE.term_missing = any(kind == "term-missing" for kind, _ in _STATE.reports)
    if _STATE.tracer is None:
        tracer = trace.Trace(count=True, trace=False, ignoredirs=[str(Path(sys.prefix).resolve())])
        _STATE.tracer = tracer
        sys.settrace(tracer.globaltrace)
        threading.settrace(tracer.globaltrace)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:  # pragma: no cover - hook
    if _pytest_cov_available():
        return
    tracer = _STATE.tracer
    if tracer is None:
        return
    sys.settrace(None)
    threading.settrace(None)
    results = tracer.results()
    executed: MutableMapping[Path, set[int]] = {}
    for (filename, lineno), count in results.counts.items():
        if count <= 0:
            continue
        path = Path(filename).resolve()
        executed.setdefault(path, set()).add(int(lineno))
    _STATE.executed = executed
    _STATE.overall, _STATE.module_summaries = _summaries()
    if _STATE.fail_under is not None and _STATE.overall < _STATE.fail_under:
        _STATE.failed = True
        session.config._cov_stub_failed = True
        session.exitstatus = max(session.exitstatus, 1)


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _summaries() -> tuple[float, list[tuple[str, float, int, int, list[tuple[_CoverageFile, int, int]]]]]:
    module_summaries: list[tuple[str, float, int, int, list[tuple[_CoverageFile, int, int]]]] = []
    total_executable = 0
    total_covered = 0
    for module, files in _STATE.files_by_module.items():
        file_entries: list[tuple[_CoverageFile, int, int]] = []
        module_exec = 0
        module_cov = 0
        for file in files:
            resolved = file.path.resolve()
            executable_lines = _load_executable_lines(file)
            if not executable_lines:
                continue
            executed = _STATE.executed.get(resolved, set())
            covered = len(executable_lines & executed)
            file_entries.append((file, len(executable_lines), covered))
            module_exec += len(executable_lines)
            module_cov += covered
        if module_exec == 0:
            continue
        coverage_pct = (module_cov / module_exec) * 100 if module_exec else 100.0
        module_summaries.append((module, coverage_pct, module_cov, module_exec, file_entries))
        total_executable += module_exec
        total_covered += module_cov
    overall = (total_covered / total_executable) * 100 if total_executable else 100.0
    return overall, module_summaries


def _render_term(module_summaries: list[tuple[str, float, int, int, list[tuple[_CoverageFile, int, int]]]]) -> str:
    lines = ["Pokrycie (trace stub)"]
    header = f"{'Moduł':40} {'Pokrycie':>10} {'Linie':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for module, pct, covered, total, _ in module_summaries:
        lines.append(f"{module:40} {pct:9.2f}% {covered:5d}/{total:<5d}")
    return "\n".join(lines)


def _render_missing(module_summaries: list[tuple[str, float, int, int, list[tuple[_CoverageFile, int, int]]]]) -> str:
    lines = ["Brakujące linie (trace stub)"]
    for module, _, _, _, files in module_summaries:
        for file, total, covered in files:
            if total == covered:
                continue
            executable = _load_executable_lines(file)
            executed = _STATE.executed.get(file.path.resolve(), set())
            missing = sorted(executable - executed)
            if not missing:
                continue
            rel = _relative_path(file.path)
            lines.append(f"- {rel}: {','.join(str(m) for m in missing)}")
    if len(lines) == 1:
        lines.append("(brak)")
    return "\n".join(lines)


def _write_xml(overall: float, module_summaries: list[tuple[str, float, int, int, list[tuple[_CoverageFile, int, int]]]], destination: Path) -> None:
    root = ET.Element(
        "coverage",
        attrib={
            "line-rate": f"{overall / 100:.4f}",
            "branch-rate": "0",
            "timestamp": str(int(time.time())),
            "version": "pytest-cov-stub",
        },
    )
    sources = ET.SubElement(root, "sources")
    ET.SubElement(sources, "source").text = str(_REPO_ROOT)
    packages_el = ET.SubElement(root, "packages")
    for module, pct, covered, total, files in module_summaries:
        pkg = ET.SubElement(
            packages_el,
            "package",
            attrib={
                "name": module,
                "line-rate": f"{pct / 100:.4f}",
                "branch-rate": "0",
                "complexity": "0",
            },
        )
        classes_el = ET.SubElement(pkg, "classes")
        for file, _, _ in files:
            executable = _load_executable_lines(file)
            if not executable:
                continue
            executed = _STATE.executed.get(file.path.resolve(), set())
            rel_path = _relative_path(file.path)
            class_el = ET.SubElement(
                classes_el,
                "class",
                attrib={
                    "name": rel_path.replace("/", "."),
                    "filename": rel_path,
                    "line-rate": f"{(len(executable & executed) / len(executable)) if executable else 1.0:.4f}",
                    "branch-rate": "0",
                    "complexity": "0",
                },
            )
            lines_el = ET.SubElement(class_el, "lines")
            for line_no in sorted(executable):
                hits = "1" if line_no in executed else "0"
                ET.SubElement(lines_el, "line", attrib={"number": str(line_no), "hits": hits})
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:  # pragma: no cover
    if _pytest_cov_available():
        return
    if not _STATE.files_by_module:
        return
    if not _STATE.module_summaries:
        terminalreporter.write_line("(cov stub) brak danych do raportu")
        return
    if (
        any(kind in {"term", "term-missing"} for kind, _ in _STATE.reports)
        or not _STATE.reports
    ):
        terminalreporter.write_line(_render_term(_STATE.module_summaries))
    if _STATE.term_missing:
        terminalreporter.write_line(_render_missing(_STATE.module_summaries))
    for kind, dest in _STATE.reports:
        if kind == "xml":
            output = Path(dest or "coverage.xml")
            _write_xml(_STATE.overall, _STATE.module_summaries, output)
            terminalreporter.write_line(f"(cov stub) zapisano raport XML do {output}")
    if _STATE.failed:
        terminalreporter.write_line(
            f"ERROR: pokrycie {_STATE.overall:.2f}% poniżej progu {_STATE.fail_under:.2f}%"
        )

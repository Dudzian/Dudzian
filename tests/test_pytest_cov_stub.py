from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


pytest_plugins = ("pytester",)


def test_cov_stub_counts_unexecuted_files(pytester):
    package = pytester.mkpydir("sample_pkg")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text("""\
def foo():
    return 1
""", encoding="utf-8")
    (package / "b.py").write_text("""\
def bar():
    return 2
""", encoding="utf-8")

    pytester.makepyfile(
        """\
from sample_pkg import a


def test_foo():
    assert a.foo() == 1
"""
    )

    repo_root = Path(__file__).resolve().parent.parent
    pytester.makeini(f"[pytest]\npythonpath = {repo_root}\n")
    result = pytester.runpytest_subprocess(
        "-p",
        "pytest_cov_stub",
        "--cov=sample_pkg",
        "--cov-report=term",
        "--cov-report=xml",
        "--cov-fail-under=60",
    )

    result.stdout.fnmatch_lines([
        "*sample_pkg*50.00%*2/4*",
        "*ERROR: pokrycie 50.00% poniżej progu 60.00%*",
    ])
    assert result.ret != 0

    coverage_xml = pytester.path / "coverage.xml"
    assert coverage_xml.exists()

    tree = ET.parse(coverage_xml)
    filenames = {
        class_el.attrib["filename"]
        for package_el in tree.getroot().findall("./packages/package")
        for class_el in package_el.findall("./classes/class")
    }
    assert any(name.endswith("sample_pkg/b.py") for name in filenames)

    b_class = next(
        class_el
        for package_el in tree.getroot().findall("./packages/package")
        for class_el in package_el.findall("./classes/class")
        if class_el.attrib["filename"].endswith("sample_pkg/b.py")
    )
    assert any(line.attrib["hits"] == "0" for line in b_class.findall("./lines/line"))

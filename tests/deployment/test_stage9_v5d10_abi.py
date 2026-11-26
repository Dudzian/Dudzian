"""Exercise Observer ABI roles with synthetic records and inert file bytes."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from scripts import stage9_v5d4_abi as legacy
from scripts import stage9_v5d10_abi as observer
from tests.deployment import test_stage9_v5d4_abi as static_abi

pytestmark = [pytest.mark.security, pytest.mark.unit]
environment = static_abi.environment


@pytest.fixture
def observer_environment(environment: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    request, expectations, records = environment
    library = tmp_path / "libpython3.12.so.1.0"
    library.write_bytes(b"synthetic shared library")
    static_abi._pin(expectations, library)
    request["additional_extensions"].append(str(library))
    records[str(library)] = static_abi._elf(
        soname=library.name,
        needed=("libc.so.6",),
        defined_python_symbols=("PyInit__abc", "PyInit__ast"),
    )
    monkeypatch.setattr(observer, "_BASE_LIBPYTHON", library)

    def analyzer(path: str) -> dict[str, Any]:
        assert Path(path).name == "stage9_v5d4_abi.py"
        return {"diagnose": partial(legacy.diagnose, reader=lambda path: records[str(path)])}

    monkeypatch.setattr(observer.runpy, "run_path", analyzer)
    return request, expectations, records, library


def test_builtin_initializers_in_observer_shared_library(observer_environment: Any) -> None:
    request, expectations, records, library = observer_environment
    with pytest.raises(legacy.Refusal, match="EXTENSION_INIT_SYMBOL_MISSING"):
        legacy.diagnose(request, expectations, reader=lambda path: records[str(path)])

    report = observer.diagnose(request, expectations)
    assert report["status"] == "STATIC_ABI_MATCH"
    assert report["elf_measurements"][str(library)] == expectations["sha256"][str(library)]
    assert report["native_load"] == "NOT_EXECUTED_REQUIRES_SEPARATE_EXPLICIT_AUTHORIZATION"


def test_observer_requires_declared_base_libpython(observer_environment: Any) -> None:
    request, expectations, _, library = observer_environment
    request["additional_extensions"].remove(str(library))
    with pytest.raises(legacy.Refusal, match="OBSERVER_LIBPYTHON_NOT_DECLARED"):
        observer.diagnose(request, expectations)


def test_observer_requires_exact_shared_library_role(observer_environment: Any) -> None:
    request, expectations, records, library = observer_environment
    records[str(library)] = replace(records[str(library)], soname="different.so")
    with pytest.raises(legacy.Refusal, match="OBSERVER_LIBPYTHON_ROLE_MISMATCH"):
        observer.diagnose(request, expectations)


def test_observer_does_not_skip_other_extension_initializers(observer_environment: Any) -> None:
    request, expectations, records, _ = observer_environment
    binding = request["libvirt_extension"]
    records[binding] = replace(records[binding], defined_python_symbols=("PyInit_wrong",))
    with pytest.raises(legacy.Refusal, match="EXTENSION_INIT_SYMBOL_MISSING"):
        observer.diagnose(request, expectations)

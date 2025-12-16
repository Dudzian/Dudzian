from __future__ import annotations

import sitecustomize
from pathlib import Path


def test_demotes_numpy_checkout_out_of_sys_path(tmp_path, monkeypatch) -> None:
    numpy_checkout = tmp_path / "numpy"
    numpy_checkout.mkdir()
    (numpy_checkout / "__init__.py").write_text("# numpy stub", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[build-system]", encoding="utf-8")

    sys_path = ["", "/safe/site-packages", str(tmp_path), "/other"]
    monkeypatch.setattr(sitecustomize.sys, "path", list(sys_path))

    sitecustomize._demote_numpy_source_shadows()  # type: ignore[attr-defined]

    assert sitecustomize.sys.path[:3] == ["", "/safe/site-packages", "/other"]
    assert sitecustomize.sys.path[-1] == str(tmp_path)


def test_keeps_site_packages_numpy(tmp_path, monkeypatch) -> None:
    numpy_site_pkg = tmp_path / "venv" / "Lib" / "site-packages"
    (numpy_site_pkg / "numpy").mkdir(parents=True)
    (numpy_site_pkg / "numpy" / "__init__.py").write_text("# real numpy", encoding="utf-8")

    sys_path = ["", str(numpy_site_pkg)]
    monkeypatch.setattr(sitecustomize.sys, "path", list(sys_path))

    sitecustomize._demote_numpy_source_shadows()  # type: ignore[attr-defined]

    assert sitecustomize.sys.path == sys_path


def test_demotes_cwd_when_it_is_numpy_checkout(tmp_path, monkeypatch) -> None:
    """Jeżeli bieżący katalog jest checkoutem NumPy, pusty wpis sys.path też jest demotowany."""

    # Przygotuj atrapę drzewa źródłowego NumPy:
    numpy_checkout = tmp_path / "numpy"
    numpy_checkout.mkdir()
    (numpy_checkout / "__init__.py").write_text("# numpy stub", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[build-system]", encoding="utf-8")

    # Udajemy, że test uruchamiany jest z katalogu checkoutu NumPy.
    monkeypatch.chdir(tmp_path)

    # Pusty wpis odpowiada cwd; drugi wpis reprezentuje site-packages.
    sys_path = ["", "/safe/site-packages"]
    monkeypatch.setattr(sitecustomize.sys, "path", list(sys_path))

    sitecustomize._demote_numpy_source_shadows()  # type: ignore[attr-defined]

    # Oczekujemy, że site-packages pozostanie przed checkoutem NumPy.
    assert sitecustomize.sys.path[0] == "/safe/site-packages"
    assert sitecustomize.sys.path[-1] == ""


def test_demotes_direct_numpy_package_dir(tmp_path, monkeypatch) -> None:
    """Bezpośredni wpis na katalog pakietu ``numpy`` też powinien być demotowany."""

    numpy_checkout = tmp_path
    numpy_package_dir = numpy_checkout / "numpy"
    numpy_package_dir.mkdir()
    (numpy_package_dir / "__init__.py").write_text("# numpy stub", encoding="utf-8")
    (numpy_checkout / "setup.py").write_text("# build script", encoding="utf-8")

    sys_path = ["/safe/site-packages", str(numpy_package_dir), "/other"]
    monkeypatch.setattr(sitecustomize.sys, "path", list(sys_path))

    sitecustomize._demote_numpy_source_shadows()  # type: ignore[attr-defined]

    assert sitecustomize.sys.path[:2] == ["/safe/site-packages", "/other"]
    assert sitecustomize.sys.path[-1] == str(numpy_package_dir)

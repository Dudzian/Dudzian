from __future__ import annotations

import sitecustomize


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

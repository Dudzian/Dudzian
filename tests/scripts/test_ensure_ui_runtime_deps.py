from scripts.dev import ensure_ui_runtime_deps


def test_google_protobuf_is_checked_by_ui_runtime_preflight() -> None:
    dependencies = ensure_ui_runtime_deps._runtime_dependencies()

    google_protobuf = [
        dependency for dependency in dependencies if dependency.module == "google.protobuf"
    ]

    assert len(google_protobuf) == 1
    assert google_protobuf[0].requirement == "protobuf>=5"


def test_missing_google_protobuf_maps_to_protobuf_package(monkeypatch) -> None:
    dependency = ensure_ui_runtime_deps.RuntimeDependency(
        module="google.protobuf",
        requirement="protobuf>=5",
        reason="test",
    )

    def fake_import_module(name: str) -> object:
        if name == "google.protobuf":
            raise ImportError("No module named google.protobuf")
        return object()

    monkeypatch.setattr(ensure_ui_runtime_deps.importlib, "import_module", fake_import_module)

    missing = ensure_ui_runtime_deps._missing_dependencies((dependency,))

    assert missing == [dependency]
    assert missing[0].requirement.startswith("protobuf")

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import textwrap
import uuid
from pathlib import Path
from types import ModuleType
from unittest import mock


def _create_logging_helper_module(tmp_path: Path, content: str) -> tuple[str, Path]:
    module_name = f"_optional_exports_logging_{uuid.uuid4().hex}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(textwrap.dedent(content))
    importlib.invalidate_caches()
    return module_name, module_path


def _remove_helper_module(module_name: str) -> None:
    sys.modules.pop(module_name, None)

import pytest


def test_lazy_loader_restores_deleted_attribute() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    original = runtime.TradingController

    # Usuń atrybut, aby wymusić przejście przez __getattr__.
    runtime.__dict__.pop("TradingController", None)

    restored = runtime.TradingController
    assert restored is original


def test_dir_lists_lazy_exports() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    names = dir(runtime)
    assert "TradingController" in names
    assert "DailyTrendPipeline" in names
    assert "create_trading_controller" in names


def test_unknown_lazy_attribute_raises_attribute_error() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    try:
        getattr(runtime, "_missing_optional_symbol_")
    except AttributeError as exc:
        assert "_missing_optional_symbol_" in str(exc)
    else:  # pragma: no cover - oczekujemy wyjątku
        raise AssertionError("Expected AttributeError for missing optional symbol")


def test_list_optional_exports_reports_lazy_names() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    names = runtime.list_optional_exports()
    assert names == sorted(runtime._LAZY_OPTIONAL_EXPORTS)


def test_list_optional_exports_filters_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    missing_name = "_SyntheticMissingExport"

    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    all_names = runtime.list_optional_exports()
    assert missing_name in all_names

    available_only = runtime.list_optional_exports(available_only=True)
    assert missing_name not in available_only


def test_is_optional_export_available_handles_known_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")

    assert runtime.is_optional_export_available("TradingController") is True

    missing_name = "_SyntheticMissingExport"
    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    assert runtime.is_optional_export_available(missing_name) is False


def test_is_optional_export_available_rejects_unknown_name() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(ValueError):
        runtime.is_optional_export_available("NotRegisteredOptionalSymbol")


def test_describe_optional_exports_reports_status_objects() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    statuses = runtime.describe_optional_exports()
    assert statuses

    trading_status = next(s for s in statuses if s.name == "TradingController")
    assert trading_status.available is True
    assert trading_status.error is None
    assert trading_status.module.endswith("controller")
    assert trading_status.attribute == "TradingController"
    assert isinstance(trading_status.cached, bool)


def test_snapshot_optional_exports_reflects_registry_state() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    runtime.evict_optional_export("TradingController")
    snapshot = runtime.snapshot_optional_exports()

    assert snapshot.registered == dict(runtime._LAZY_OPTIONAL_EXPORTS)
    assert "TradingController" not in snapshot.cached_names
    assert "TradingController" in snapshot.available_names
    
    runtime.require_optional_export("TradingController")
    updated_snapshot = runtime.snapshot_optional_exports()

    assert "TradingController" in updated_snapshot.cached_names
    assert "TradingController" in updated_snapshot.available_names
    assert not updated_snapshot.missing


def test_probe_optional_export_reports_cached_state() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    runtime.evict_optional_export("TradingController")
    uncached_status = runtime.probe_optional_export("TradingController")
    assert uncached_status.available is True
    assert uncached_status.cached is False

    runtime.require_optional_export("TradingController")
    cached_status = runtime.probe_optional_export("TradingController")
    assert cached_status.available is True
    assert cached_status.cached is True


def test_probe_optional_export_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    missing_name = "_SyntheticMissingExport"

    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    status = runtime.probe_optional_export(missing_name)
    assert status.name == missing_name
    assert status.available is False
    assert status.cached is False
    assert status.error is not None


def test_probe_optional_export_rejects_unknown_symbol() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(ValueError):
        runtime.probe_optional_export("NotRegisteredOptionalSymbol")


def test_describe_optional_exports_includes_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    missing_name = "_SyntheticMissingExport"

    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    statuses = runtime.describe_optional_exports()
    status_map = {status.name: status for status in statuses}
    assert missing_name in status_map

    missing_status = status_map[missing_name]
    assert missing_status.available is False
    assert missing_status.error is not None
    assert "_module_does_not_exist" in missing_status.error

    available_only = runtime.describe_optional_exports(available_only=True)
    available_names = {status.name for status in available_only}
    assert missing_name not in available_names


def test_snapshot_optional_exports_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    missing_name = "_SyntheticMissingExport"

    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    snapshot = runtime.snapshot_optional_exports()
    assert missing_name in snapshot.registered
    assert missing_name not in snapshot.cached_names
    assert missing_name not in snapshot.available_names

    assert missing_name in snapshot.missing


def test_restore_optional_exports_removes_dynamic_symbols() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    baseline = runtime.snapshot_optional_exports()

    runtime.register_optional_export(
        "_SyntheticExport",
        "math",
        "pi",
    )
    runtime.require_optional_export("_SyntheticExport")

    assert "_SyntheticExport" in runtime._LAZY_OPTIONAL_EXPORTS
    assert "_SyntheticExport" in runtime.__dict__

    runtime.restore_optional_exports(baseline)

    assert "_SyntheticExport" not in runtime._LAZY_OPTIONAL_EXPORTS
    assert "_SyntheticExport" not in runtime.__dict__
    assert "_SyntheticExport" not in runtime.__all__


def test_restore_optional_exports_reload_cached() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    runtime.require_optional_export("TradingController")
    baseline = runtime.snapshot_optional_exports()

    runtime.evict_optional_export("TradingController")
    assert runtime.is_optional_export_cached("TradingController") is False

    runtime.restore_optional_exports(baseline, reload_cached=True)

    assert runtime.is_optional_export_cached("TradingController") is True


def test_restore_optional_exports_resets_uncached_entries() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    runtime.evict_optional_export("TradingController")
    baseline = runtime.snapshot_optional_exports()

    runtime.require_optional_export("TradingController")
    assert runtime.is_optional_export_cached("TradingController") is True

    runtime.restore_optional_exports(baseline)

    assert runtime.is_optional_export_cached("TradingController") is False


def test_restore_optional_exports_rejects_invalid_snapshot() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    baseline = runtime.snapshot_optional_exports()
    registered = dict(baseline.registered)
    registered.pop("TradingController")

    invalid_snapshot = runtime.OptionalExportRegistrySnapshot(
        registered=registered,
        cached_names=frozenset(),
        statuses={},
    )

    with pytest.raises(ValueError):
        runtime.restore_optional_exports(invalid_snapshot)


def test_restore_optional_exports_rejects_invalid_type() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(TypeError):
        runtime.restore_optional_exports(object())  # type: ignore[arg-type]


def test_require_optional_export_returns_symbol() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    value = runtime.require_optional_export("TradingController")
    assert value is runtime.TradingController


def test_require_optional_export_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("bot_core.runtime")

    missing_name = "_SyntheticMissingExport"
    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    with pytest.raises(runtime.OptionalExportUnavailableError) as exc_info:
        runtime.require_optional_export(missing_name)

    error = exc_info.value
    assert error.status.name == missing_name
    assert error.status.available is False
    assert "_module_does_not_exist" in str(error)


def test_require_optional_export_rejects_unknown_symbol() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(ValueError):
        runtime.require_optional_export("NotRegisteredOptionalSymbol")


def test_get_optional_export_returns_symbol() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    value = runtime.get_optional_export("TradingController")
    assert value is runtime.TradingController


def test_get_optional_export_returns_default_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    missing_name = "_SyntheticMissingExport"

    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    sentinel = object()
    value = runtime.get_optional_export(missing_name, default=sentinel)
    assert value is sentinel


def test_get_optional_export_raises_when_missing_without_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    missing_name = "_SyntheticMissingExport"

    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    with pytest.raises(runtime.OptionalExportUnavailableError):
        runtime.get_optional_export(missing_name)


def test_get_optional_export_rejects_unknown_symbol() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(ValueError):
        runtime.get_optional_export("NotRegisteredOptionalSymbol")


def test_register_optional_export_adds_mapping() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    synthetic_name = "_SyntheticExport"

    runtime.register_optional_export(
        synthetic_name,
        "math",
        "pi",
    )

    try:
        value = runtime.require_optional_export(synthetic_name)
        import math

        assert value == math.pi
        assert synthetic_name in runtime.list_optional_exports()
    finally:
        runtime.unregister_optional_export(synthetic_name)


def test_register_optional_export_requires_override() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    synthetic_name = "_SyntheticExport"

    runtime.register_optional_export(
        synthetic_name,
        "math",
        "pi",
    )

    try:
        with pytest.raises(ValueError):
            runtime.register_optional_export(
                synthetic_name,
                "math",
                "tau",
            )

        runtime.register_optional_export(
            synthetic_name,
            "math",
            "tau",
            override=True,
        )

        import math

        value = runtime.require_optional_export(synthetic_name)
        assert value == math.tau
    finally:
        runtime.unregister_optional_export(synthetic_name)


def test_unregister_optional_export_removes_dynamic_symbol() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    synthetic_name = "_SyntheticExport"

    runtime.register_optional_export(
        synthetic_name,
        "math",
        "pi",
    )

    import math

    value = runtime.require_optional_export(synthetic_name)
    assert value == math.pi

    runtime.unregister_optional_export(synthetic_name)

    assert synthetic_name not in runtime._LAZY_OPTIONAL_EXPORTS
    assert synthetic_name not in runtime.__dict__

    with pytest.raises(ValueError):
        runtime.require_optional_export(synthetic_name)


def test_unregister_optional_export_rejects_builtin() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(ValueError):
        runtime.unregister_optional_export("TradingController")


def test_temporary_optional_export_registers_and_cleans_up() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    synthetic_name = "_SyntheticTemporaryExport"

    with runtime.temporary_optional_export(
        synthetic_name,
        "math",
        "pi",
    ) as value:
        import math

        assert value == math.pi
        assert runtime.require_optional_export(synthetic_name) == math.pi

    assert synthetic_name not in runtime._LAZY_OPTIONAL_EXPORTS
    assert synthetic_name not in runtime.__dict__


def test_temporary_optional_export_overrides_existing_symbol() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    original = runtime.require_optional_export("TradingController")

    with runtime.temporary_optional_export(
        "TradingController",
        "math",
        "tau",
        override=True,
    ) as value:
        import math

        assert value == math.tau
        assert runtime.require_optional_export("TradingController") == math.tau

    restored = runtime.require_optional_export("TradingController")
    assert restored is original


def test_temporary_optional_export_requires_override_for_existing() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(ValueError):
        with runtime.temporary_optional_export(
            "TradingController",
            "math",
            "pi",
        ):
            pass


def test_refresh_optional_export_updates_cached_value() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    synthetic_module_name = "tests.synthetic_refresh_module"

    module = ModuleType(synthetic_module_name)
    module.VALUE = 1
    sys.modules[synthetic_module_name] = module

    runtime.register_optional_export(
        "_SyntheticRefreshExport",
        synthetic_module_name,
        "VALUE",
    )

    try:
        initial = runtime.require_optional_export("_SyntheticRefreshExport")
        assert initial == 1

        module.VALUE = 2

        refreshed = runtime.refresh_optional_export("_SyntheticRefreshExport")
        assert refreshed == 2
        assert runtime.require_optional_export("_SyntheticRefreshExport") == 2
    finally:
        runtime.unregister_optional_export("_SyntheticRefreshExport")
        sys.modules.pop(synthetic_module_name, None)


def test_refresh_optional_export_with_module_reload(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    module_path = tmp_path / "synthetic_reload_module.py"
    module_path.write_text("VALUE = 1\n")

    monkeypatch.syspath_prepend(str(tmp_path))

    runtime.register_optional_export(
        "_SyntheticReloadExport",
        "synthetic_reload_module",
        "VALUE",
    )

    try:
        first = runtime.require_optional_export("_SyntheticReloadExport")
        assert first == 1

        module_path.write_text("VALUE = 2\n")
        stats = module_path.stat()
        os.utime(module_path, (stats.st_atime + 5, stats.st_mtime + 5))
        importlib.invalidate_caches()

        refreshed = runtime.refresh_optional_export(
            "_SyntheticReloadExport",
            reload_module=True,
        )
        assert refreshed == 2
    finally:
        runtime.unregister_optional_export("_SyntheticReloadExport")
        sys.modules.pop("synthetic_reload_module", None)


def test_refresh_optional_export_handles_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    missing_name = "_SyntheticMissingExport"

    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    with pytest.raises(runtime.OptionalExportUnavailableError):
        runtime.refresh_optional_export(missing_name)


def test_refresh_optional_export_rejects_unknown_symbol() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(ValueError):
        runtime.refresh_optional_export("NotRegisteredOptionalSymbol")


def test_ensure_optional_exports_loads_available_and_reports_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    missing_name = "_SyntheticMissingExport"

    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    loaded, missing = runtime.ensure_optional_exports(
        ["TradingController", missing_name]
    )

    assert "TradingController" in loaded
    assert missing_name not in loaded

    assert missing_name in missing
    status = missing[missing_name]
    assert status.available is False
    assert "_module_does_not_exist" in status.error


def test_ensure_optional_exports_require_all(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    missing_name = "_SyntheticMissingExport"

    monkeypatch.setitem(
        runtime._LAZY_OPTIONAL_EXPORTS,
        missing_name,
        ("bot_core.runtime._module_does_not_exist", "Missing"),
    )

    with pytest.raises(runtime.OptionalExportUnavailableError) as exc_info:
        runtime.ensure_optional_exports(
            ["TradingController", missing_name],
            require_all=True,
        )

    status = exc_info.value.status
    assert status.name == missing_name


def test_is_optional_export_cached_reflects_runtime_cache_state() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    # Upewnij się, że zaczynamy bez zcache'owanej wartości.
    runtime.evict_optional_export("TradingController")
    assert runtime.is_optional_export_cached("TradingController") is False

    runtime.require_optional_export("TradingController")
    assert runtime.is_optional_export_cached("TradingController") is True


def test_evict_optional_export_clears_cached_value() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    runtime.require_optional_export("TradingController")
    assert runtime.is_optional_export_cached("TradingController") is True

    evicted = runtime.evict_optional_export("TradingController")
    assert evicted is True
    assert runtime.is_optional_export_cached("TradingController") is False

    # Ponowna próba powinna zwrócić False, bo cache jest pusty.
    evicted_again = runtime.evict_optional_export("TradingController")
    assert evicted_again is False


def test_optional_export_cache_helpers_reject_unknown_symbol() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(ValueError):
        runtime.is_optional_export_cached("NotRegisteredOptionalSymbol")

    with pytest.raises(ValueError):
        runtime.evict_optional_export("NotRegisteredOptionalSymbol")


def test_optional_exports_snapshot_serialization_roundtrip() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    snapshot = runtime.snapshot_optional_exports()
    payload = runtime.optional_exports_snapshot_to_dict(snapshot)

    # Symulujemy przejście przez JSON, aby upewnić się, że struktura jest serializowalna.
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    restored = runtime.optional_exports_snapshot_from_dict(decoded)

    assert restored.registered == snapshot.registered
    assert restored.cached_names == snapshot.cached_names
    assert restored.statuses == snapshot.statuses


def test_optional_exports_snapshot_to_dict_validates_input() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(TypeError):
        runtime.optional_exports_snapshot_to_dict(object())  # type: ignore[arg-type]


def test_optional_exports_snapshot_from_dict_validates_input() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    snapshot = runtime.snapshot_optional_exports()
    payload = runtime.optional_exports_snapshot_to_dict(snapshot)

    # Usuń wymagany klucz, aby wymusić błąd walidacji.
    payload.pop("registered")

    with pytest.raises(ValueError):
        runtime.optional_exports_snapshot_from_dict(payload)

    with pytest.raises(TypeError):
        runtime.optional_exports_snapshot_from_dict(object())  # type: ignore[arg-type]


def test_optional_exports_snapshot_json_roundtrip() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    runtime.require_optional_export("TradingController")
    snapshot = runtime.snapshot_optional_exports()

    encoded_pretty = runtime.optional_exports_snapshot_to_json(snapshot, indent=2)
    restored = runtime.optional_exports_snapshot_from_json(encoded_pretty)

    assert restored == snapshot

    encoded_bytes = encoded_pretty.encode("utf-8")
    restored_from_bytes = runtime.optional_exports_snapshot_from_json(encoded_bytes)
    assert restored_from_bytes == snapshot


def test_optional_exports_snapshot_json_validates_input() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    snapshot = runtime.snapshot_optional_exports()

    with pytest.raises(TypeError):
        runtime.optional_exports_snapshot_to_json("not a snapshot")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.optional_exports_snapshot_from_json(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        runtime.optional_exports_snapshot_from_json("{invalid json}")


def test_optional_exports_snapshot_file_roundtrip(tmp_path: Path) -> None:
    runtime = importlib.import_module("bot_core.runtime")

    runtime.require_optional_export("TradingController")
    snapshot = runtime.snapshot_optional_exports()

    target = tmp_path / "runtime_snapshot.json"
    runtime.optional_exports_snapshot_to_file(snapshot, target, indent=2)

    restored = runtime.optional_exports_snapshot_from_file(target)
    assert restored == snapshot


def test_optional_exports_snapshot_to_file_validates_snapshot(tmp_path: Path) -> None:
    runtime = importlib.import_module("bot_core.runtime")

    target = tmp_path / "invalid_snapshot.json"

    with pytest.raises(TypeError):
        runtime.optional_exports_snapshot_to_file(object(), target)  # type: ignore[arg-type]


def test_optional_exports_snapshot_from_file_validates_contents(tmp_path: Path) -> None:
    runtime = importlib.import_module("bot_core.runtime")

    target = tmp_path / "broken_snapshot.json"
    target.write_text("{invalid json}")

    with pytest.raises(ValueError):
        runtime.optional_exports_snapshot_from_file(target)


def test_diff_optional_exports_snapshots_detects_no_changes() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    snapshot = runtime.snapshot_optional_exports()
    diff = runtime.diff_optional_exports_snapshots(snapshot, snapshot)

    assert diff.has_changes is False
    assert diff.added == {}
    assert diff.removed == {}
    assert diff.changed_targets == {}
    assert diff.status_changes == {}
    assert diff.cache_gained == frozenset()
    assert diff.cache_lost == frozenset()


def test_diff_optional_exports_snapshots_reports_added_and_status() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    baseline = runtime.snapshot_optional_exports()

    runtime.register_optional_export(
        "_SyntheticDiffExport",
        "math",
        "pi",
    )

    try:
        updated = runtime.snapshot_optional_exports()
        diff = runtime.diff_optional_exports_snapshots(baseline, updated)

        assert "_SyntheticDiffExport" in diff.added
        assert diff.has_changes is True

        before_status, after_status = diff.status_changes["_SyntheticDiffExport"]
        assert before_status is None
        assert after_status is not None and after_status.available is True
    finally:
        runtime.unregister_optional_export("_SyntheticDiffExport")


def test_diff_optional_exports_snapshots_reports_removed() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    runtime.register_optional_export(
        "_SyntheticDiffExport",
        "math",
        "pi",
    )

    try:
        baseline = runtime.snapshot_optional_exports()
    finally:
        runtime.unregister_optional_export("_SyntheticDiffExport")

    updated = runtime.snapshot_optional_exports()
    diff = runtime.diff_optional_exports_snapshots(baseline, updated)

    assert diff.removed["_SyntheticDiffExport"] == ("math", "pi")
    before_status, after_status = diff.status_changes["_SyntheticDiffExport"]
    assert before_status is not None and before_status.available is True
    assert after_status is None


def test_diff_optional_exports_snapshots_reports_target_changes() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    runtime.register_optional_export(
        "_SyntheticDiffExport",
        "math",
        "pi",
    )

    try:
        baseline = runtime.snapshot_optional_exports()

        runtime.register_optional_export(
            "_SyntheticDiffExport",
            "math",
            "tau",
            override=True,
        )

        updated = runtime.snapshot_optional_exports()
        diff = runtime.diff_optional_exports_snapshots(baseline, updated)

        assert diff.changed_targets["_SyntheticDiffExport"] == (
            ("math", "pi"),
            ("math", "tau"),
        )
    finally:
        runtime.unregister_optional_export("_SyntheticDiffExport")


def test_diff_optional_exports_snapshots_reports_cache_changes() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    runtime.evict_optional_export("TradingController")
    baseline = runtime.snapshot_optional_exports()

    runtime.require_optional_export("TradingController")
    updated = runtime.snapshot_optional_exports()

    diff = runtime.diff_optional_exports_snapshots(baseline, updated)

    assert diff.cache_gained == frozenset({"TradingController"})
    before_status, after_status = diff.status_changes["TradingController"]
    assert before_status is not None and before_status.cached is False
    assert after_status is not None and after_status.cached is True


def test_diff_optional_exports_snapshots_validates_input() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    snapshot = runtime.snapshot_optional_exports()

    with pytest.raises(TypeError):
        runtime.diff_optional_exports_snapshots(object(), snapshot)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.diff_optional_exports_snapshots(snapshot, object())  # type: ignore[arg-type]


def _sample_diff(runtime: ModuleType) -> object:
    return runtime.OptionalExportRegistryDiff(
        added={"Added": ("pkg.added", "symbol")},
        removed={"Removed": ("pkg.removed", "old_symbol")},
        changed_targets={
            "Changed": (("pkg.before", "attr_before"), ("pkg.after", "attr_after"))
        },
        status_changes={
            "Added": (
                None,
                runtime.OptionalExportStatus(
                    name="Added",
                    module="pkg.added",
                    attribute="symbol",
                    available=True,
                    cached=False,
                    error=None,
                ),
            ),
            "Removed": (
                runtime.OptionalExportStatus(
                    name="Removed",
                    module="pkg.removed",
                    attribute="old_symbol",
                    available=True,
                    cached=True,
                    error=None,
                ),
                None,
            ),
            "Changed": (
                runtime.OptionalExportStatus(
                    name="Changed",
                    module="pkg.before",
                    attribute="attr_before",
                    available=False,
                    cached=False,
                    error="ImportError: missing",
                ),
                runtime.OptionalExportStatus(
                    name="Changed",
                    module="pkg.after",
                    attribute="attr_after",
                    available=True,
                    cached=False,
                    error=None,
                ),
            ),
        },
        cache_gained=frozenset({"Added", "Changed"}),
        cache_lost=frozenset({"Removed"}),
    )


def test_optional_exports_diff_dict_roundtrip() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    diff = _sample_diff(runtime)
    payload = runtime.optional_exports_diff_to_dict(diff)
    restored = runtime.optional_exports_diff_from_dict(payload)

    assert restored == diff


def test_optional_exports_diff_json_roundtrip() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    diff = _sample_diff(runtime)
    json_payload = runtime.optional_exports_diff_to_json(diff, indent=2)
    restored = runtime.optional_exports_diff_from_json(json_payload)

    assert restored == diff


def test_optional_exports_diff_file_roundtrip(tmp_path: Path) -> None:
    runtime = importlib.import_module("bot_core.runtime")

    diff = _sample_diff(runtime)
    target = tmp_path / "diff.json"

    runtime.optional_exports_diff_to_file(diff, target, indent=2)
    restored = runtime.optional_exports_diff_from_file(target)

    assert restored == diff


def test_optional_exports_diff_serialization_validates_input(tmp_path: Path) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    diff = _sample_diff(runtime)

    with pytest.raises(TypeError):
        runtime.optional_exports_diff_to_dict(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.optional_exports_diff_from_dict(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.optional_exports_diff_to_json(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.optional_exports_diff_from_json(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        runtime.optional_exports_diff_from_json("not json")

    target = tmp_path / "invalid.json"
    target.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError):
        runtime.optional_exports_diff_from_file(target)

    # ensure helper still writes out valid payloads
    runtime.optional_exports_diff_to_file(diff, target)
    restored = runtime.optional_exports_diff_from_file(target)
    assert restored == diff


def test_format_optional_export_status_reports_details() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    status = runtime.OptionalExportStatus(
        name="Synthetic",
        module="pkg.module",
        attribute="symbol",
        available=False,
        cached=False,
        error="ImportError: missing dependency",
    )

    with_error = runtime.format_optional_export_status(status)
    assert "Synthetic" in with_error
    assert "pkg.module.symbol" in with_error
    assert "missing" in with_error

    without_error = runtime.format_optional_export_status(
        status,
        include_error=False,
    )
    assert "missing dependency" not in without_error


def test_format_optional_exports_summary_sorts_and_formats() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    statuses = [
        runtime.OptionalExportStatus(
            name="Beta",
            module="pkg.beta",
            attribute="attr",
            available=True,
            cached=True,
            error=None,
        ),
        runtime.OptionalExportStatus(
            name="Alpha",
            module="pkg.alpha",
            attribute="value",
            available=False,
            cached=False,
            error="ImportError",
        ),
    ]

    summary = runtime.format_optional_exports_summary(statuses)
    lines = summary.splitlines()

    assert lines[0] == "Optional exports:"
    assert lines[1].startswith("  - Alpha")
    assert "pkg.alpha.value" in lines[1]
    assert "error=ImportError" in lines[1]
    assert lines[2].startswith("  - Beta")


def test_format_optional_exports_diff_outputs_sections() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    diff = runtime.OptionalExportRegistryDiff(
        added={"Added": ("pkg.added", "symbol")},
        removed={"Removed": ("pkg.removed", "symbol")},
        changed_targets={"Changed": (("pkg.old", "attr"), ("pkg.new", "attr"))},
        status_changes={
            "Status": (
                runtime.OptionalExportStatus(
                    name="Status",
                    module="pkg.before",
                    attribute="attr",
                    available=False,
                    cached=False,
                    error="ImportError",
                ),
                runtime.OptionalExportStatus(
                    name="Status",
                    module="pkg.after",
                    attribute="attr",
                    available=True,
                    cached=False,
                    error=None,
                ),
            )
        },
        cache_gained=frozenset({"Added"}),
        cache_lost=frozenset({"Removed"}),
    )

    summary = runtime.format_optional_exports_diff(diff)
    assert summary.startswith("Optional export diff:")
    assert "Added:" in summary
    assert "Removed:" in summary
    assert "Changed targets:" in summary
    assert "Status changes:" in summary
    assert "Cache gained:" in summary
    assert "Cache lost:" in summary


def test_log_optional_export_status_logs_and_returns_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    status = runtime.probe_optional_export("TradingController")

    with caplog.at_level(logging.INFO, logger="bot_core.runtime"):
        message = runtime.log_optional_export_status(status)

    assert status.name in message
    assert message in caplog.messages


def test_log_optional_exports_summary_defaults_to_describe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with caplog.at_level(logging.INFO, logger="bot_core.runtime"):
        message = runtime.log_optional_exports_summary(include_errors=False)

    assert message.startswith("Optional exports:")
    assert any("Optional exports:" in record.message for record in caplog.records)


def test_log_optional_exports_summary_accepts_custom_statuses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    status = runtime.probe_optional_export("TradingController")

    with caplog.at_level(logging.WARNING, logger="bot_core.runtime"):
        message = runtime.log_optional_exports_summary(
            [status],
            include_errors=False,
            logger=None,
            level="WARNING",
        )

    assert "TradingController" in message
    assert caplog.records[0].levelno == logging.WARNING


def test_log_optional_export_status_accepts_logger_adapter_and_extra(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    status = runtime.probe_optional_export("TradingController")

    adapter_logger = logging.getLogger("bot_core.runtime.adapter")

    class MergingAdapter(logging.LoggerAdapter):
        def process(self, msg: str, kwargs: dict[str, object]) -> tuple[str, dict[str, object]]:
            merged_extra = dict(self.extra)
            merged_extra.update(kwargs.get("extra", {}))
            kwargs["extra"] = merged_extra
            return msg, kwargs

    adapter = MergingAdapter(adapter_logger, {"component": "adapter"})

    with caplog.at_level(logging.INFO, logger="bot_core.runtime.adapter"):
        message = runtime.log_optional_export_status(
            status,
            logger=adapter,
            extra={"phase": "test"},
        )

    record = next(record for record in caplog.records if record.message == message)
    assert getattr(record, "component") == "adapter"
    assert getattr(record, "phase") == "test"


def test_log_optional_exports_diff_logs_changes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    baseline = runtime.snapshot_optional_exports()

    runtime.register_optional_export(
        "_SyntheticDiffLogExport",
        "math",
        "pi",
    )

    try:
        updated = runtime.snapshot_optional_exports()
        diff = runtime.diff_optional_exports_snapshots(baseline, updated)

        with caplog.at_level(logging.INFO, logger="bot_core.runtime"):
            message = runtime.log_optional_exports_diff(diff)

        assert "Optional export diff" in message
        assert any("Optional export diff" in record.message for record in caplog.records)
    finally:
        runtime.unregister_optional_export("_SyntheticDiffLogExport")


def test_log_optional_export_helpers_forward_extra_and_stacklevel() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    status = runtime.probe_optional_export("TradingController")

    mock_logger = mock.Mock()

    runtime.log_optional_export_status(
        status,
        logger=mock_logger,
        extra={"source": "test"},
        stacklevel=3,
    )

    mock_logger.log.assert_called_once()
    _, kwargs = mock_logger.log.call_args
    assert kwargs["extra"] == {"source": "test"}
    assert kwargs["stacklevel"] == 3


def test_log_optional_export_helpers_validate_inputs() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    diff = runtime.diff_optional_exports_snapshots(
        runtime.snapshot_optional_exports(),
        runtime.snapshot_optional_exports(),
    )
    status = runtime.probe_optional_export("TradingController")

    with pytest.raises(TypeError):
        runtime.log_optional_export_status(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.log_optional_exports_summary(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.log_optional_exports_diff(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.log_optional_exports_diff(diff, level=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.log_optional_export_status(status, extra=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.log_optional_exports_summary(extra=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.log_optional_export_status(status, stacklevel="1")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        runtime.log_optional_exports_summary(stacklevel=0)


def test_set_optional_exports_logger_overrides_default(caplog: pytest.LogCaptureFixture) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    status = runtime.probe_optional_export("TradingController")
    mock_logger = mock.Mock()

    runtime.set_optional_exports_logger(mock_logger)
    try:
        runtime.log_optional_export_status(status, logger=None)
        mock_logger.log.assert_called_once()
    finally:
        runtime.set_optional_exports_logger(None)

    with caplog.at_level(logging.INFO, logger="bot_core.runtime"):
        message = runtime.log_optional_export_status(status)

    assert any(record.name == "bot_core.runtime" and record.message == message for record in caplog.records)


def test_temporary_optional_exports_logger_restores_previous(caplog: pytest.LogCaptureFixture) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    status = runtime.probe_optional_export("TradingController")
    mock_logger = mock.Mock()

    with runtime.temporary_optional_exports_logger(mock_logger):
        runtime.log_optional_export_status(status)

    mock_logger.log.assert_called_once()

    with caplog.at_level(logging.INFO, logger="bot_core.runtime"):
        message = runtime.log_optional_export_status(status)

    assert any(record.name == "bot_core.runtime" and record.message == message for record in caplog.records)


def test_configure_optional_exports_logging_sets_up_logger() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    try:
        configured = runtime.configure_optional_exports_logging(
            level="DEBUG",
            clear_handlers=True,
            ensure_handler=True,
            propagate=False,
        )
        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
        assert configured.level == logging.DEBUG
        assert configured.propagate is False
        assert configured.handlers  # przynajmniej jeden handler został podpięty
    finally:
        runtime.set_optional_exports_logger(None)


def test_configure_optional_exports_logging_accepts_custom_logger() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    custom_logger = logging.getLogger("bot_core.runtime.custom_optional")
    custom_logger.handlers.clear()

    formatter = logging.Formatter("%(levelname)s:%(message)s")
    new_handler = logging.NullHandler()

    configured = runtime.configure_optional_exports_logging(
        logger=custom_logger,
        handler=new_handler,
        formatter=formatter,
        clear_handlers=True,
        propagate=True,
        set_as_default=False,
    )

    assert configured is custom_logger
    assert configured.propagate is True
    assert configured.handlers == [new_handler]
    assert new_handler.formatter is formatter
    assert runtime.get_optional_exports_logger() is not configured


def test_configure_optional_exports_logging_validates_inputs() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging(logger=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging(logger_name=123)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging(handler=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging(formatter=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        runtime.configure_optional_exports_logging(
            formatter=logging.Formatter("%(message)s"),
            clear_handlers=True,
            set_as_default=False,
        )

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging(propagate="yes")  # type: ignore[arg-type]


def test_set_optional_exports_logger_validates_input() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(TypeError):
        runtime.set_optional_exports_logger(object())  # type: ignore[arg-type]


def _build_dict_logging_config() -> dict[str, object]:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"simple": {"format": "%(message)s"}},
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "bot_core.runtime.optional_exports": {
                "level": "INFO",
                "handlers": ["stdout"],
                "propagate": False,
            }
        },
    }


def test_configure_optional_exports_logging_from_dict_sets_default_logger() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    target_logger = logging.getLogger("bot_core.runtime.optional_exports")
    target_logger.handlers.clear()

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_dict(
            _build_dict_logging_config()
        )
        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
        assert configured.propagate is False
        assert any(
            isinstance(handler, logging.StreamHandler)
            for handler in configured.handlers
        )
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()


def test_configure_optional_exports_logging_from_dict_respects_set_as_default() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)
    default_logger = runtime.get_optional_exports_logger()
    configured = runtime.configure_optional_exports_logging_from_dict(
        _build_dict_logging_config(),
        set_as_default=False,
    )

    try:
        assert runtime.get_optional_exports_logger() is default_logger
        assert configured is logging.getLogger("bot_core.runtime.optional_exports")
    finally:
        runtime.set_optional_exports_logger(None)
        configured.handlers.clear()


def test_configure_optional_exports_logging_from_dict_validates_inputs() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging_from_dict(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging_from_dict({}, logger_name=123)  # type: ignore[arg-type]


def test_configure_optional_exports_logging_from_file_sets_default_logger(
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)
    target_logger = logging.getLogger("bot_core.runtime.optional_exports")
    target_logger.handlers.clear()

    config_text = textwrap.dedent(
        """
        [loggers]
        keys=root,optional

        [handlers]
        keys=stream

        [formatters]
        keys=simple

        [logger_root]
        level=WARNING
        handlers=stream

        [logger_optional]
        level=INFO
        handlers=stream
        qualname=bot_core.runtime.optional_exports
        propagate=0

        [handler_stream]
        class=StreamHandler
        level=INFO
        formatter=simple
        args=(sys.stdout,)

        [formatter_simple]
        format=%(message)s
        """
    )

    config_path = tmp_path / "logging.ini"
    config_path.write_text(config_text)

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_file(
            config_path,
            defaults={"sys": sys},
            disable_existing_loggers=False,
        )
        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
        assert not configured.propagate
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()


def test_configure_optional_exports_logging_from_file_respects_set_as_default(
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)
    default_logger = runtime.get_optional_exports_logger()

    config_path = tmp_path / "logging.ini"
    config_path.write_text(
        textwrap.dedent(
            """
            [loggers]
            keys=root,optional

            [handlers]
            keys=stream

            [formatters]
            keys=simple

            [logger_root]
            level=WARNING
            handlers=stream

            [logger_optional]
            level=DEBUG
            handlers=stream
            qualname=bot_core.runtime.optional_exports
            propagate=0

            [handler_stream]
            class=StreamHandler
            level=DEBUG
            formatter=simple
            args=(sys.stdout,)

            [formatter_simple]
            format=%(message)s
            """
        )
    )

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_file(
            config_path,
            defaults={"sys": sys},
            disable_existing_loggers=False,
            set_as_default=False,
        )

        assert runtime.get_optional_exports_logger() is default_logger
        assert configured is logging.getLogger("bot_core.runtime.optional_exports")
        assert configured.level == logging.DEBUG
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()


def test_configure_optional_exports_logging_from_file_validates_inputs(
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    config_path = tmp_path / "logging.ini"
    config_path.write_text("[loggers]\nkeys=root\n\n[handlers]\nkeys=stream\n")

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging_from_file(
            config_path,
            defaults=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging_from_file(
            config_path,
            disable_existing_loggers="no",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging_from_file(
            config_path,
            logger_name=123,  # type: ignore[arg-type]
        )


def test_configure_optional_exports_logging_from_python_callable_returns_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    module_name, _ = _create_logging_helper_module(
        tmp_path,
        """
        import logging

        def build_config():
            return {
                "version": 1,
                "disable_existing_loggers": False,
                "handlers": {
                    "null": {"class": "logging.NullHandler"}
                },
                "loggers": {
                    "bot_core.runtime.optional_exports": {
                        "handlers": ["null"],
                        "level": "INFO",
                        "propagate": False,
                    }
                },
            }
        """,
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_python(
            f"{module_name}:build_config",
            set_as_default=False,
        )

        assert isinstance(configured, logging.Logger)
        assert configured is logging.getLogger("bot_core.runtime.optional_exports")
        assert configured.level == logging.INFO
        assert runtime.get_optional_exports_logger() is not configured
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()
        _remove_helper_module(module_name)


def test_configure_optional_exports_logging_from_python_returns_logger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    module_name, _ = _create_logging_helper_module(
        tmp_path,
        """
        import logging

        def provide_logger():
            logger = logging.getLogger("bot_core.runtime.optional_exports")
            logger.handlers.clear()
            handler = logging.NullHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            return logger
        """,
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_python(
            f"{module_name}:provide_logger",
            set_as_default=True,
        )

        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
        assert configured.level == logging.DEBUG
        assert any(
            isinstance(handler, logging.NullHandler) for handler in configured.handlers
        )
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()
        _remove_helper_module(module_name)


def test_configure_optional_exports_logging_from_python_attribute_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    module_name, _ = _create_logging_helper_module(
        tmp_path,
        """
        CONFIG = {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "null": {"class": "logging.NullHandler"}
            },
            "loggers": {
                "bot_core.runtime.optional_exports": {
                    "handlers": ["null"],
                    "level": "WARNING",
                    "propagate": False,
                }
            },
        }
        """,
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_python(
            module_name,
            attribute="CONFIG",
            set_as_default=False,
        )

        assert isinstance(configured, logging.Logger)
        assert configured.level == logging.WARNING
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()
        _remove_helper_module(module_name)


def test_configure_optional_exports_logging_from_python_validates_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    module_name, _ = _create_logging_helper_module(
        tmp_path,
        """
        BAD_VALUE = 42
        """,
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    try:
        with pytest.raises(TypeError):
            runtime.configure_optional_exports_logging_from_python(
                module_name,
                attribute="BAD_VALUE",
            )
    finally:
        runtime.set_optional_exports_logger(None)
        _remove_helper_module(module_name)


def test_parse_optional_exports_logging_spec_variants() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    parsed_json = runtime.parse_optional_exports_logging_spec(
        "json:  {\n  \"version\": 1\n}"
    )
    assert parsed_json.kind == "json"
    assert parsed_json.origin == "prefixed"
    assert parsed_json.attribute is None
    assert "\"version\"" in parsed_json.value

    inline_payload = json.dumps({"version": 1})
    parsed_inline = runtime.parse_optional_exports_logging_spec(inline_payload)
    assert parsed_inline.kind == "json"
    assert parsed_inline.origin == "inline"

    parsed_file = runtime.parse_optional_exports_logging_spec("/tmp/config.ini")
    assert parsed_file.kind == "file"
    assert parsed_file.origin == "bare"

    parsed_python = runtime.parse_optional_exports_logging_spec(
        "python:pkg.module:factory"
    )
    assert parsed_python.kind == "python"
    assert parsed_python.value == "pkg.module"
    assert parsed_python.attribute == "factory"


def test_parse_optional_exports_logging_spec_validates() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(TypeError):
        runtime.parse_optional_exports_logging_spec(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        runtime.parse_optional_exports_logging_spec("   ")

    with pytest.raises(ValueError):
        runtime.parse_optional_exports_logging_spec("json:")

    with pytest.raises(ValueError):
        runtime.parse_optional_exports_logging_spec("file:")

    with pytest.raises(ValueError):
        runtime.parse_optional_exports_logging_spec("python:")

    with pytest.raises(ValueError):
        runtime.parse_optional_exports_logging_spec("yaml:spec")


def test_configure_optional_exports_logging_from_parsed_spec_json() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    payload = json.dumps(_build_dict_logging_config())
    parsed = runtime.parse_optional_exports_logging_spec(f"json:{payload}")

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_parsed_spec(parsed)
        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()


def test_configure_optional_exports_logging_from_parsed_spec_file(
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    config_text = textwrap.dedent(
        """
        [loggers]
        keys=root,optional

        [handlers]
        keys=stream

        [formatters]
        keys=simple

        [logger_root]
        level=WARNING
        handlers=stream

        [logger_optional]
        level=INFO
        handlers=stream
        qualname=bot_core.runtime.optional_exports
        propagate=0

        [handler_stream]
        class=StreamHandler
        level=INFO
        formatter=simple
        args=(sys.stdout,)

        [formatter_simple]
        format=%(message)s
        """
    )

    config_path = tmp_path / "logging.ini"
    config_path.write_text(config_text)

    parsed = runtime.parse_optional_exports_logging_spec(f"file:{config_path}")

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_parsed_spec(
            parsed,
            defaults={"sys": sys},
            disable_existing_loggers=False,
        )
        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()


def test_configure_optional_exports_logging_from_parsed_spec_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    module_name, module_path = _create_logging_helper_module(
        tmp_path,
        """
        import logging

        def build_config():
            return {
                "version": 1,
                "disable_existing_loggers": False,
                "handlers": {"null": {"class": "logging.NullHandler"}},
                "loggers": {
                    "bot_core.runtime.optional_exports": {
                        "handlers": ["null"],
                        "level": "ERROR",
                        "propagate": False,
                    }
                },
            }
        """,
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    parsed = runtime.parse_optional_exports_logging_spec(
        f"python:{module_name}:build_config"
    )

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_parsed_spec(parsed)
        assert isinstance(configured, logging.Logger)
        assert configured.level == logging.ERROR
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()
        _remove_helper_module(module_name)
        module_path.unlink(missing_ok=True)


def test_configure_optional_exports_logging_from_parsed_spec_validates() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging_from_parsed_spec(  # type: ignore[arg-type]
            "not-a-spec"
        )


def test_configure_optional_exports_logging_from_spec_json() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    payload = json.dumps(_build_dict_logging_config())

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_spec(f"json:{payload}")
        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()


def test_configure_optional_exports_logging_from_spec_inline_json() -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    payload = json.dumps(_build_dict_logging_config())

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_spec(payload)
        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()


def test_configure_optional_exports_logging_from_spec_file(tmp_path: Path) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    config_text = textwrap.dedent(
        """
        [loggers]
        keys=root,optional

        [handlers]
        keys=stream

        [formatters]
        keys=simple

        [logger_root]
        level=WARNING
        handlers=stream

        [logger_optional]
        level=INFO
        handlers=stream
        qualname=bot_core.runtime.optional_exports
        propagate=0

        [handler_stream]
        class=StreamHandler
        level=INFO
        formatter=simple
        args=(sys.stdout,)

        [formatter_simple]
        format=%(message)s
        """
    )

    config_path = tmp_path / "logging.ini"
    config_path.write_text(config_text)

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_spec(
            f"file:{config_path}",
            defaults={"sys": sys},
            disable_existing_loggers=False,
        )
        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()


def test_configure_optional_exports_logging_from_spec_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    module_name, module_path = _create_logging_helper_module(
        tmp_path,
        """
        import logging

        def build_config():
            return {
                "version": 1,
                "disable_existing_loggers": False,
                "handlers": {"null": {"class": "logging.NullHandler"}},
                "loggers": {
                    "bot_core.runtime.optional_exports": {
                        "handlers": ["null"],
                        "level": "ERROR",
                        "propagate": False,
                    }
                },
            }
        """
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_spec(
            f"python:{module_name}:build_config"
        )
        assert isinstance(configured, logging.Logger)
        assert configured.level == logging.ERROR
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()
        _remove_helper_module(module_name)
        module_path.unlink(missing_ok=True)


def test_configure_optional_exports_logging_from_spec_validates() -> None:
    runtime = importlib.import_module("bot_core.runtime")

    with pytest.raises(TypeError):
        runtime.configure_optional_exports_logging_from_spec(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        runtime.configure_optional_exports_logging_from_spec("   ")

    with pytest.raises(ValueError):
        runtime.configure_optional_exports_logging_from_spec("json:")

    with pytest.raises(ValueError):
        runtime.configure_optional_exports_logging_from_spec("file:")

    with pytest.raises(FileNotFoundError):
        runtime.configure_optional_exports_logging_from_spec("missing-file.ini")

    with pytest.raises(ValueError):
        runtime.configure_optional_exports_logging_from_spec("python:module_only")

    with pytest.raises(ValueError):
        runtime.configure_optional_exports_logging_from_spec("yaml:spec")


def test_configure_optional_exports_logging_from_env_json(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    payload = json.dumps(_build_dict_logging_config())
    monkeypatch.setenv("BOT_CORE_OPTIONAL_EXPORTS_LOGGING", payload)

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_env()
        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()


def test_configure_optional_exports_logging_from_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    config_text = textwrap.dedent(
        """
        [loggers]
        keys=root,optional

        [handlers]
        keys=stream

        [formatters]
        keys=simple

        [logger_root]
        level=WARNING
        handlers=stream

        [logger_optional]
        level=INFO
        handlers=stream
        qualname=bot_core.runtime.optional_exports
        propagate=0

        [handler_stream]
        class=StreamHandler
        level=INFO
        formatter=simple
        args=(sys.stdout,)

        [formatter_simple]
        format=%(message)s
        """
    )

    config_path = tmp_path / "logging.ini"
    config_path.write_text(config_text)

    monkeypatch.setenv("BOT_CORE_OPTIONAL_EXPORTS_LOGGING", f"file:{config_path}")

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_env(
            defaults={"sys": sys},
            disable_existing_loggers=False,
        )
        assert isinstance(configured, logging.Logger)
        assert runtime.get_optional_exports_logger() is configured
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()


def test_configure_optional_exports_logging_from_env_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    module_name, module_path = _create_logging_helper_module(
        tmp_path,
        """
        import logging

        def build_config():
            return {
                "version": 1,
                "disable_existing_loggers": False,
                "handlers": {
                    "null": {"class": "logging.NullHandler"}
                },
                "loggers": {
                    "bot_core.runtime.optional_exports": {
                        "handlers": ["null"],
                        "level": "ERROR",
                        "propagate": False,
                    }
                },
            }
        """,
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(
        "BOT_CORE_OPTIONAL_EXPORTS_LOGGING",
        f"python:{module_name}:build_config",
    )

    configured: logging.Logger | None = None
    try:
        configured = runtime.configure_optional_exports_logging_from_env()
        assert isinstance(configured, logging.Logger)
        assert configured.level == logging.ERROR
    finally:
        runtime.set_optional_exports_logger(None)
        if configured is not None:
            configured.handlers.clear()
        _remove_helper_module(module_name)
        module_path.unlink(missing_ok=True)


def test_configure_optional_exports_logging_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)
    monkeypatch.delenv("BOT_CORE_OPTIONAL_EXPORTS_LOGGING", raising=False)

    with pytest.raises(KeyError):
        runtime.configure_optional_exports_logging_from_env()

    assert (
        runtime.configure_optional_exports_logging_from_env(missing_ok=True) is None
    )


def test_configure_optional_exports_logging_from_env_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("bot_core.runtime")
    runtime.set_optional_exports_logger(None)

    monkeypatch.setenv("BOT_CORE_OPTIONAL_EXPORTS_LOGGING", "json: not json")
    with pytest.raises(ValueError):
        runtime.configure_optional_exports_logging_from_env()

    monkeypatch.setenv("BOT_CORE_OPTIONAL_EXPORTS_LOGGING", "file:")
    with pytest.raises(ValueError):
        runtime.configure_optional_exports_logging_from_env()

    monkeypatch.setenv("BOT_CORE_OPTIONAL_EXPORTS_LOGGING", "missing-file.ini")
    with pytest.raises(FileNotFoundError):
        runtime.configure_optional_exports_logging_from_env()

    monkeypatch.setenv("BOT_CORE_OPTIONAL_EXPORTS_LOGGING", "python:module_only")
    with pytest.raises(ValueError):
        runtime.configure_optional_exports_logging_from_env()

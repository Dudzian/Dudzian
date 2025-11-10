import json
import os
import sys
from pathlib import Path

import pytest

import pathbootstrap as pb


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv(pb.ENV_SENTINELS, raising=False)
    monkeypatch.delenv(pb.ENV_ALLOW_GIT, raising=False)
    monkeypatch.delenv(pb.ENV_ADDITIONAL_PATHS, raising=False)
    monkeypatch.delenv(pb.ENV_CONFIG_FILES, raising=False)
    monkeypatch.delenv(pb.ENV_CONFIG_INLINE, raising=False)
    monkeypatch.delenv(pb.ENV_PROFILES, raising=False)
    monkeypatch.delenv(pb.ENV_ROOT_HINT, raising=False)
    monkeypatch.delenv(pb.ENV_MAX_DEPTH, raising=False)
    yield


def test_resolve_sentinels_prefers_environment(monkeypatch):
    monkeypatch.setenv(pb.ENV_SENTINELS, os.pathsep.join([" one ", "", "two", "three "]))
    assert pb._resolve_sentinels(("default", "value")) == ("one", "two", "three")


def test_resolve_sentinels_rejects_empty(monkeypatch):
    with pytest.raises(ValueError):
        pb._resolve_sentinels(())
    monkeypatch.setenv(pb.ENV_SENTINELS, os.pathsep.join(["   ", ""]))
    with pytest.raises(ValueError):
        pb._resolve_sentinels(("ignored",))


def test_resolve_allow_git_flag_from_environment(monkeypatch):
    assert pb._resolve_allow_git_flag(None) is False
    monkeypatch.setenv(pb.ENV_ALLOW_GIT, "1")
    assert pb._resolve_allow_git_flag(None) is True
    assert pb._resolve_allow_git_flag(False) is False


def test_coerce_str_sequence_validations():
    assert pb._coerce_str_sequence("single", key="sentinels") == ("single",)
    assert pb._coerce_str_sequence([" first ", "second"], key="sentinels") == (
        "first",
        "second",
    )
    with pytest.raises(TypeError):
        pb._coerce_str_sequence(123, key="sentinels")
    with pytest.raises(TypeError):
        pb._coerce_str_sequence(["ok", 42], key="sentinels")
    with pytest.raises(ValueError):
        pb._coerce_str_sequence(["   "], key="sentinels")


def test_coerce_optional_path_handles_relative(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    result = pb._coerce_optional_path("../target", key="root_hint", base_dir=config_dir)
    assert result == str(target.resolve())
    with pytest.raises(TypeError):
        pb._coerce_optional_path(123, key="root_hint", base_dir=config_dir)
    with pytest.raises(ValueError):
        pb._coerce_optional_path("   ", key="root_hint", base_dir=config_dir)


def test_coerce_optional_int_variants():
    assert pb._coerce_optional_int(None, key="max_depth") is None
    assert pb._coerce_optional_int("5", key="max_depth") == 5
    assert pb._coerce_optional_int(3, key="max_depth", min_value=1) == 3
    with pytest.raises(ValueError):
        pb._coerce_optional_int("   ", key="max_depth")
    with pytest.raises(ValueError):
        pb._coerce_optional_int(2, key="max_depth", min_value=5)
    with pytest.raises(TypeError):
        pb._coerce_optional_int(3.14, key="max_depth")


def test_normalize_config_mapping_with_profiles(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    include_file = tmp_path / "extras.txt"
    include_file.write_text("/tmp/custom", encoding="utf-8")
    mapping = {
        "root_hint": "./repo",
        "sentinels": [" pyproject.toml ", "setup.cfg"],
        "additional_paths": [" lib ", "./plugins"],
        "additional_path_files": [include_file.name],
        "allow_git": "false",
        "max_depth": "3",
        "use_env_additional_paths": "no",
        "position": "append",
        "path_style": "posix",
        "pythonpath_var": " CUSTOM_PYTHONPATH ",
        "default_profiles": [" ci "],
        "profiles": {
            "ci": {
                "extends": ["base"],
                "additional_paths": ["ci"],
                "allow_git": False,
            },
            "base": {
                "additional_paths": ["base"],
            },
        },
    }
    normalized = pb._normalize_config_mapping(mapping, source=config_path)
    assert normalized["root_hint"].endswith("repo")
    assert normalized["sentinels"] == ("pyproject.toml", "setup.cfg")
    assert normalized["additional_paths"] == ("lib", "./plugins")
    assert include_file.resolve().as_posix() in normalized["additional_path_files"]
    assert normalized["allow_git"] is False
    assert normalized["max_depth"] == 3
    assert normalized["use_env_additional_paths"] is False
    assert normalized["position"] == "append"
    assert normalized["path_style"] == "posix"
    assert normalized["pythonpath_var"] == "CUSTOM_PYTHONPATH"
    assert normalized["default_profiles"] == ("ci",)
    profiles = normalized["profiles"]
    assert set(profiles) == {"ci", "base"}
    assert profiles["ci"].extends == ("base",)
    assert profiles["ci"].values["additional_paths"] == ("ci",)


def test_normalize_config_mapping_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        pb._normalize_config_mapping({"unsupported": True}, source=config_path)


def test_merge_config_data_combines_profiles():
    accumulated: dict[str, object] = {"profiles": {"base": pb.ProfileConfig((), {"a": 1})}}
    pb._merge_config_data(
        accumulated,
        {
            "profiles": {
                "ci": pb.ProfileConfig(("base",), {"b": 2}),
            },
            "position": "append",
        },
    )
    assert accumulated["position"] == "append"
    assert set(accumulated["profiles"]) == {"base", "ci"}


def test_serialize_config_definition_roundtrip():
    mapping = {
        "additional_paths": ("a", "b"),
        "profiles": {
            "ci": pb.ProfileConfig(("base",), {"values": ("x",)}),
        },
    }
    serialized = pb._serialize_config_definition(mapping)
    assert serialized["additional_paths"] == ["a", "b"]
    ci_profile = serialized["profiles"]["ci"]
    assert ci_profile["extends"] == ["base"]
    assert ci_profile["values"]["values"] == ["x"]


def test_resolve_profile_values_and_cycle_detection():
    profiles = {
        "base": pb.ProfileConfig((), {"root": "value"}),
        "child": pb.ProfileConfig(("base",), {"child": True}),
    }
    cache: dict[str, dict[str, object]] = {}
    resolved = pb._resolve_profile_values("child", profiles, cache, ())
    assert resolved == {"root": "value", "child": True}
    assert cache["child"] is resolved
    cyclic = {"loop": pb.ProfileConfig(("loop",), {})}
    with pytest.raises(ValueError):
        pb._resolve_profile_values("loop", cyclic, {}, ())


def test_resolve_additional_paths_merges_env(monkeypatch, tmp_path):
    env_path = tmp_path / "env"
    env_path.mkdir()
    repo_root = tmp_path
    monkeypatch.setenv(
        pb.ENV_ADDITIONAL_PATHS,
        os.pathsep.join([str(env_path), str(repo_root / "lib"), str(repo_root / "local")]),
    )
    result = pb._resolve_additional_paths(repo_root, ["./local", repo_root / "lib"], include_env=True)
    assert str(env_path) in result
    assert result.count(str(env_path)) == 1
    resolved_local = str((repo_root / "local").resolve())
    assert resolved_local in result
    assert result.count(resolved_local) == 1


def test_format_env_assignment_variants():
    assert pb._format_env_assignment("VAR", "/tmp", "plain") == "VAR=/tmp"
    assert pb._format_env_assignment("VAR", "/tmp", "posix") == "export VAR=/tmp"
    assert pb._format_env_assignment("VAR", "/tmp", "powershell") == "$Env:VAR = '/tmp'"
    assert pb._format_env_assignment("VAR", "/tmp", "cmd") == "set VAR=/tmp"
    with pytest.raises(ValueError):
        pb._format_env_assignment("VAR", "/tmp", "unknown")


def test_repo_on_sys_path_context(tmp_path):
    sentinel = tmp_path / "pyproject.toml"
    sentinel.write_text("[project]", encoding="utf-8")
    original = list(sys.path)
    try:
        with pb.repo_on_sys_path(root_hint=tmp_path, sentinels=(sentinel.name,)) as repo_root:
            assert repo_root == tmp_path.resolve()
            assert str(repo_root) in sys.path
        assert original == sys.path
    finally:
        sys.path[:] = original


def test_load_config_file_supports_json_and_toml(tmp_path):
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    assert pb._load_config_file(json_path) == {"value": 1}

    toml_path = tmp_path / "config.toml"
    toml_path.write_text("value = 2\n", encoding="utf-8")
    assert pb._load_config_file(toml_path) == {"value": 2}

    invalid_ext = tmp_path / "config.txt"
    invalid_ext.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(ValueError):
        pb._load_config_file(invalid_ext)

    invalid_top = tmp_path / "invalid.json"
    invalid_top.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        pb._load_config_file(invalid_top)


def test_load_configurations_with_includes(tmp_path):
    base = tmp_path / "base.json"
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"sentinels": ["custom"], "position": "append"}), encoding="utf-8")
    base.write_text(
        json.dumps(
            {
                "includes": [extra.name],
                "additional_paths": ["plugins"],
                "profiles": {"ci": {"additional_paths": ["ci"]}},
            }
        ),
        encoding="utf-8",
    )

    config, used, edges = pb._load_configurations([base])
    assert config["sentinels"] == ("custom",)
    assert config["position"] == "append"
    assert used == (str(extra.resolve()), str(base.resolve()))
    assert edges == ((str(base.resolve()), str(extra.resolve())),)
    profiles = config["profiles"]
    assert profiles["ci"].values["additional_paths"] == ("ci",)


def test_load_configurations_cycle_detection(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"includes": [second.name]}), encoding="utf-8")
    second.write_text(json.dumps({"includes": [first.name]}), encoding="utf-8")
    with pytest.raises(ValueError):
        pb._load_configurations([first])


def test_load_sentinels_and_additional_paths_from_file(tmp_path):
    sentinel_file = tmp_path / "sentinels.txt"
    sentinel_file.write_text("# comment\nfoo\n\nbar\n", encoding="utf-8")
    assert pb._load_sentinels_from_file(sentinel_file) == ("foo", "bar")

    empty_sentinel = tmp_path / "empty.txt"
    empty_sentinel.write_text("\n# comment only\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pb._load_sentinels_from_file(empty_sentinel)

    paths_file = tmp_path / "paths.txt"
    paths_file.write_text(" ./lib \n# ignored\n../share\n", encoding="utf-8")
    assert pb._load_additional_paths_from_file(paths_file) == ("./lib", "../share")

    empty_paths = tmp_path / "paths_empty.txt"
    empty_paths.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pb._load_additional_paths_from_file(empty_paths)


def test_parse_inline_config_variants():
    parsed = pb._parse_inline_config('{"sentinels": ["pyproject.toml"]}', source="CLI")
    assert parsed["sentinels"] == ("pyproject.toml",)

    parsed_toml = pb._parse_inline_config("sentinels = ['repo.cfg']", source="ENV")
    assert parsed_toml["sentinels"] == ("repo.cfg",)

    with pytest.raises(TypeError):
        pb._parse_inline_config("[1, 2, 3]", source="CLI")

    with pytest.raises(ValueError):
        pb._parse_inline_config("   ", source="CLI")

    with pytest.raises(ValueError):
        pb._parse_inline_config("not valid", source="CLI")


def test_parse_profile_tokens_and_unique_entries():
    tokens = pb._parse_profile_tokens(
        [" base ", "!sunset", "-obsolete", "", "ci"],
        allow_remove=True,
        source="cli",
    )
    assert tokens == (
        pb.ProfileToken("base", "add"),
        pb.ProfileToken("sunset", "remove"),
        pb.ProfileToken("obsolete", "remove"),
        pb.ProfileToken("ci", "add"),
    )

    with pytest.raises(ValueError):
        pb._parse_profile_tokens(["-sunset"], allow_remove=False, source="config")

    assert pb._unique_entries(["a", "b", "a", "c"]) == ("a", "b", "c")


def test_format_path_value_and_sequence():
    win_style = "C:\\temp\\repo"
    assert pb._format_path_value("/tmp/repo", "auto") == "/tmp/repo"
    assert pb._format_path_value(win_style, "posix") == "C:/temp/repo"
    assert pb._format_path_value("C:/repo", "windows") == "C:\\repo"
    with pytest.raises(ValueError):
        pb._format_path_value("/tmp", "invalid")
    assert pb._format_path_sequence(["a", "b"], "auto") == ("a", "b")


def test_build_shell_command_variants():
    pwsh = pb._build_shell_command("/usr/bin/pwsh", ["echo", "hi there"])
    assert pwsh[:2] == ("/usr/bin/pwsh", "-Command")
    assert "hi there" in pwsh[2]

    posix = pb._build_shell_command("/bin/sh", ["echo", "hi"])
    assert posix == ("/bin/sh", "-c", "echo hi")

    no_command = pb._build_shell_command("/bin/sh", [])
    assert no_command == ("/bin/sh",)


def test_get_repo_info_and_clear_cache(tmp_path):
    sentinel = tmp_path / "pyproject.toml"
    sentinel.write_text("[build-system]", encoding="utf-8")
    info = pb.get_repo_info(root_hint=tmp_path, sentinels=(sentinel.name,))
    assert info.root == tmp_path.resolve()
    pb.clear_cache()
    info_again = pb.get_repo_info(root_hint=tmp_path, sentinels=(sentinel.name,))
    assert info_again.root == info.root
    with pytest.raises(ValueError):
        pb.get_repo_info(root_hint=tmp_path, sentinels=(sentinel.name,), max_depth=-1)


def test_chdir_repo_root_context(tmp_path):
    sentinel = tmp_path / "pyproject.toml"
    sentinel.write_text("[project]", encoding="utf-8")
    cwd_before = Path.cwd()
    try:
        with pb.chdir_repo_root(root_hint=tmp_path, sentinels=(sentinel.name,)) as repo_root:
            assert repo_root == tmp_path.resolve()
            assert Path.cwd() == repo_root
        assert Path.cwd() == cwd_before
    finally:
        os.chdir(cwd_before)

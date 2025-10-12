import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.telemetry_risk_profiles as telemetry_profiles_cli

from bot_core.runtime.telemetry_risk_profiles import (
    get_metrics_service_env_overrides,
    get_metrics_service_overrides,
    get_risk_profile,
    load_risk_profiles_from_file,
    load_risk_profiles_with_metadata,
    register_risk_profiles,
    risk_profile_metadata,
    summarize_risk_profile,
)


def test_load_risk_profiles_from_directory(tmp_path: Path) -> None:
    directory = tmp_path / "profiles"
    directory.mkdir()

    (directory / "alpha.json").write_text(
        json.dumps({"risk_profiles": {"alpha_dir": {"severity_min": "warning"}}}),
        encoding="utf-8",
    )
    (directory / "beta.yaml").write_text(
        "risk_profiles:\n  beta_dir:\n    severity_min: info\n",
        encoding="utf-8",
    )

    registered = load_risk_profiles_from_file(directory)
    assert set(registered) >= {"alpha_dir", "beta_dir"}

    alpha_meta = risk_profile_metadata("alpha_dir")
    assert alpha_meta["origin"].startswith("dir:")
    assert alpha_meta["origin"].endswith("#alpha.json")

    beta_profile = get_risk_profile("beta_dir")
    assert beta_profile["severity_min"] == "info"

    registered_with_meta, metadata = load_risk_profiles_with_metadata(directory)
    assert set(registered_with_meta) >= {"alpha_dir", "beta_dir"}
    assert metadata["type"] == "directory"
    assert metadata["path"] == str(directory)
    assert {entry["path"] for entry in metadata["files"]} == {
        str(directory / "alpha.json"),
        str(directory / "beta.yaml"),
    }


def test_load_risk_profiles_directory_empty(tmp_path: Path) -> None:
    directory = tmp_path / "empty_profiles"
    directory.mkdir()

    with pytest.raises(ValueError):
        load_risk_profiles_from_file(directory)

    with pytest.raises(ValueError):
        load_risk_profiles_with_metadata(directory)


def test_load_risk_profiles_directory_with_origin(tmp_path: Path) -> None:
    directory = tmp_path / "profiles"
    directory.mkdir()

    (directory / "ops.json").write_text(
        json.dumps({"risk_profiles": {"ops_dir": {"severity_min": "error"}}}),
        encoding="utf-8",
    )

    _, metadata = load_risk_profiles_with_metadata(directory, origin_label="cli:profiles")
    assert metadata["origin"] == "cli:profiles"
    assert metadata["type"] == "directory"
    assert metadata["files"][0]["registered_profiles"] == ["ops_dir"]


def test_summarize_risk_profile_provides_limits() -> None:
    metadata = risk_profile_metadata("conservative")
    summary = summarize_risk_profile(metadata)

    assert summary["name"] == "conservative"
    assert summary["severity_min"] == "warning"
    assert summary["max_event_counts"]["overlay_budget"] == 0
    assert "extends_chain" in summary


def test_register_profile_extends_builtin(tmp_path: Path) -> None:
    profile_path = tmp_path / "ops.json"
    profile_path.write_text(
        json.dumps(
            {
                "risk_profiles": {
                    "ops": {
                        "extends": "balanced",
                        "max_event_counts": {"reduce_motion": 7},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    registered = load_risk_profiles_from_file(profile_path)
    assert registered == ["ops"]

    profile = get_risk_profile("ops")
    assert profile["extends"] == "balanced"
    assert profile["max_event_counts"]["overlay_budget"] == 2  # z profilu bazowego
    assert profile["max_event_counts"]["reduce_motion"] == 7  # nadpisanie
    overrides = get_metrics_service_overrides("ops")
    assert overrides["ui_alerts_overlay_critical_threshold"] == 2


def test_register_profile_extends_chain(tmp_path: Path) -> None:
    profiles = {
        "ops-base": {
            "extends": "balanced",
            "metrics_service_overrides": {"ui_alerts_jank_mode": "jsonl"},
        },
        "ops-final": {
            "extends": "ops-base",
            "severity_min": "critical",
        },
    }

    registered = register_risk_profiles(profiles, origin="tests")
    assert registered == ["ops-base", "ops-final"]

    profile = get_risk_profile("ops-final")
    assert profile["extends"] == "ops-base"
    assert profile["extends_chain"][-1] == "ops-base"
    assert "balanced" in profile["extends_chain"]
    assert profile["metrics_service_overrides"]["ui_alerts_jank_mode"] == "jsonl"


def test_metrics_service_env_overrides() -> None:
    env_overrides = get_metrics_service_env_overrides("conservative")
    assert (
        env_overrides["RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CRITICAL_THRESHOLD"]
        == 1
    )
    assert (
        env_overrides["RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_ACTIVE_SEVERITY"]
        == "critical"
    )


def test_register_profile_extends_unknown() -> None:
    with pytest.raises(ValueError):
        register_risk_profiles({"ops": {"extends": "nope"}}, origin="tests")


def test_register_profile_extends_cycle() -> None:
    with pytest.raises(ValueError):
        register_risk_profiles(
            {"alpha": {"extends": "beta"}, "beta": {"extends": "alpha"}},
            origin="tests",
        )


def test_cli_list_profiles(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    profile_path = tmp_path / "cli_profiles.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-list:\n    severity_min: critical\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        ["list", "--risk-profiles-file", str(profile_path), "--verbose"]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["sources"][0]["path"] == str(profile_path)
    assert payload["profiles"]["cli-list"]["severity_min"] == "critical"


def test_cli_list_profiles_yaml_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "cli_profiles_yaml.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-list-yaml:\n    severity_min: info\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "list",
            "--risk-profiles-file",
            str(profile_path),
            "--verbose",
            "--format",
            "yaml",
        ]
    )
    assert exit_code == 0

    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["profiles"]["cli-list-yaml"]["severity_min"] == "info"


def test_cli_bundle_generates_templates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "bundle"
    exit_code = telemetry_profiles_cli.main(
        [
            "bundle",
            "--output-dir",
            str(output_dir),
            "--stage",
            "demo=conservative",
            "--stage",
            "paper=balanced",
            "--config-format",
            "json",
        ]
    )
    assert exit_code == 0

    manifest_stdout = json.loads(capsys.readouterr().out)
    assert manifest_stdout["manifest_path"].endswith("manifest.json")

    manifest_path = output_dir / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = {entry["stage"] for entry in manifest_payload["stages"]}
    assert stages == {"demo", "paper", "live"}

    demo_env = (output_dir / "demo" / "metrics.env").read_text(encoding="utf-8")
    assert "risk_profile_summary" in demo_env

    demo_config = json.loads((output_dir / "demo" / "metrics.json").read_text(encoding="utf-8"))
    assert demo_config["risk_profile"] == "conservative"
    assert (
        demo_config["metrics_service"]["env_overrides"][
            "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CRITICAL_THRESHOLD"
        ]
        == 1
    )


def test_cli_bundle_defaults(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle-default"
    exit_code = telemetry_profiles_cli.main([
        "bundle",
        "--output-dir",
        str(output_dir),
    ])
    assert exit_code == 0

    manifest_payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["stage"] for entry in manifest_payload["stages"]] == [
        "demo",
        "paper",
        "live",
    ]


def test_cli_bundle_rejects_invalid_stage(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle-invalid"
    with pytest.raises(SystemExit) as excinfo:
        telemetry_profiles_cli.main([
            "bundle",
            "--output-dir",
            str(output_dir),
            "--stage",
            "invalid",
        ])
    assert excinfo.value.code == 2


def test_cli_list_profiles_infers_yaml_from_output(tmp_path: Path) -> None:
    profile_path = tmp_path / "cli_profiles_infer.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-list-infer:\n    severity_min: warning\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "profiles_dump.yaml"
    exit_code = telemetry_profiles_cli.main(
        [
            "list",
            "--risk-profiles-file",
            str(profile_path),
            "--verbose",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0

    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["profiles"]["cli-list-infer"]["severity_min"] == "warning"


def test_cli_list_profiles_format_conflict(tmp_path: Path) -> None:
    profile_path = tmp_path / "cli_profiles_conflict.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-list-conflict:\n    severity_min: warning\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        telemetry_profiles_cli.main(
            [
                "list",
                "--risk-profiles-file",
                str(profile_path),
                "--format",
                "json",
                "--output",
                str(tmp_path / "profiles.yaml"),
            ]
        )

    assert excinfo.value.code != 0


def test_cli_show_profile(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    profile_path = tmp_path / "cli_show.json"
    profile_path.write_text(
        json.dumps(
            {
                "risk_profiles": {
                    "cli-show": {
                        "severity_min": "error",
                        "metrics_service_overrides": {
                            "ui_alerts_reduce_mode": "enable",
                            "ui_alerts_overlay_mode": "enable",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        ["show", "cli-show", "--risk-profiles-file", str(profile_path)]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["name"] == "cli-show"
    assert payload["risk_profile"]["severity_min"] == "error"
    assert payload["metrics_service_overrides"]["ui_alerts_reduce_mode"] == "enable"


def test_cli_show_profile_yaml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "cli_show_yaml.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-show-yaml:\n    severity_min: warning\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "show",
            "cli-show-yaml",
            "--risk-profiles-file",
            str(profile_path),
            "--format",
            "yaml",
        ]
    )
    assert exit_code == 0

    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["name"] == "cli-show-yaml"
    assert payload["risk_profile"]["severity_min"] == "warning"


def test_cli_render_profile_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "cli_render.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-render:\n    severity_min: warning\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "render",
            "cli-render",
            "--risk-profiles-file",
            str(profile_path),
            "--include-profile",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["name"] == "cli-render"
    assert payload["metrics_service_config_overrides"] == {}
    assert payload["metrics_service_env_overrides"] == {}
    assert payload["cli_flags"] == []
    assert payload["env_assignments"] == []
    assert payload["env_assignments_format"] == "dotenv"
    assert payload["risk_profile"]["severity_min"] == "warning"
    assert payload["sources"][0]["path"] == str(profile_path)


def test_cli_render_profile_cli_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "cli_render_cli.json"
    profile_path.write_text(
        json.dumps(
            {
                "risk_profiles": {
                    "cli-render-cli": {
                        "metrics_service_overrides": {
                            "ui_alerts_reduce_mode": "enable",
                            "ui_alerts_overlay_critical_threshold": 2,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "render",
            "cli-render-cli",
            "--risk-profiles-file",
            str(profile_path),
            "--format",
            "cli",
            "--cli-style",
            "space",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert "--ui-alerts-reduce-mode enable" in lines
    assert "--ui-alerts-overlay-critical-threshold 2" in lines


def test_cli_render_profile_env_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "cli_render_env.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-render-env:\n    metrics_service_overrides:\n"
        "      ui_alerts_jank_mode: enable\n"
        "      ui_alerts_jank_critical_over_ms: 17\n"
        "      ui_alerts_audit_dir: /var/log/ui alerts\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "render",
            "cli-render-env",
            "--risk-profiles-file",
            str(profile_path),
            "--format",
            "env",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert "RUN_METRICS_SERVICE_UI_ALERTS_JANK_MODE=enable" in lines
    assert "RUN_METRICS_SERVICE_UI_ALERTS_JANK_CRITICAL_OVER_MS=17" in lines
    assert (
        'RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_DIR="/var/log/ui alerts"' in lines
    )


def test_cli_render_profile_env_export_style(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "cli_render_env_export.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-render-env-export:\n    metrics_service_overrides:\n"
        "      ui_alerts_audit_dir: /var/log/ui alerts\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "render",
            "cli-render-env-export",
            "--risk-profiles-file",
            str(profile_path),
            "--format",
            "env",
            "--env-style",
            "export",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert (
        "export RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_DIR='/var/log/ui alerts'" in lines
    )


def test_cli_render_profile_env_with_include_profile_rejected(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "cli_render_env_include.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-render-env-include:\n    metrics_service_overrides:\n"
        "      ui_alerts_jank_mode: enable\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        telemetry_profiles_cli.main(
            [
                "render",
                "cli-render-env-include",
                "--risk-profiles-file",
                str(profile_path),
                "--format",
                "env",
                "--include-profile",
            ]
        )

    assert excinfo.value.code == 2


def test_cli_render_profile_yaml_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "cli_render_yaml.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-render-yaml:\n    severity_min: notice\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "render",
            "cli-render-yaml",
            "--risk-profiles-file",
            str(profile_path),
            "--format",
            "yaml",
            "--include-profile",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = yaml.safe_load(captured.out)
    assert payload["name"] == "cli-render-yaml"
    assert payload["risk_profile"]["severity_min"] == "notice"
    assert payload["metrics_service_overrides"] == {}
    assert payload["env_assignments_format"] == "dotenv"
    assert payload["sources"][0]["path"] == str(profile_path)


def test_cli_render_infers_format_from_output_extension(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "auto.yaml"

    exit_code = telemetry_profiles_cli.main(
        ["render", "balanced", "--output", str(out_path)]
    )
    assert exit_code == 0

    rendered = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert rendered["name"] == "balanced"
    assert rendered["metrics_service_overrides"]


def test_cli_render_profile_json_sections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "cli_render_sections.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-render-sections:\n    metrics_service_overrides:\n"
        "      ui_alerts_reduce_mode: enable\n"
        "      ui_alerts_overlay_critical_threshold: 1\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "render",
            "cli-render-sections",
            "--risk-profiles-file",
            str(profile_path),
            "--format",
            "json",
            "--section",
            "metrics_service_config_overrides",
            "--section",
            "env_assignments",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload.keys()) == {
        "name",
        "metrics_service_config_overrides",
        "env_assignments",
        "env_assignments_format",
    }
    assert (
        payload["metrics_service_config_overrides"]["reduce_motion_mode"]
        == "enable"
    )
    assert any(
        line == "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_MODE=enable"
        for line in payload["env_assignments"]
    )


def test_cli_render_profile_json_sections_include_risk_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "cli_render_sections_profile.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-render-sections-profile:\n    severity_min: warning\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "render",
            "cli-render-sections-profile",
            "--risk-profiles-file",
            str(profile_path),
            "--format",
            "json",
            "--section",
            "risk_profile",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload.keys()) == {"name", "risk_profile"}
    assert payload["risk_profile"]["severity_min"] == "warning"


def test_cli_render_profile_json_sections_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "cli_render_sections_summary.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-render-sections-summary:\n    severity_min: warning\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "render",
            "cli-render-sections-summary",
            "--risk-profiles-file",
            str(profile_path),
            "--format",
            "json",
            "--section",
            "summary",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload.keys()) == {"name", "summary"}
    assert payload["summary"]["severity_min"] == "warning"
    assert "extends_chain" in payload["summary"]


def test_cli_render_profile_sections_not_supported_for_cli_format(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "cli_render_sections_cli.yaml"
    profile_path.write_text(
        "risk_profiles:\n  cli-render-sections-cli:\n    metrics_service_overrides:\n"
        "      ui_alerts_reduce_mode: enable\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        telemetry_profiles_cli.main(
            [
                "render",
                "cli-render-sections-cli",
                "--risk-profiles-file",
                str(profile_path),
                "--format",
                "cli",
                "--section",
                "metrics_service_overrides",
            ]
        )

    assert excinfo.value.code == 2


def test_cli_diff_builtin_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = telemetry_profiles_cli.main(["diff", "balanced", "conservative"])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["base"] == "balanced"
    assert payload["target"] == "conservative"

    overrides_diff = payload["diff"]["metrics_service_overrides"]
    change = overrides_diff["changed"]["ui_alerts_overlay_critical_threshold"]
    assert change["from"] == 2
    assert change["to"] == 1

    assert payload["diff"]["severity_min"]["from"] == "notice"
    assert payload["diff"]["severity_min"]["to"] == "warning"
    assert payload["diff"]["extends"] == {"unchanged": None}
    assert payload["diff"]["extends_chain"] == {"unchanged": None}

    cli_changes = payload["cli"]["added_or_changed"]
    assert any("--ui-alerts-overlay-critical-threshold=1" in line for line in cli_changes)

    env_changes = payload["env"]["added_or_changed"]
    assert (
        "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CRITICAL_THRESHOLD=1" in env_changes
    )
    assert payload["env"]["format"] == "dotenv"


def test_cli_diff_hide_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = telemetry_profiles_cli.main(
        ["diff", "balanced", "conservative", "--hide-unchanged"]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert "unchanged" not in payload["diff"]["metrics_service_overrides"]
    assert "unchanged" not in payload["diff"]["metrics_service_config_overrides"]
    assert "unchanged" not in payload["diff"]["metrics_service_env_overrides"]
    assert payload["diff"]["expect_summary_enabled"] == {}
    assert payload["diff"]["extends"] == {}
    assert payload["diff"]["extends_chain"] == {}


def test_cli_diff_yaml_output(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = telemetry_profiles_cli.main(
        ["diff", "balanced", "conservative", "--format", "yaml"]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = yaml.safe_load(captured.out)
    assert payload["base"] == "balanced"
    assert payload["target"] == "conservative"
    severity_section = payload["diff"]["severity_min"]
    assert severity_section["from"] == "notice"
    assert severity_section["to"] == "warning"
    assert "metrics_service_overrides" in payload["diff"]


def test_cli_diff_section_filtering(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = telemetry_profiles_cli.main(
        [
            "diff",
            "balanced",
            "conservative",
            "--section",
            "diff",
            "--section",
            "cli",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)

    assert set(payload.keys()) == {"base", "target", "diff", "cli"}
    assert "env" not in payload
    assert "summary" not in payload


def test_cli_diff_sections_auto_include_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = telemetry_profiles_cli.main(
        [
            "diff",
            "balanced",
            "conservative",
            "--section",
            "profiles",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)

    assert "profiles" in payload
    assert payload["profiles"]["base"]["name"] == "balanced"


def test_cli_diff_fail_on_diff_sets_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = telemetry_profiles_cli.main(
        ["diff", "balanced", "conservative", "--fail-on-diff"]
    )
    assert exit_code == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["base"] == "balanced"
    assert payload["target"] == "conservative"


def test_cli_diff_fail_on_diff_passes_when_no_changes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = telemetry_profiles_cli.main(
        ["diff", "balanced", "balanced", "--fail-on-diff"]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    overrides_diff = payload["diff"]["metrics_service_overrides"]
    assert overrides_diff["added"] == {}
    assert overrides_diff["removed"] == []
    assert overrides_diff["changed"] == {}


def test_cli_diff_infers_yaml_from_output_extension(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "diff.yaml"

    exit_code = telemetry_profiles_cli.main(
        ["diff", "balanced", "conservative", "--output", str(out_path)]
    )
    assert exit_code == 0

    payload = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert payload["base"] == "balanced"
    assert payload["target"] == "conservative"


def test_cli_diff_detects_format_extension_mismatch(tmp_path: Path) -> None:
    out_path = tmp_path / "diff.yaml"

    with pytest.raises(SystemExit) as excinfo:
        telemetry_profiles_cli.main(
            [
                "diff",
                "balanced",
                "conservative",
                "--format",
                "json",
                "--output",
                str(out_path),
            ]
        )

    assert excinfo.value.code == 2


def test_cli_render_detects_format_extension_mismatch(tmp_path: Path) -> None:
    out_path = tmp_path / "render.yaml"

    with pytest.raises(SystemExit) as excinfo:
        telemetry_profiles_cli.main(
            [
                "render",
                "balanced",
                "--format",
                "json",
                "--output",
                str(out_path),
            ]
        )

    assert excinfo.value.code == 2


def test_cli_diff_with_custom_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "diff_profiles.yaml"
    profile_path.write_text(
        "risk_profiles:\n"
        "  balanced-ops:\n"
        "    extends: balanced\n"
        "    metrics_service_overrides:\n"
        "      ui_alerts_overlay_critical_threshold: 4\n"
        "      ui_alerts_jank_mode: jsonl\n",
        encoding="utf-8",
    )

    exit_code = telemetry_profiles_cli.main(
        [
            "diff",
            "balanced",
            "balanced-ops",
            "--risk-profiles-file",
            str(profile_path),
            "--include-profiles",
            "--cli-style",
            "space",
            "--env-style",
            "export",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["sources"][0]["path"] == str(profile_path)
    assert payload["profiles"]["target"]["extends"] == "balanced"
    assert payload["summary"]["target"]["extends_chain"][-1] == "balanced"
    assert payload["diff"]["extends"]["from"] is None
    assert payload["diff"]["extends"]["to"] == "balanced"
    assert payload["diff"]["extends_chain"]["from"] is None
    assert payload["diff"]["extends_chain"]["to"][-1] == "balanced"

    cli_changes = payload["cli"]["added_or_changed"]
    assert "--ui-alerts-overlay-critical-threshold 4" in cli_changes
    assert "--ui-alerts-jank-mode jsonl" in cli_changes

    env_changes = payload["env"]["added_or_changed"]
    assert payload["env"]["format"] == "export"
    assert (
        "export RUN_METRICS_SERVICE_UI_ALERTS_JANK_MODE=jsonl" in env_changes
    )
    assert payload["diff"]["metrics_service_overrides"]["added"] == {}
    assert payload["diff"]["metrics_service_overrides"]["removed"] == []


def test_cli_validate_requires_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = telemetry_profiles_cli.main(
        ["validate", "--require", "balanced", "--require", "missing-profile"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert "missing-profile" in payload["missing"]


def test_cli_validate_yaml_output(tmp_path: Path) -> None:
    output_path = tmp_path / "validate.yaml"
    exit_code = telemetry_profiles_cli.main(
        [
            "validate",
            "--require",
            "balanced",
            "--format",
            "yaml",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0

    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["missing"] == []

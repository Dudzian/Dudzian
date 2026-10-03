import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment.windows_installer import provision
from deployment.windows_stage9_production_trust import CEREMONY_ID


def _base(tmp_path: Path) -> tuple[Path, Path, Path]:
    program_files = tmp_path / "Program Files" / "CryptoHunter"
    program_data = tmp_path / "ProgramData" / "CryptoHunter"
    source = tmp_path / "public-final-package"
    program_files.mkdir(parents=True)
    (program_data / "Config").mkdir(parents=True)
    source.mkdir()
    for number in range(12):
        (source / f"public-{number}.json").write_text("{}\n", encoding="utf-8")
    (program_data / provision.OWNERSHIP_RECORD).write_text(
        json.dumps(
            {
                "schema_version": provision.SCHEMA,
                "state": "COMMITTED",
                "program_files": str(program_files.resolve()),
                "program_data": str(program_data.resolve()),
                "service_names": list(provision.SERVICES),
            }
        ),
        encoding="utf-8",
    )
    return program_files, program_data, source


def _qualified(monkeypatch: pytest.MonkeyPatch, program_data: Path) -> list[Path]:
    import deployment.platforms.windows as windows
    import deployment.windows_stage9_production_trust as trust

    destination = program_data / "Config" / "ProductionTrust" / CEREMONY_ID
    loads: list[Path] = []
    monkeypatch.setattr(
        windows,
        "production_trust_package_path",
        lambda ceremony: destination,
    )
    monkeypatch.setattr(
        windows,
        "resolve_paths",
        lambda: SimpleNamespace(
            install=program_data.parents[1] / "Program Files" / "CryptoHunter",
            configuration=program_data / "Config",
        ),
    )
    monkeypatch.setattr(trust, "load_production_trust", lambda path: loads.append(path))
    monkeypatch.setattr(provision, "qualify_dacl", lambda *args: None)
    monkeypatch.setattr(provision, "qualify_production_trust_dacl", lambda path: None)

    def install(source: Path, target: Path) -> Path:
        target.parent.mkdir(parents=True)
        shutil.copytree(source, target)
        return target

    monkeypatch.setattr(trust, "install_public_production_trust", install)
    return loads


def _rewrite_ownership(
    program_data: Path, program_files: Path, owned_program_data: Path | None = None
) -> None:
    path = program_data / provision.OWNERSHIP_RECORD
    record = json.loads(path.read_text(encoding="utf-8"))
    record["program_files"] = str(program_files.resolve())
    record["program_data"] = str((owned_program_data or program_data).resolve())
    path.write_text(json.dumps(record), encoding="utf-8")


def test_command_requires_committed_base_without_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    program_files = tmp_path / "missing-files"
    program_data = tmp_path / "missing-data"
    with pytest.raises(provision.ProvisionError, match="base installation"):
        provision.install_production_trust(program_files, program_data, source)
    assert tuple(tmp_path.iterdir()) == (source,)


@pytest.mark.parametrize("field", ("state", "program_files", "program_data", "service_names"))
def test_command_rejects_invalid_ownership_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    program_files, program_data, source = _base(tmp_path)
    record_path = program_data / provision.OWNERSHIP_RECORD
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record[field] = "INVALID"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    _qualified(monkeypatch, program_data)
    with pytest.raises(provision.ProvisionError):
        provision.install_production_trust(program_files, program_data, source)
    assert not (program_data / "Config" / "ProductionTrust").exists()


def test_command_installs_exact_canonical_public_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program_files, program_data, source = _base(tmp_path)
    loads = _qualified(monkeypatch, program_data)
    destination = provision.install_production_trust(program_files, program_data, source)
    assert destination == program_data / "Config" / "ProductionTrust" / CEREMONY_ID
    assert len(tuple(destination.iterdir())) == 12
    assert provision._package_hashes(destination) == provision._package_hashes(source)
    assert loads == [source, destination]


def test_caller_controlled_fake_program_files_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, program_data, source = _base(tmp_path)
    fake_program_files = tmp_path / "attacker" / "CryptoHunter"
    fake_program_files.mkdir(parents=True)
    _rewrite_ownership(program_data, fake_program_files)
    _qualified(monkeypatch, program_data)
    with pytest.raises(provision.ProvisionError, match="ProgramFiles.*native canonical"):
        provision.install_production_trust(fake_program_files, program_data, source)
    assert not (program_data / "Config" / "ProductionTrust").exists()


def test_caller_controlled_fake_program_data_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program_files, canonical_data, source = _base(tmp_path)
    fake_data = tmp_path / "attacker-data" / "CryptoHunter"
    shutil.copytree(canonical_data, fake_data)
    _rewrite_ownership(fake_data, program_files, fake_data)
    _qualified(monkeypatch, canonical_data)
    with pytest.raises(provision.ProvisionError, match="ProgramData.*native canonical"):
        provision.install_production_trust(program_files, fake_data, source)
    assert not (fake_data / "Config" / "ProductionTrust").exists()


def test_both_caller_controlled_roots_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, canonical_data, source = _base(tmp_path)
    fake_files = tmp_path / "fake-files" / "CryptoHunter"
    fake_data = tmp_path / "fake-data" / "CryptoHunter"
    fake_files.mkdir(parents=True)
    shutil.copytree(canonical_data, fake_data)
    _rewrite_ownership(fake_data, fake_files, fake_data)
    _qualified(monkeypatch, canonical_data)
    with pytest.raises(provision.ProvisionError, match="ProgramFiles.*native canonical"):
        provision.install_production_trust(fake_files, fake_data, source)
    assert not (fake_data / "Config" / "ProductionTrust").exists()


def test_command_refuses_overwrite_before_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program_files, program_data, source = _base(tmp_path)
    _qualified(monkeypatch, program_data)
    destination = program_data / "Config" / "ProductionTrust" / CEREMONY_ID
    destination.mkdir(parents=True)
    with pytest.raises(provision.ProvisionError, match="overwrite"):
        provision.install_production_trust(program_files, program_data, source)


def test_preinstall_dacl_mismatch_fails_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program_files, program_data, source = _base(tmp_path)
    _qualified(monkeypatch, program_data)
    monkeypatch.setattr(
        provision,
        "qualify_dacl",
        lambda *args: (_ for _ in ()).throw(provision.ProvisionError("DACL differs")),
    )
    with pytest.raises(provision.ProvisionError, match="DACL"):
        provision.install_production_trust(program_files, program_data, source)
    assert not (program_data / "Config" / "ProductionTrust").exists()


def test_invalid_verified_source_fails_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deployment.windows_stage9_production_trust as trust

    program_files, program_data, source = _base(tmp_path)
    _qualified(monkeypatch, program_data)
    monkeypatch.setattr(
        trust,
        "load_production_trust",
        lambda path: (_ for _ in ()).throw(trust.ProductionTrustUnavailable("invalid")),
    )
    with pytest.raises(trust.ProductionTrustUnavailable, match="invalid"):
        provision.install_production_trust(program_files, program_data, source)
    assert not (program_data / "Config" / "ProductionTrust").exists()


def test_post_publish_failure_removes_published_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program_files, program_data, source = _base(tmp_path)
    _qualified(monkeypatch, program_data)
    destination = program_data / "Config" / "ProductionTrust" / CEREMONY_ID
    monkeypatch.setattr(
        provision,
        "qualify_production_trust_dacl",
        lambda path: (_ for _ in ()).throw(provision.ProvisionError("DACL differs")),
    )
    with pytest.raises(provision.ProvisionError, match="DACL"):
        provision.install_production_trust(program_files, program_data, source)
    assert not destination.exists()


def test_source_reparse_is_rejected_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program_files, program_data, source = _base(tmp_path)
    _qualified(monkeypatch, program_data)
    (source / "public-0.json").unlink()
    (source / "public-0.json").symlink_to(source / "public-1.json")
    with pytest.raises(provision.ProvisionError, match="reparse"):
        provision.install_production_trust(program_files, program_data, source)
    assert not (program_data / "Config" / "ProductionTrust").exists()


def test_cli_exposes_separate_source_only_command() -> None:
    source = Path(provision.__file__).read_text(encoding="utf-8")
    assert '"install-production-trust"' in source
    assert 'parser.add_argument("--source", type=Path)' in source
    assert "install_production_trust(args.program_files, args.program_data, args.source)" in source
    function = source[
        source.index("def install_production_trust(") : source.index("def qualify_dacl(")
    ]
    assert "enroll(" not in function


def test_lifecycle_status_remains_pre_enrollment_and_stage10_not_started() -> None:
    status = json.loads(Path("deployment/stage9_current_status.json").read_text(encoding="utf-8"))
    assert status["production_provisioning_ready"] is False
    assert status["windows_production_ready"] == "NOT_READY"
    assert status["stage_9"] == "IN_PROGRESS"
    assert status["stage_10_production_lifecycle_live"] == "NOT_STARTED"
    assert status["stage_10_prerequisite"] == "BLOCKED_UNTIL_LEGAL_ENROLLMENT"

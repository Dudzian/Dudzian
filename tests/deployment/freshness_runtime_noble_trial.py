"""Rzeczywista próba tylko w jednorazowym kontenerze Ubuntu 24.04.

Skrypt tworzy wyłącznie testowy klucz Ed25519, publikuje wydanie pod /opt,
przerywa drugi publish i sprawdza odmowy podmian. Nie tworzy kont ani klastra,
nie uruchamia systemd. Nigdy nie uruchamiać na fizycznym hoście ani Dellu.
Wymaga CRYPTOHUNTER_DISPOSABLE_NOBLE_TRIAL=1 oraz znacznika Docker.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--wheelhouse", type=Path, required=True)
parser.add_argument("--revision", required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--base-image", required=True)
args = parser.parse_args()
if (
    sys.platform != "linux"
    or os.geteuid() != 0
    or os.environ.get("CRYPTOHUNTER_DISPOSABLE_NOBLE_TRIAL") != "1"
    or not Path("/.dockerenv").is_file()
    or platform.freedesktop_os_release().get("ID") != "ubuntu"
    or platform.freedesktop_os_release().get("VERSION_ID") != "24.04"
):
    raise SystemExit("Odmowa: próba wymaga jawnie wybranego jednorazowego Docker Noble")
SOURCE = Path(__file__).resolve().parents[2]
TRIAL = Path("/opt/cryptohunter-runtime-trial")
if TRIAL.exists() or Path("/opt/cryptohunter/freshness-runtime").exists() or args.report.exists():
    raise SystemExit("Odmowa: próba wymaga pustego kontenera, bez istniejących wydań")
TRIAL.mkdir(mode=0o755)
CEREMONY = TRIAL / "ceremony"
CEREMONY.mkdir(mode=0o700)
subprocess.run(
    [
        str(Path(sys.executable)),
        "-I",
        "-s",
        "-B",
        str(SOURCE / "deployment/build_freshness_runtime.py"),
        "--wheelhouse",
        str(args.wheelhouse),
        "--destination",
        str(TRIAL / "candidate"),
        "--environment-id",
        "noble-cp312-v1",
        "--release-revision",
        args.revision,
        "--manifest",
        str(CEREMONY / "unsigned.json"),
    ],
    check=True,
)
subprocess.run(
    [
        "/usr/bin/openssl",
        "genpkey",
        "-algorithm",
        "Ed25519",
        "-out",
        str(CEREMONY / "test-only.key"),
    ],
    check=True,
)
public = subprocess.run(
    [
        "/usr/bin/openssl",
        "pkey",
        "-in",
        str(CEREMONY / "test-only.key"),
        "-pubout",
        "-outform",
        "DER",
    ],
    capture_output=True,
    check=True,
).stdout
assert len(public) == 44
(CEREMONY / "authority.pub").write_bytes(public[-32:])
subprocess.run(
    [
        "/usr/bin/openssl",
        "pkeyutl",
        "-sign",
        "-rawin",
        "-inkey",
        str(CEREMONY / "test-only.key"),
        "-in",
        str(CEREMONY / "unsigned.json"),
        "-out",
        str(CEREMONY / "runtime.sig"),
    ],
    check=True,
)

ns = runpy.run_path(str(SOURCE / "deployment/freshness_runtime.py"))
Config, verify, publish = ns["RuntimeConfig"], ns["verify_runtime"], ns["publish_runtime"]
base = Path("/opt/cryptohunter/freshness-runtime")
base.mkdir(parents=True)
previous = base / "releases/previous-release"
previous.mkdir(parents=True)
(previous / "preserve").write_bytes(b"previous release")
revision = args.revision
config = Config(
    runtime=(TRIAL / "candidate"),
    manifest=(CEREMONY / "unsigned.json"),
    signature=(CEREMONY / "runtime.sig"),
    trusted_key=(CEREMONY / "authority.pub"),
    expected_environment_id="noble-cp312-v1",
    expected_revision=revision,
)
checks = {}
verified = verify(config)
checks["verify_real_signed_runtime"] = True
assert not any(str(TRIAL) in name for name in verified.manifest["external_files"])
checks["no_staging_path_in_external_inventory"] = True
installed = publish(config)
checks["atomic_publish"] = True
assert publish(config).manifest_sha256 == installed.manifest_sha256
checks["identical_reinstall"] = True
assert (previous / "preserve").read_bytes() == b"previous release"
checks["previous_release_preserved"] = True
runtime = installed.executable.parents[1]
result = subprocess.run(
    [
        str(installed.executable),
        "-I",
        "-s",
        "-B",
        "-c",
        runpy.run_path(str(SOURCE / "deployment/build_freshness_runtime.py"))["_PROBE"],
    ],
    capture_output=True,
    text=True,
    check=True,
)
checks["published_real_import_crypto_ed25519_pq"] = True
installed_config = replace(
    config, runtime=runtime, manifest=runtime / ns["MANIFEST"], signature=runtime / ns["SIGNATURE"]
)


def reject(label, mutate, restore):
    mutate()
    try:
        try:
            verify(installed_config)
        except ns["RuntimeTrustError"]:
            checks[label] = True
        else:
            raise AssertionError(label)
    finally:
        restore()


cfg = runtime / "pyvenv.cfg"
old = cfg.read_bytes()
reject(
    "reject_pyvenv_tamper",
    lambda: cfg.write_bytes(old + b"include-system-site-packages = true\n"),
    lambda: cfg.write_bytes(old),
)
foreign = runtime / "lib/python3.12/site-packages/foreign.pth"
reject("reject_foreign_pth", lambda: foreign.write_text("import foreign"), lambda: foreign.unlink())
extra = runtime / "lib/python3.12/site-packages/foreign-1.dist-info"
reject("reject_extra_package", lambda: extra.mkdir(), lambda: extra.rmdir())
reject("reject_wrong_permission", lambda: cfg.chmod(0o666), lambda: cfg.chmod(0o644))
exe = runtime / "bin/python3.12"
reject(
    "reject_interpreter_symlink_tamper",
    lambda: (exe.unlink(), exe.symlink_to("/usr/bin/false")),
    lambda: (exe.unlink(), exe.symlink_to("/usr/bin/python3.12")),
)
original_sig = config.signature.read_bytes()
config.signature.write_bytes(b"0" * 64)
try:
    try:
        verify(config)
    except ns["RuntimeTrustError"]:
        checks["reject_invalid_signature"] = True
    else:
        raise AssertionError("signature")
finally:
    config.signature.write_bytes(original_sig)
# Nowe ID symuluje przerwaną publikację: fail w kontrolowanej operacji zapisu,
# rzeczywisty journal i staging pozostają; retry musi odmówić.
doc = json.loads(config.manifest.read_text())
doc["environment_id"] = "noble-interrupted-v1"
doc["installation_path"] = str(base / "releases/noble-interrupted-v1")
manifest2, sig2 = (CEREMONY / "interrupted.json"), (CEREMONY / "interrupted.sig")
manifest2.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
subprocess.run(
    [
        "openssl",
        "pkeyutl",
        "-sign",
        "-rawin",
        "-inkey",
        str(CEREMONY / "test-only.key"),
        "-in",
        str(manifest2),
        "-out",
        str(sig2),
    ],
    check=True,
)
config2 = replace(
    config, manifest=manifest2, signature=sig2, expected_environment_id="noble-interrupted-v1"
)
globals_ = publish.__globals__
original_write = globals_["_write_new"]


def fail_write(path, content, mode):
    if str(path).endswith("/build-probe.json"):
        raise OSError("injected interruption")
    return original_write(path, content, mode)


globals_["_write_new"] = fail_write
try:
    try:
        publish(config2)
    except OSError:
        pass
    else:
        raise AssertionError("interruption")
finally:
    globals_["_write_new"] = original_write
assert (base / "noble-interrupted-v1.journal").exists()
try:
    publish(config2)
except ns["RuntimeTrustError"]:
    checks["interrupted_publication_refused"] = True
else:
    raise AssertionError("retry")
assert (previous / "preserve").exists()
# Bez modyfikacji bazowej binarki (decyzja o systemowych aktualizacjach nie
# należy do tej próby): test odmowy driftu stdlib z rzeczywistym plikiem.
stdlib = Path("/usr/lib/python3.12/stat.py")
original = stdlib.read_bytes()
reject(
    "reject_external_stdlib_changed_after_update",
    lambda: stdlib.write_bytes(original + b"\n# changed\n"),
    lambda: stdlib.write_bytes(original),
)
# Symulacja aktualizacji interpretera dotyczy wyłącznie rootfs tego
# jednorazowego kontenera. Nie zapisujemy aktywnego inode (ETXTBSY), nie
# uruchamiamy zmienionego programu i przywracamy oryginał atomowym rename.
base_executable = Path("/usr/bin/python3.12")
saved_executable = base_executable.with_name("python3.12.trial-original")
substituted_executable = TRIAL / "substituted-python3.12"
assert not saved_executable.exists()
shutil.copyfile(base_executable, substituted_executable)
with substituted_executable.open("ab") as output:
    output.write(b"\ncryptohunter-disposable-update-drift\n")
substituted_executable.chmod(0o755)


def substitute_base_executable():
    os.replace(base_executable, saved_executable)
    try:
        os.replace(substituted_executable, base_executable)
    except BaseException:
        os.replace(saved_executable, base_executable)
        raise


reject(
    "reject_system_interpreter_update",
    substitute_base_executable,
    lambda: os.replace(saved_executable, base_executable),
)
checks["final_installed_integrity"] = bool(verify(installed_config))
report = {
    "container_os": "Ubuntu 24.04 amd64",
    "operator_base_image": args.base_image,
    "network": "none",
    "python": verified.manifest["interpreter"],
    "packages": verified.packages,
    "runtime_files": len(verified.manifest["files"]),
    "external_files": len(verified.manifest["external_files"]),
    "probe": json.loads(result.stdout),
    "checks": checks,
}
args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(
    json.dumps(
        {
            "checks": checks,
            "runtime_files": report["runtime_files"],
            "external_files": report["external_files"],
        }
    )
)

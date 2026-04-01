from __future__ import annotations

from pathlib import Path

from scripts import package_update


def test_main_delegates_to_packaging_builder(monkeypatch, tmp_path: Path) -> None:
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    (payload_dir / "demo.txt").write_text("demo", encoding="utf-8")
    output_path = tmp_path / "bundle.dudzianpkg"

    captured: dict[str, object] = {}

    def fake_build_offline_package(**kwargs):
        captured.update(kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("archive", encoding="utf-8")
        return output

    monkeypatch.setattr(package_update, "build_offline_package", fake_build_offline_package)

    exit_code = package_update.main(
        [
            str(payload_dir),
            str(output_path),
            "--package-id",
            "demo",
            "--version",
            "1.0.0",
            "--fingerprint",
            "HW-123",
            "--metadata",
            "channel=stable",
            "build=ci",
            "--signing-key",
            "secret",
            "--signing-key-id",
            "k1",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert captured == {
        "package_id": "demo",
        "version": "1.0.0",
        "payload_dir": payload_dir,
        "output_path": output_path,
        "fingerprint": "HW-123",
        "metadata": {"channel": "stable", "build": "ci"},
        "signing_key": b"secret",
        "signing_key_id": "k1",
    }

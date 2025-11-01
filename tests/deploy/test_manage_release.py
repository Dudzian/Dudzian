from __future__ import annotations

from pathlib import Path

import pytest

from scripts import manage_release


def test_generate_checklist(tmp_path: Path) -> None:
    output = tmp_path / "checklist.md"
    result = manage_release.main(
        [
            "generate-checklist",
            "--version",
            "2.0.0",
            "--output",
            str(output),
            "--owner",
            "Zespół OEM",
            "--release-tag",
            "v2.0.0-rc1",
            "--license-report",
            "reports/oem/2.0.0/license.md",
            "--compliance-report",
            "reports/oem/2.0.0/compliance.md",
            "--test-report",
            "reports/oem/2.0.0/tests.md",
            "--notes",
            "Pakiet RC gotowy do publikacji.",
        ]
    )

    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "Checklista wydania OEM" in content
    assert "wersja 2.0.0" in content
    assert "Zespół OEM" in content
    assert "Pakiet RC gotowy" in content
    assert result.path == output
    assert result.template == "oem_checklist_template.md"
    assert "timestamp" in result.context


@pytest.mark.parametrize(
    "command, marker",
    [
        ("generate-license-report", "Raport licencyjny"),
        ("generate-compliance-report", "Raport zgodności"),
        ("generate-test-report", "Raport testów"),
    ],
)
def test_generate_reports(tmp_path: Path, command: str, marker: str) -> None:
    output = tmp_path / f"{command}.md"
    args = [command, "--version", "2.0.0", "--output", str(output)]
    if command == "generate-license-report":
        args.extend(["--summary", "Licencje aktywne."])
    elif command == "generate-compliance-report":
        args.extend(["--summary", "Brak naruszeń."])
    elif command == "generate-test-report":
        args.extend(["--environment", "Lab OEM", "--summary", "Smoke zakończone sukcesem."])

    manage_release.main(args)

    text = output.read_text(encoding="utf-8")
    assert marker in text
    assert "2.0.0" in text

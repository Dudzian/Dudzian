from pathlib import Path
import re


WORKFLOW_PATH = Path(".github/workflows/ci.yml")


def test_marketing_bundle_parity_uses_least_privilege_artifact_permissions() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    job = re.search(r"(?ms)^  marketing-bundle-parity:\n(?P<body>.*?)(?=^  \S|\Z)", text)
    assert job is not None
    permissions = re.search(
        r"(?m)^    permissions:\n(?P<body>(?:^      \S.*\n)+)", job.group("body")
    )
    assert permissions is not None

    assert permissions.group("body").splitlines() == [
        "      contents: read",
        "      actions: read",
        "      artifact-metadata: write",
    ]

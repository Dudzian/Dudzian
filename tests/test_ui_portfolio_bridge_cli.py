from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from scripts import ui_portfolio_bridge


@pytest.mark.parametrize("payload", [
    {
        "portfolio_id": "master-x",
        "primary_preset": "grid-pro",
        "fallback_presets": ["ml-ai"],
        "followers": [
            {"portfolio_id": "follower-1", "scaling": 0.5},
            {"portfolio_id": "follower-2", "scaling": 0.75},
        ],
    }
])
def test_bridge_apply_and_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], payload: dict[str, object]) -> None:
    store = tmp_path / "state.json"
    stdin = io.StringIO(json.dumps(payload))
    monkeypatch.setattr(sys, "stdin", stdin)
    ui_portfolio_bridge.main(["--store", str(store), "apply"])

    ui_portfolio_bridge.main(["--store", str(store), "list"])
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["portfolios"][0]["portfolio_id"] == payload["portfolio_id"]

    ui_portfolio_bridge.main(["--store", str(store), "remove", payload["portfolio_id"]])
    ui_portfolio_bridge.main(["--store", str(store), "list"])
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["portfolios"] == []

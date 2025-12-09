import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import scripts.publish_licensing_drift_metrics as metrics


def test_parse_grouping_labels_valid():
    labels = metrics.parse_grouping_labels(["region=eu", "env=prod"])
    assert labels == {"region": "eu", "env": "prod"}


def test_parse_grouping_labels_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        metrics.parse_grouping_labels(["invalid"])


def test_build_target_url_with_labels():
    url = metrics.build_target_url(
        "https://push.example.com/", "job-name", {"run_id": "123", "env": "prod"}
    )
    assert url == "https://push.example.com/metrics/job/job-name/run_id/123/env/prod"


def test_push_metrics_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        metrics.push_metrics(
            tmp_path / "missing.prom",
            "https://example.com/metrics/job/demo",
            headers={},
            timeout=1.0,
        )


def test_push_metrics_empty_file(tmp_path: Path):
    prom = tmp_path / "empty.prom"
    prom.write_text("\n")
    with pytest.raises(ValueError):
        metrics.push_metrics(
            prom,
            "https://example.com/metrics/job/demo",
            headers={},
            timeout=1.0,
        )


def test_push_metrics_sends_payload_and_headers(tmp_path: Path):
    prom = tmp_path / "metrics.prom"
    prom.write_text("metric_total 1\n")

    sent_requests = SimpleNamespace(headers=None, data=None, timeout=None)

    def fake_urlopen(request, timeout=None):  # pragma: no cover - exercised indirectly
        sent_requests.headers = dict(request.headers)
        sent_requests.data = request.data
        sent_requests.timeout = timeout

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"ok"

        return FakeResponse()

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        metrics.push_metrics(
            prom,
            "https://example.com/metrics/job/demo",
            headers=metrics.build_headers("user:pass"),
            timeout=2.5,
        )

    headers = {k.lower(): v for k, v in sent_requests.headers.items()}

    assert sent_requests.data == prom.read_bytes()
    assert sent_requests.timeout == 2.5
    assert headers["user-agent"].startswith("licensing-drift-metrics")
    assert headers["authorization"].startswith("Basic ")


def test_build_headers_invalid_auth():
    with pytest.raises(argparse.ArgumentTypeError):
        metrics.build_headers("useronly")

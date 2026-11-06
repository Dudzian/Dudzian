"""Powtarzalna seria MODEL_ONLY na istniejącej fixture; bez usług i VM.

Opcjonalny CRYPTOHUNTER_D07_MODEL_OUTPUT zapisuje nowy JSON z surowymi próbkami.
Nie stosujemy patchy remediacji, nie tworzymy połączenia PostgreSQL ani klienta.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import pytest

from bot_core import freshness_semantic_verifier as semantic
from bot_core.postgresql_freshness_authority import canonical_json_bytes
from deployment import stage9_vm_performance as performance
from deployment.stage9_vm_performance import Recorder
from tests.security.test_freshness_semantic_verifier import (
    Resolver,
    Writer,
    fixture_candidate,
)


def test_existing_verifier_model_series():
    """Mierz bazowy kod, prawdziwe Ed25519 i jawnie modelowane zależności."""
    recorder = Recorder()
    outcomes = {}
    corpus = {}
    for scenario, profiled in (
        ("valid", False),
        ("invalid_signature", False),
        ("malformed_json", False),
        ("valid_profiled", True),
    ):
        verifier, writer, proposer, document, receipt, *_ = fixture_candidate()
        if scenario == "invalid_signature":
            receipt = {**receipt, "authentication_tag_or_signature": "A" * 86}
        inputs = tuple(canonical_json_bytes(item) for item in (proposer, document, receipt))
        if scenario == "malformed_json":
            inputs = (b"{", *inputs[1:])
        corpus[scenario] = [
            {"bytes": len(item), "sha256": hashlib.sha256(item).hexdigest()} for item in inputs
        ]
        outcomes[scenario] = {"completed": 0, "expected_rejections": 0}
        for index in range(101):
            # Pierwszy request nowego obiektu nie oznacza zimnego procesu/cache.
            temperature = "unspecified" if index == 0 else "warm"
            coordinates = {"scenario": scenario, "temperature": temperature}
            writer.calls.clear()

            def exchange():
                with recorder.sample(
                    **coordinates, metric="semantic_verification", source="PROCESS"
                ):
                    return verifier.verify_exact_candidate(*inputs)

            if scenario in {"invalid_signature", "malformed_json"}:
                expected_error = (
                    json.JSONDecodeError
                    if scenario == "malformed_json"
                    else semantic.FreshnessSemanticVerificationError
                )
                with pytest.raises(expected_error):
                    exchange()
                assert not writer.calls
                outcomes[scenario]["expected_rejections"] += 1
            elif profiled:
                with recorder.profile(
                    {
                        semantic._verify.__code__: ("cryptography", "PROCESS"),
                        Resolver.resolve.__code__: ("model_dependency", "MODEL"),
                        Resolver.predecessor_complete_head_digest.__code__: (
                            "model_dependency",
                            "MODEL",
                        ),
                        Writer.prepare.__code__: ("model_dependency", "MODEL"),
                    },
                    **coordinates,
                ):
                    result = exchange()
                assert result.preparation_id == "opaque-preparation"
                assert len(writer.calls) == 1
                outcomes[scenario]["completed"] += 1
            else:
                result = exchange()
                assert result.preparation_id == "opaque-preparation"
                assert len(writer.calls) == 1
                outcomes[scenario]["completed"] += 1

    report = recorder.summary()
    assert report["classification"] == "MODEL_ONLY"
    assert not any(sample["source"] == "POSTGRESQL" for sample in report["samples"])
    report["outcomes"] = outcomes
    report["corpus"] = corpus
    report["environment"] = {
        "python": platform.python_version(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "logical_cpus_visible": os.cpu_count(),
        "cryptography": importlib.metadata.version("cryptography"),
        "psycopg": importlib.metadata.version("psycopg"),
        "base_commit": "5c8a600c3d4133362518513b80b6ac540d99dd19",
        "verifier_sha256": hashlib.sha256(Path(semantic.__file__).read_bytes()).hexdigest(),
        "observer_sha256": hashlib.sha256(Path(performance.__file__).read_bytes()).hexdigest(),
        "measurement_test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "postgresql": "NOT_RUN",
        "systemd_start": "NOT_RUN",
        "cold_process_and_caches": "NOT_RUN",
        "ram_and_cpu_allocation": "UNKNOWN_SHARED_EXECUTOR",
    }
    output = os.environ.get("CRYPTOHUNTER_D07_MODEL_OUTPUT")
    if output:
        with Path(output).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

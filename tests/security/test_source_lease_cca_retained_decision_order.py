"""S9-RPI-18: TEST-only source-read-lock -> durable CCA decision order.

The existing Linux worker, same-UID source floor and TEST CCA decision log
remain unqualified. This exercises a cooperative ordering protocol, not a
production source lease, atomic cross-authority transaction or CCA signing.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from bot_core import authenticated_custody_mutation_source as source
from bot_core import independent_cca_decision as cca
from bot_core import test_only_cca_decision_log as log_module
from bot_core.test_only_retained_source_original import TestOnlyRetainedSourceError
from tests.security._cca_decision_fixtures import build_cca_case, sign_cca_payload
from tests.security.test_retained_source_real_linux_disk_bridge import (
    _LinuxFixtureOriginalReader,
    _inspect,
    source_on_disk as source_on_disk,
)
from tests.security.test_retained_source_test_floor_witness import (
    _FloorGuardedReader,
    witnessed_source as witnessed_source,
)
from tests.security.test_source_custody_shared_lease_linux import (
    _competing_lock_probe,
    hold_synthetic_source_read_lease,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "linux", reason="requires Linux flock + fsync"),
]


def _cca_fixture_bound_to_retained_original(original):
    """Closed synthetic CCA claims with exact source bytes; never qualified.

    The existing CCA fixture creates a separate TEST private signing key.
    Its freshly invented source sample is replaced with the actual retained
    Linux worker envelope, so the CCA document's exact source bytes match
    the independently reread TEST original. Parsing only validates structure.
    """
    event = source.parse_custody_mutation_execution(original)
    descriptor = base64.b64decode(event.payload["originals"]["operation_descriptor_b64"])
    complete = base64.b64decode(event.payload["originals"]["complete_journal_b64"])
    case = build_cca_case(descriptor, complete)
    claimed = deepcopy(case.payload)
    claimed["source_event_envelope_b64"] = base64.b64encode(original).decode("ascii")
    claimed["source_event_envelope_sha256"] = hashlib.sha256(original).hexdigest()
    synthetic = SimpleNamespace(private_key=case.private_key, payload=claimed)
    envelope = sign_cca_payload(synthetic)
    decision = cca.parse_independent_cca_decision(envelope)
    assert base64.b64decode(decision.payload["source_event_envelope_b64"]) == original
    assert decision.payload["originals"] == event.payload["originals"]
    return synthetic, decision.immutable_binding_bytes


def _write_sample(synthetic, position):
    payload = deepcopy(synthetic.payload)
    payload["decision_id"] = position.entry_id
    payload["decision_log_position"] = position.document
    return sign_cca_payload(synthetic, payload)


@pytest.fixture
def source_with_cca_log(witnessed_source, tmp_path):
    config, original, floor_pin = witnessed_source
    synthetic, binding = _cca_fixture_bound_to_retained_original(original)
    source_payload = source.parse_custody_mutation_execution(original).payload
    log_path = tmp_path / "isolated-cca-decision-log"
    floor_path = tmp_path / "isolated-cca-decision-floor"
    for path in (log_path, floor_path):
        path.mkdir(mode=0o700)
    cca_pin = log_module.provision_test_only_cca_log(
        log_path,
        floor_path,
        installation_id=source_payload["installation"]["installation_id"],
        namespace=synthetic.payload["completion_authority"]["provider_namespace"],
    )
    return config, original, floor_pin, synthetic, binding, cca_pin


def _readers(config, original, floor_pin):
    disk = _LinuxFixtureOriginalReader(config, original)
    return disk, _FloorGuardedReader(disk, floor_pin)


def _decide(cca_pin, synthetic, binding, *, create=None, fault=None):
    return log_module.TestOnlyCCADecisionLog(cca_pin).decide(
        synthetic.payload["close"]["operation_id"],
        binding,
        create_decision=create or (lambda position: _write_sample(synthetic, position)),
        fault_injector=fault,
    )


def _lookup(cca_pin, synthetic, binding):
    return log_module.TestOnlyCCADecisionLog(cca_pin).exact_lookup(
        synthetic.payload["close"]["operation_id"], binding
    )


def test_source_lease_covers_durable_cca_decision_and_exact_retry(source_with_cca_log):
    config, original, floor_pin, synthetic, binding, cca_pin = source_with_cca_log
    disk, guarded = _readers(config, original, floor_pin)
    milestones = []

    def fault(phase):
        milestones.append(phase)

    with hold_synthetic_source_read_lease(config) as lease:
        assert lease.status == "NOT_QUALIFIED"
        assert lease.production_authorization == "BLOCKED"
        with _inspect(guarded, original) as evidence:
            assert evidence.status == "NOT_QUALIFIED"
            retained = _decide(cca_pin, synthetic, binding, fault=fault)
            parsed = cca.parse_independent_cca_decision(retained)
            assert base64.b64decode(parsed.payload["source_event_envelope_b64"]) == original
            assert retained == _lookup(cca_pin, synthetic, binding)
            assert "after_durable_decision" in milestones
            assert disk.lookups == 2
        assert disk.lookups == 3
    # Restart-side exact lookup and retry are reads of one original, never a re-sign.
    assert _lookup(cca_pin, synthetic, binding) == retained

    def forbidden(_position):
        pytest.fail("a retained CCA operation may never sign again")

    with hold_synthetic_source_read_lease(config):
        assert _decide(cca_pin, synthetic, binding, create=forbidden) == retained
    assert len(list(Path(cca_pin.log_directory).glob("*.decision"))) == 1
    assert log_module.TestOnlyCCADecisionLog.status == "NOT_QUALIFIED"
    assert log_module.TestOnlyCCADecisionLog.production_emission == "BLOCKED"


def test_competing_source_exclusive_lock_waits_through_durable_cca_floor(source_with_cca_log):
    config, original, floor_pin, synthetic, binding, cca_pin = source_with_cca_log
    disk, guarded = _readers(config, original, floor_pin)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    contender = context.Process(target=_competing_lock_probe, args=(sender, config["root"]))
    started = False
    try:
        with hold_synthetic_source_read_lease(config):
            with _inspect(guarded, original):
                contender.start()
                started = True
                sender.close()
                assert receiver.poll(15) and receiver.recv() == "attempting-exclusive"
                assert not receiver.poll(0.2), "source writer crossed held lease"

                def fault(phase):
                    if phase == "after_floor_reservation":
                        assert not receiver.poll(0.1), "source lock released before CCA intent"
                    if phase == "after_durable_decision":
                        assert not receiver.poll(0.1), "source lock released before CCA fsync"

                retained = _decide(cca_pin, synthetic, binding, fault=fault)
                assert _lookup(cca_pin, synthetic, binding) == retained
                assert not receiver.poll(0.1)
            assert disk.lookups == 3
            assert not receiver.poll(0.1), "source lock released before final source recheck"
        assert receiver.poll(15) and receiver.recv() == "acquired-exclusive"
        contender.join(10)
        assert contender.exitcode == 0
        assert len(list(Path(cca_pin.log_directory).glob("*.decision"))) == 1
    finally:
        receiver.close()
        sender.close()
        if started and contender.is_alive():
            contender.kill()
        if started:
            contender.join(5)
        contender.close()


def test_interrupted_cca_intent_is_indeterminate_without_resigning(source_with_cca_log):
    config, original, floor_pin, synthetic, binding, cca_pin = source_with_cca_log
    _disk, guarded = _readers(config, original, floor_pin)

    class SyntheticInterruption(RuntimeError):
        pass

    def fault(phase):
        if phase == "after_floor_reservation":
            raise SyntheticInterruption("TEST interrupted after durable floor reservation")

    with hold_synthetic_source_read_lease(config):
        with _inspect(guarded, original):
            with pytest.raises(SyntheticInterruption):
                _decide(cca_pin, synthetic, binding, fault=fault)
    with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
        _lookup(cca_pin, synthetic, binding)
    assert not list(Path(cca_pin.log_directory).glob("*.decision"))

    def forbidden(_position):
        pytest.fail("indeterminate intent must not be signed a second time")

    with hold_synthetic_source_read_lease(config):
        with pytest.raises(log_module.TestOnlyCCALogIndeterminate):
            _decide(cca_pin, synthetic, binding, create=forbidden)


def test_unqualified_source_refuses_before_cca_intent_or_decision(source_with_cca_log):
    config, original, floor_pin, synthetic, binding, cca_pin = source_with_cca_log
    source_path = Path(config["root"]) / "events" / "original"
    source_path.write_bytes(original[:-1])
    # The new reader pins already-mutated bytes, but the external test witness
    # was retained before this mutation and must still refuse the original.
    _disk, guarded = _readers(config, original, floor_pin)
    with hold_synthetic_source_read_lease(config):
        with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied"):
            with _inspect(guarded, original):
                pytest.fail("tampered original must not be accepted before CCA writes")
    assert not list(Path(cca_pin.log_directory).glob("*.intent"))
    assert not list(Path(cca_pin.log_directory).glob("*.decision"))

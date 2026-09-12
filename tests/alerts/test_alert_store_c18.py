"""S9D-C18 blocked-boundary and C17 carrier-shape hardening tests."""
from dataclasses import replace
from types import MappingProxyType

import pytest

from bot_core.alerts.store import AlertStore, AlertStoreError
from tests.alerts.test_alert_store import initial


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("evidence_ids", ("",)),
        ("evidence_ids", ("ev", "ev")),
        ("evidence_ids", (1,)),
        ("source_fence", (("source-a", 1),)),
        ("source_fence", (("", 1, 1),)),
        ("source_fence", (("source-a", True, 1),)),
        ("source_fence", (("source-a", 1, False),)),
        ("source_fence", (("source-a", 0, 1),)),
        ("source_fence", (("source-b", 1, 1), ("source-a", 1, 1))),
        ("result", 1),
        ("resolution_policy_id", ""),
    ],
)
def test_malformed_historical_source_nested_identity_fails_closed(
    tmp_path, field, malformed
):
    _, authority, sources, carrier, _, _ = initial(tmp_path)
    state = carrier._state
    reference, decision = next(iter(state.committed_historical_source_decisions.items()))
    corrupted = replace(decision, **{field: malformed})
    carrier._state = replace(
        state,
        committed_historical_source_decisions=MappingProxyType({reference: corrupted}),
    )

    with pytest.raises(AlertStoreError, match="SOURCE_EVIDENCE_UNACCEPTED"):
        AlertStore.restore(authority, sources, carrier)

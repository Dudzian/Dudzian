from __future__ import annotations

from pathlib import Path

from ui.backend.decision_source_selector import DecisionSourceFallbackCoordinator


def test_decision_source_selector_transitions_and_active_path() -> None:
    coordinator = DecisionSourceFallbackCoordinator()

    assert coordinator.active_decision_log_path() == ""

    coordinator.activate_grpc(profile="paper", target="localhost:50051")
    assert coordinator.active_decision_log_path() == "grpc://localhost:50051"

    log_path = Path("/tmp/decision-log.jsonl")
    coordinator.activate_jsonl(profile="paper", log_path=log_path)
    assert coordinator.active_decision_log_path() == str(log_path)

    coordinator.activate_demo(profile="paper")
    assert coordinator.active_decision_log_path() == "offline-demo"


def test_decision_source_selector_current_feed_mode_contract() -> None:
    coordinator = DecisionSourceFallbackCoordinator()

    coordinator.activate_grpc(profile="paper", target="localhost:50051")
    assert coordinator.current_feed_mode() == "grpc"

    coordinator.activate_jsonl(profile="paper", log_path=Path("/tmp/decision-log.jsonl"))
    assert coordinator.current_feed_mode() == "file"

    coordinator.activate_demo(profile="paper")
    assert coordinator.current_feed_mode() == "demo"


def test_decision_source_selector_current_feed_adapter_label_contract() -> None:
    coordinator = DecisionSourceFallbackCoordinator()

    coordinator.activate_grpc(profile="paper", target="localhost:50051")
    assert coordinator.current_feed_adapter_label(status="connected", loader_is_demo=False) == "grpc"

    coordinator.activate_jsonl(profile="paper", log_path=Path("/tmp/decision-log.jsonl"))
    assert coordinator.current_feed_adapter_label(status="connected", loader_is_demo=False) == "jsonl"

    coordinator.activate_demo(profile="paper")
    assert coordinator.current_feed_adapter_label(status="connected", loader_is_demo=True) == "demo"

    assert coordinator.current_feed_adapter_label(status="fallback", loader_is_demo=True) == "fallback"


def test_decision_source_selector_current_transport_key_contract() -> None:
    coordinator = DecisionSourceFallbackCoordinator()

    coordinator.activate_grpc(profile="paper", target="localhost:50051")
    assert coordinator.current_transport_key(grpc_stream_active=False) == "grpc"

    coordinator.activate_jsonl(profile="paper", log_path=Path("/tmp/decision-log.jsonl"))
    assert coordinator.current_transport_key(grpc_stream_active=False) == "fallback"

    coordinator.activate_demo(profile="paper")
    assert coordinator.current_transport_key(grpc_stream_active=False) == "fallback"

    coordinator.set_state(profile="paper", log_path=None, stream_label=None)
    assert coordinator.current_transport_key(grpc_stream_active=False) == "fallback"


def test_decision_source_selector_fallback_rule_prefers_jsonl_then_demo() -> None:
    coordinator = DecisionSourceFallbackCoordinator()

    assert coordinator.fallback_source(jsonl_available=True) == "jsonl"
    assert coordinator.fallback_source(jsonl_available=False) == "demo"

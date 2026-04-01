"""Koordynator wyboru źródła decyzji (gRPC / JSONL / demo)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RuntimeDecisionMode = Literal["prod", "demo", "test"]


@dataclass(frozen=True)
class DecisionSourceState:
    profile: str | None = None
    log_path: Path | None = None
    stream_label: str | None = None
    runtime_mode: RuntimeDecisionMode = "prod"


class DecisionSourceFallbackCoordinator:
    """Enkapsuluje wybór aktywnego źródła i reguły fallbacku."""

    def __init__(self, *, runtime_mode: RuntimeDecisionMode = "prod") -> None:
        self._runtime_mode: RuntimeDecisionMode = self.normalize_runtime_mode(runtime_mode)
        self._state = DecisionSourceState(runtime_mode=self._runtime_mode)

    @property
    def state(self) -> DecisionSourceState:
        return self._state

    def set_state(
        self,
        *,
        profile: str | None,
        log_path: Path | None,
        stream_label: str | None,
    ) -> DecisionSourceState:
        self._state = DecisionSourceState(
            profile=profile,
            log_path=log_path,
            stream_label=stream_label,
            runtime_mode=self._runtime_mode,
        )
        return self._state

    @property
    def runtime_mode(self) -> RuntimeDecisionMode:
        return self._runtime_mode

    @classmethod
    def normalize_runtime_mode(cls, value: str | None) -> RuntimeDecisionMode:
        normalized = (value or "").strip().lower()
        if normalized in {"prod", "production", "live"}:
            return "prod"
        if normalized in {"demo", "paper"}:
            return "demo"
        if normalized in {"test", "testing"}:
            return "test"
        return "prod"

    def activate_grpc(self, *, profile: str | None, target: str) -> DecisionSourceState:
        return self.set_state(profile=profile, log_path=None, stream_label=f"grpc://{target}")

    def activate_jsonl(self, *, profile: str | None, log_path: Path) -> DecisionSourceState:
        return self.set_state(profile=profile, log_path=log_path, stream_label=None)

    def activate_demo(self, *, profile: str | None) -> DecisionSourceState:
        return self.set_state(profile=profile, log_path=None, stream_label="offline-demo")

    def active_decision_log_path(self) -> str:
        label = self._state.stream_label
        if label:
            return label
        if self._state.log_path is None:
            return ""
        return str(self._state.log_path)

    def current_feed_mode(self) -> str:
        label = self._state.stream_label or ""
        if label.startswith("grpc://"):
            return "grpc"
        if label == "offline-demo":
            return "demo"
        if label:
            return "file"
        if self._state.log_path is not None:
            return "file"
        return "demo"

    def current_feed_label(self) -> str:
        if self._state.stream_label:
            return self._state.stream_label
        if self._state.log_path is not None:
            return str(self._state.log_path)
        return ""

    def current_feed_adapter_label(self, *, status: str, loader_is_demo: bool) -> str:
        if status == "fallback":
            return "fallback"
        label = self._state.stream_label or ""
        if label.startswith("grpc://"):
            return "grpc"
        if label == "offline-demo":
            return "demo"
        if label:
            prefix, _, _ = label.partition("://")
            return prefix or label
        if self._state.log_path is not None:
            return "jsonl"
        if loader_is_demo:
            return "demo"
        return "unknown"

    def current_transport_key(self, *, grpc_stream_active: bool) -> str:
        label = self._state.stream_label or ""
        if label.startswith("grpc://") or grpc_stream_active:
            return "grpc"
        return "fallback"

    def fallback_source(self, *, jsonl_available: bool) -> str:
        if self._runtime_mode == "demo":
            return "demo"
        return "jsonl" if jsonl_available else "demo"

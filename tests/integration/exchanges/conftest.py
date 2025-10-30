import pytest

from bot_core.exchanges.rate_limiter import (
    RateLimiterRegistry,
    get_global_rate_limiter_registry,
    set_global_rate_limiter_registry,
)


class _FakeRateLimiter:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def acquire(self, weight: float = 1.0) -> None:  # noqa: D401 - stub
        self.calls.append(weight)


class _RegistryStub(RateLimiterRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.created: dict[str, _FakeRateLimiter] = {}

    def configure(self, key: str, rules, *, clock=None, metrics_registry=None, metric_labels=None):
        limiter = _FakeRateLimiter()
        self.created[key] = limiter
        return limiter


@pytest.fixture
def rate_limiter_registry():
    original = get_global_rate_limiter_registry()
    registry = _RegistryStub()
    set_global_rate_limiter_registry(registry)
    try:
        yield registry
    finally:
        set_global_rate_limiter_registry(original)

"""Adaptive learning helpers integrating regime-aware strategy selection."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from .inference import ModelRepository
from .models import ModelArtifact

try:  # pragma: no cover - DecisionOrchestrator may be absent in trimmed builds
    from bot_core.decision.orchestrator import DecisionOrchestrator
except Exception:  # pragma: no cover - fallback for stripped distributions
    DecisionOrchestrator = None  # type: ignore[misc, assignment]

try:  # pragma: no cover - optional enum import to avoid cyclic dependency
    from bot_core.ai.regime import MarketRegime
except Exception:  # pragma: no cover - fallback for minimal builds
    MarketRegime = None  # type: ignore[misc, assignment]


_LOGGER = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalise_regime(value: str | object) -> str:
    if MarketRegime is not None:
        try:
            return MarketRegime(value).value  # type: ignore[arg-type]
        except Exception:
            pass
    if isinstance(value, str) and value:
        return value.lower()
    return "trend"


@dataclass(slots=True)
class AdaptiveStrategyStats:
    """Aggregated reward statistics for a single strategy arm."""

    name: str
    plays: int = 0
    total_reward: float = 0.0
    total_squared_reward: float = 0.0
    last_reward: float = 0.0
    updated_at: datetime = field(default_factory=_now_utc)

    def record(self, reward: float, timestamp: datetime | None = None) -> None:
        reference = timestamp or _now_utc()
        self.plays += 1
        self.total_reward += reward
        self.total_squared_reward += reward * reward
        self.last_reward = reward
        self.updated_at = reference

    @property
    def mean_reward(self) -> float:
        if self.plays <= 0:
            return 0.0
        return self.total_reward / float(self.plays)

    @property
    def reward_variance(self) -> float:
        if self.plays <= 1:
            return 0.0
        mean = self.mean_reward
        return max(0.0, (self.total_squared_reward / float(self.plays)) - mean * mean)


@dataclass(slots=True)
class AdaptiveRegimePolicy:
    """Contextual multi-armed bandit for a single market regime."""

    regime: str
    exploration: float = 0.65
    min_trials: int = 3
    _stats: MutableMapping[str, AdaptiveStrategyStats] = field(default_factory=dict, repr=False)
    _total_plays: int = 0

    def ensure_strategy(self, name: str, *, bootstrap_reward: float | None = None) -> None:
        key = name.lower()
        stats = self._stats.get(key)
        if stats is None:
            stats = AdaptiveStrategyStats(name=name)
            if bootstrap_reward is not None and math.isfinite(bootstrap_reward):
                stats.record(float(bootstrap_reward))
                stats.plays -= 1  # undo count increment while keeping reward baseline
            self._stats[key] = stats

    def update(self, name: str, reward: float, timestamp: datetime | None = None) -> None:
        key = name.lower()
        if key not in self._stats:
            self._stats[key] = AdaptiveStrategyStats(name=name)
        self._stats[key].record(reward, timestamp=timestamp)
        self._total_plays += 1

    def recommend(self) -> str | None:
        if not self._stats:
            return None
        # explore untested strategies first
        for stats in self._stats.values():
            if stats.plays < max(1, self.min_trials):
                return stats.name
        total = max(1, self._total_plays)
        best_name: str | None = None
        best_score: float | None = None
        for stats in self._stats.values():
            mean = stats.mean_reward
            bonus = math.sqrt(math.log(total + 1.0) / max(1, stats.plays))
            score = mean + self.exploration * bonus
            if best_score is None or score > best_score:
                best_score = score
                best_name = stats.name
        return best_name

    def strategies(self) -> Sequence[AdaptiveStrategyStats]:
        return tuple(self._stats.values())

    def to_payload(self) -> Mapping[str, object]:
        return {
            "regime": self.regime,
            "total_plays": self._total_plays,
            "strategies": [
                {
                    "name": stats.name,
                    "plays": stats.plays,
                    "total_reward": stats.total_reward,
                    "total_squared_reward": stats.total_squared_reward,
                    "last_reward": stats.last_reward,
                    "updated_at": stats.updated_at.isoformat(),
                }
                for stats in self._stats.values()
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptiveRegimePolicy":
        regime = _normalise_regime(payload.get("regime"))
        policy = cls(regime=regime)
        policy._total_plays = int(payload.get("total_plays", 0) or 0)
        entries = payload.get("strategies")
        if isinstance(entries, Sequence):
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                name = str(entry.get("name") or "").strip()
                if not name:
                    continue
                stats = AdaptiveStrategyStats(name=name)
                stats.plays = int(entry.get("plays", 0) or 0)
                stats.total_reward = float(entry.get("total_reward", 0.0) or 0.0)
                stats.total_squared_reward = float(entry.get("total_squared_reward", 0.0) or 0.0)
                stats.last_reward = float(entry.get("last_reward", 0.0) or 0.0)
                updated_at = entry.get("updated_at")
                if isinstance(updated_at, str):
                    try:
                        stats.updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    except ValueError:
                        stats.updated_at = _now_utc()
                policy._stats[name.lower()] = stats
        return policy


@dataclass(slots=True)
class AdaptiveStrategyLearner:
    """Coordinates adaptive strategy selection with runtime orchestrator."""

    repository: ModelRepository
    orchestrator: DecisionOrchestrator | None
    model_name: str = "adaptive_strategy_policy.json"
    exploration: float = 0.65
    min_trials: int = 3
    _policies: MutableMapping[str, AdaptiveRegimePolicy] = field(default_factory=dict, init=False)
    _dirty: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._load_state()

    # ------------------------------------------------------------------ persistence --
    def _state_path(self) -> Path:
        return self.repository.base_path / self.model_name

    def _load_state(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            artifact = self.repository.load(path)
        except Exception:  # pragma: no cover - corrupted state should not break runtime
            _LOGGER.exception("Nie udało się odczytać stanu adaptive learnera z %s", path)
            return
        state = artifact.model_state.get("policies") if isinstance(artifact.model_state, Mapping) else None
        if not isinstance(state, Mapping):
            return
        for regime_key, payload in state.items():
            if not isinstance(payload, Mapping):
                continue
            policy = AdaptiveRegimePolicy.from_payload(payload)
            policy.exploration = float(artifact.metadata.get("exploration", self.exploration))
            policy.min_trials = int(artifact.metadata.get("min_trials", self.min_trials))
            self._policies[str(regime_key)] = policy
        _LOGGER.debug("Załadowano %s polityk adaptive learnera", len(self._policies))

    def _build_artifact(self) -> ModelArtifact:
        now = _now_utc()
        policies = {key: policy.to_payload() for key, policy in self._policies.items()}
        total_plays = sum(policy._total_plays for policy in self._policies.values())
        summary = {
            "policies": len(self._policies),
            "total_plays": total_plays,
            "exploration": self.exploration,
        }
        metadata = {
            "policy_type": "adaptive_bandit",
            "exploration": self.exploration,
            "min_trials": self.min_trials,
            "updated_at": now.isoformat(),
        }
        artifact = ModelArtifact(
            feature_names=("regime", "strategy"),
            model_state={"policies": policies},
            trained_at=now,
            metrics={"summary": summary},
            metadata=metadata,
            target_scale=1.0,
            training_rows=total_plays,
            validation_rows=0,
            test_rows=0,
            feature_scalers={},
        )
        return artifact

    def persist(self) -> Path | None:
        if not self._dirty:
            return None
        artifact = self._build_artifact()
        timestamp = _now_utc().strftime("%Y%m%dT%H%M%S")
        try:
            path = self.repository.save(
                artifact,
                self.model_name,
                version=timestamp,
                aliases=("latest",),
                activate=True,
            )
        except Exception:  # pragma: no cover - persistence errors should be logged only
            _LOGGER.exception("Nie udało się zapisać stanu adaptive learnera")
            return None
        self._dirty = False
        return path

    # ------------------------------------------------------------------ runtime API --
    def _policy_for(self, regime: str) -> AdaptiveRegimePolicy:
        key = _normalise_regime(regime)
        policy = self._policies.get(key)
        if policy is None:
            policy = AdaptiveRegimePolicy(regime=key, exploration=self.exploration, min_trials=self.min_trials)
            self._policies[key] = policy
        return policy

    def register_strategies(self, regime: str, strategies: Iterable[str]) -> None:
        policy = self._policy_for(regime)
        for name in strategies:
            policy.ensure_strategy(name)

    def observe(
        self,
        *,
        regime: str,
        strategy: str,
        metrics: Mapping[str, float] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        policy = self._policy_for(regime)
        reward = self._reward_from_metrics(metrics or {})
        policy.update(strategy, reward, timestamp=timestamp)
        self._dirty = True
        orchestrator = self.orchestrator
        if orchestrator is not None:
            try:
                orchestrator.record_strategy_performance(
                    strategy,
                    regime,
                    hit_rate=float(metrics.get("hit_rate", reward)) if metrics else reward,
                    pnl=float(metrics.get("pnl", 0.0)) if metrics else reward,
                    sharpe=float(metrics.get("sharpe", reward)) if metrics else reward,
                    observations=1,
                    timestamp=timestamp,
                )
            except Exception:  # pragma: no cover - orchestrator integration should not be fatal
                _LOGGER.debug("Nie udało się zarejestrować metryk strategii w orchestratorze", exc_info=True)

    @staticmethod
    def _reward_from_metrics(metrics: Mapping[str, float]) -> float:
        hit_rate = float(metrics.get("hit_rate", 0.5) or 0.5)
        pnl = float(metrics.get("pnl", 0.0) or 0.0)
        sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
        reward = 0.6 * hit_rate + 0.3 * math.tanh(pnl / 50.0) + 0.1 * math.tanh(sharpe / 2.0)
        return max(-1.0, min(1.0, reward))

    def recommend(self, regime: str) -> str | None:
        policy = self._policies.get(_normalise_regime(regime))
        candidate = policy.recommend() if policy is not None else None
        if candidate:
            return candidate
        orchestrator = self.orchestrator
        if orchestrator is None:
            return None
        try:
            return orchestrator.select_strategy(regime)
        except Exception:  # pragma: no cover - orchestrator may not implement selection
            _LOGGER.debug("Nie udało się pobrać strategii z orchestratora", exc_info=True)
            return None

    def build_dynamic_preset(
        self,
        regime: str,
        *,
        metrics: Mapping[str, float] | None = None,
    ) -> Mapping[str, object] | None:
        normalized_metrics: dict[str, float] = {}
        if metrics:
            for key, value in metrics.items():
                try:
                    normalized_metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        strategy = self.recommend(regime)
        if strategy is None:
            return None
        regime_key = _normalise_regime(regime)
        policy = self._policies.get(regime_key)
        stats_map: dict[str, AdaptiveStrategyStats] = {}
        if policy is not None:
            for stats in policy.strategies():
                stats_map[stats.name] = stats
        if strategy not in stats_map:
            stats_map[strategy] = AdaptiveStrategyStats(name=strategy)
        ranked = sorted(
            stats_map.values(),
            key=lambda item: (item.name == strategy, item.mean_reward, item.plays),
            reverse=True,
        )
        top_candidates = ranked[: max(1, min(3, len(ranked)))]
        base_scores: dict[str, float] = {}
        total_plays = float(policy._total_plays) if policy is not None else 0.0
        for stats in top_candidates:
            exploitation = max(-1.0, min(1.0, stats.mean_reward))
            exploration_bonus = 0.0
            if policy is not None and stats.plays > 0 and total_plays > 0:
                exploration_bonus = math.sqrt(
                    math.log(total_plays + 1.0) / max(1, stats.plays)
                )
            base_scores[stats.name] = max(
                0.0, exploitation + self.exploration * exploration_bonus + 1.0
            )
        if not base_scores:
            base_scores[strategy] = 1.0
        total_score = sum(base_scores.values())
        if total_score <= 0:
            base_distribution = {
                name: 1.0 / float(len(base_scores)) for name in base_scores
            }
        else:
            base_distribution = {
                name: value / total_score for name, value in base_scores.items()
            }
        confidence = normalized_metrics.get("confidence", 0.6)
        if not math.isfinite(confidence):
            confidence = 0.6
        confidence = max(0.0, min(1.0, float(confidence)))
        risk_score = normalized_metrics.get("risk_score", 0.0)
        if not math.isfinite(risk_score):
            risk_score = 0.0
        risk_score = max(0.0, min(1.0, float(risk_score)))
        primary_weight = max(0.35, min(0.9, 0.5 + 0.4 * confidence))
        risk_penalty = max(0.25, 1.0 - 0.5 * risk_score)
        primary_weight = min(0.95, primary_weight * risk_penalty)
        weights: dict[str, float] = {}
        if len(base_distribution) == 1:
            weights[strategy] = 1.0
        else:
            residual = max(0.0, 1.0 - primary_weight)
            others_total = sum(
                value for name, value in base_distribution.items() if name != strategy
            )
            for name, fraction in base_distribution.items():
                if name == strategy:
                    weights[name] = primary_weight
                elif others_total > 0:
                    weights[name] = residual * (fraction / others_total)
                else:
                    weights[name] = 0.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: value / total_weight for name, value in weights.items()}
        else:
            weights = {strategy: 1.0}
        strategies_payload = [
            {
                "name": name,
                "weight": value,
                "metadata": {
                    "source": "adaptive_learner",
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "metrics": dict(normalized_metrics),
                },
            }
            for name, value in weights.items()
        ]
        payload = {
            "name": f"adaptive::{regime_key}",
            "regime": regime_key,
            "strategies": strategies_payload,
            "metadata": {
                "plays": policy._total_plays if policy is not None else 0,
                "exploration": self.exploration,
                "confidence": confidence,
                "risk_score": risk_score,
            },
            "metrics": dict(normalized_metrics),
            "generated_at": _now_utc().isoformat(),
        }
        return payload

    def snapshot(self) -> Mapping[str, object]:
        return {key: policy.to_payload() for key, policy in self._policies.items()}


__all__ = ["AdaptiveStrategyLearner", "AdaptiveRegimePolicy", "AdaptiveStrategyStats"]

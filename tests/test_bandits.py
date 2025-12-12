from __future__ import annotations

import numpy as np

from bot_core.decision.bandits import _LinUCBArm, _ThompsonArm


def test_linucb_predict_reflects_exploration_alpha():
    context = np.array([1.0, -0.5, 0.25])
    low_alpha = _LinUCBArm(dimension=1, alpha=0.1)
    high_alpha = _LinUCBArm(dimension=1, alpha=2.0)

    low_score = low_alpha.predict(context)
    high_score = high_alpha.predict(context)

    assert high_score > low_score > 0.0


def test_linucb_updates_weights_after_reward():
    arm = _LinUCBArm(dimension=2, alpha=0.5)
    context = np.array([1.0, 0.0])

    base_score = arm.predict(context)
    arm.update(context, reward=1.0)
    improved_score = arm.predict(context)

    assert improved_score > base_score


def test_linucb_expands_dimension_on_larger_context():
    arm = _LinUCBArm(dimension=1, alpha=0.3)
    context = np.array([1.0, 0.5, -0.25, 0.1])

    arm.predict(context)

    assert arm.dimension == context.size
    assert arm.A.shape == (context.size, context.size)
    assert arm.b.shape == (context.size,)


def test_thompson_arm_posterior_updates():
    arm = _ThompsonArm(alpha=1.0, beta=1.0)

    prior_mean = arm.posterior_mean()
    arm.update(1.0)
    mean_after_success = arm.posterior_mean()
    arm.update(0.0)
    mean_after_failure = arm.posterior_mean()

    assert prior_mean == 0.5
    assert mean_after_success > prior_mean
    assert abs(mean_after_failure - 0.5) < 1e-6


def test_thompson_arm_clamps_outcomes_to_unit_interval():
    arm = _ThompsonArm(alpha=2.0, beta=3.0)

    arm.update(10.0)
    arm.update(-5.0)

    mean = arm.posterior_mean()

    assert 0.0 < mean < 1.0

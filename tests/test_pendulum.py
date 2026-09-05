from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gae_credit.envs.pendulum import PendulumEnv


def make_env(**overrides):
    parameters = dict(
        max_steps=200,
        dt=0.05,
        max_torque=2.0,
        max_speed=8.0,
        g=10.0,
        m=1.0,
        l=1.0,
        include_time_remaining=True,
    )
    parameters.update(overrides)
    return PendulumEnv(SimpleNamespace(**parameters))


def test_reset_observation_and_initial_distribution():
    env = make_env()
    for seed in range(25):
        observation, info = env.reset(seed=seed)
        assert observation.shape == (4,)
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)
        assert -np.pi <= info["theta"] < np.pi
        assert -1 <= info["theta_dot"] <= 1
        assert observation[-1] == 1
        assert info["timestep"] == 0


def test_seeded_reset_is_reproducible_and_distinct():
    env = make_env()
    first, _ = env.reset(seed=7)
    same, _ = env.reset(seed=7)
    different, _ = env.reset(seed=8)
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize("action,bound", [(100.0, 2.0), (-100.0, -2.0)])
def test_action_is_clipped_before_dynamics_and_cost(action, bound):
    env = make_env()
    env.reset(seed=0)
    snapshot = env.get_state()
    actual = env.step(np.array([action]))
    env.set_state(snapshot)
    expected = env.step(np.array([bound]))
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    assert actual[4]["torque"] == bound
    assert actual[4]["torque_cost"] == pytest.approx(0.004)


def test_post_transition_dynamics_and_cost_hand_calculation():
    env = make_env()
    env.reset(seed=0)
    state = env.get_state()
    state.update(theta=0.0, theta_dot=0.0)
    env.set_state(state)
    _, reward, _, _, info = env.step([1.0])
    # Acceleration=3, new velocity=0.15, new theta=0.0075.
    assert info["theta_dot"] == pytest.approx(0.15)
    assert info["theta"] == pytest.approx(0.0075)
    assert reward == pytest.approx(-(0.0075**2 + 0.1 * 0.15**2 + 0.001))


def test_finite_task_terminal_and_time_remaining():
    env = make_env()
    env.reset(seed=42)
    for step in range(1, 201):
        observation, reward, terminated, truncated, info = env.step([0.0])
        assert info["timestep"] == step
        assert observation[-1] == pytest.approx((200 - step) / 200)
        assert terminated is (step == 200)
        assert truncated is False
        assert reward == -(info["angle_cost"] + info["velocity_cost"] + info["torque_cost"])
    with pytest.raises(RuntimeError, match="terminated"):
        env.step([0.0])


def test_snapshot_restores_next_transition_and_rng():
    env = make_env()
    env.reset(seed=93)
    env.step([0.25])
    snapshot = env.get_state()
    expected = env.step([-0.7])
    expected_reset, _ = env.reset()
    env.set_state(snapshot)
    actual = env.step([-0.7])
    actual_reset, _ = env.reset()
    np.testing.assert_array_equal(expected[0], actual[0])
    assert expected[1:] == actual[1:]
    np.testing.assert_array_equal(expected_reset, actual_reset)


def test_snapshot_has_no_shared_mutable_rng_state():
    env = make_env()
    env.reset(seed=12)
    snapshot = env.get_state()
    original = deepcopy(snapshot)
    snapshot["rng_state"]["state"]["state"] += 10
    assert env.get_state() == original
    env.set_state(original)
    original["rng_state"]["state"]["state"] += 10
    assert env.get_state()["rng_state"] != original["rng_state"]


def test_random_rollout_is_finite_and_within_observation_bounds():
    env = make_env()
    env.reset(seed=0)
    generator = np.random.default_rng(20)
    for _ in range(200):
        observation, reward, _, _, info = env.step(generator.uniform(-2, 2, size=1))
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert np.all(np.isfinite(list(info.values())))


@pytest.mark.parametrize("action", [[np.nan], [np.inf], [1, 2], []])
def test_reject_invalid_actions_without_mutating_state(action):
    env = make_env()
    env.reset(seed=0)
    before = env.get_state()
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    assert env.get_state() == before


def test_requires_reset_before_step():
    with pytest.raises(RuntimeError, match="reset"):
        make_env().step([0.0])


@pytest.mark.parametrize("overrides", [{"max_steps": 0}, {"max_steps": -1}, {"dt": 0}])
def test_invalid_environment_parameters(overrides):
    with pytest.raises(ValueError):
        make_env(**overrides)


def test_terminal_snapshot_cannot_be_stepped():
    env = make_env(max_steps=1)
    env.reset(seed=0)
    env.step([0.0])
    state = env.get_state()
    env.reset()
    env.set_state(state)
    with pytest.raises(RuntimeError, match="terminated"):
        env.step([0.0])

from types import SimpleNamespace

import numpy as np
import pytest

from gae_credit.envs.rewards import (
    BlockDelayedReward,
    DenseReward,
    SparseUprightReward,
    make_reward,
)


def transform(schedule, reward=-1.0, theta=0.0, velocity=0.0, step=1, final=False):
    return schedule.transform(
        base_dense_reward=reward,
        theta=theta,
        theta_dot=velocity,
        timestep=step,
        terminated=final,
    )


def test_dense_reward_is_exactly_base_reward():
    schedule = DenseReward()
    for reward in [-15.19, 0.0, 1.7]:
        assert transform(schedule, reward=reward)[0] == reward
    state = schedule.get_state()
    schedule.reset()
    schedule.set_state(state)


def test_delayed_one_equals_dense():
    delayed = BlockDelayedReward(1, 0.995)
    for step, base in enumerate([-1.0, -2.3, 0.0], start=1):
        reward, diagnostics = transform(delayed, reward=base, step=step)
        assert reward == base
        assert diagnostics["emitted_payout"]
        assert diagnostics["payout_count"] == 1
        assert diagnostics["pending_reward"] == 0


def test_full_block_discount_correction_hand_calculation():
    schedule = BlockDelayedReward(2, 0.5)
    reward, first = transform(schedule, reward=1.0)
    assert reward == 0
    assert first["block_id"] == 0
    assert first["block_position"] == 1
    assert first["pending_reward"] == 1.0
    assert first["payout_count"] == 0
    reward, second = transform(schedule, reward=2.0, step=2)
    assert reward == 4.0  # 1 / 0.5 + 2.
    assert second["discount_corrected_payout"] == 4.0
    assert second["payout_count"] == 2
    assert second["pending_reward"] == 0
    assert second["pending_corrected_reward"] == 0
    _, third = transform(schedule, reward=3.0, step=3)
    assert third["block_id"] == 1
    assert third["block_position"] == 1


def test_partial_block_flush_hand_calculation():
    schedule = BlockDelayedReward(4, 0.5)
    assert transform(schedule, reward=1.0)[0] == 0
    assert transform(schedule, reward=2.0, step=2)[0] == 0
    reward, info = transform(schedule, reward=3.0, step=3, final=True)
    assert reward == 11.0  # 1 / 0.5**2 + 2 / 0.5 + 3.
    assert info["payout_count"] == 3
    assert schedule.get_state()["count"] == 0


@pytest.mark.parametrize("delay", [8, 32])
@pytest.mark.parametrize("length", [5, 200])
@pytest.mark.parametrize("gamma", [0.995, 1.0])
def test_discounted_return_preserved_for_full_and_partial_blocks(delay, length, gamma):
    schedule = BlockDelayedReward(delay, gamma)
    dense = np.random.default_rng(0).normal(-2.0, 0.5, size=length)
    delayed = np.asarray(
        [
            transform(schedule, reward=reward, step=i + 1, final=i == length - 1)[0]
            for i, reward in enumerate(dense)
        ]
    )
    discounts = gamma ** np.arange(length)
    assert delayed @ discounts == pytest.approx(dense @ discounts, rel=1e-13, abs=1e-12)
    assert schedule.get_state()["pending_reward"] == 0


def test_reset_clears_delayed_buffer():
    schedule = BlockDelayedReward(8, 0.5)
    transform(schedule, reward=10.0)
    schedule.reset()
    reward, info = transform(schedule, reward=1.0, final=True)
    assert reward == 1.0
    assert info["block_id"] == 0
    assert info["payout_count"] == 1


def test_delayed_snapshot_restores_buffer_exactly():
    schedule = BlockDelayedReward(3, 0.995)
    transform(schedule, reward=-0.17)
    snapshot = schedule.get_state()
    transform(schedule, reward=-1.9, step=2)
    expected = transform(schedule, reward=-3.1, step=3)
    schedule.set_state(snapshot)
    snapshot["pending_reward"] = 1000
    transform(schedule, reward=-1.9, step=2)
    actual = transform(schedule, reward=-3.1, step=3)
    assert actual == expected


@pytest.mark.parametrize("delay,gamma", [(0, 0.99), (-1, 0.99), (True, 0.99), (8, 0)])
def test_invalid_delayed_configuration(delay, gamma):
    with pytest.raises(ValueError):
        BlockDelayedReward(delay, gamma)


@pytest.mark.parametrize("angle", [0.0, 0.1, -0.1, 2 * np.pi + 0.1, -2 * np.pi - 0.1])
def test_sparse_reward_wraps_angle(angle):
    reward, info = transform(SparseUprightReward(), reward=-50.0, theta=angle)
    assert reward == 1
    assert info["upright"]
    assert info["entered_upright"]
    assert info["first_upright_timestep"] == 1


@pytest.mark.parametrize("angle,velocity", [(0.263, 0), (0, 1), (0, -1), (0, 3)])
def test_sparse_reward_rejects_bad_angle_or_high_velocity(angle, velocity):
    reward, info = transform(SparseUprightReward(), theta=angle, velocity=velocity)
    assert reward == 0
    assert not info["upright"]
    assert info["first_upright_timestep"] is None


def test_repeated_sparse_rewards_and_stable_success():
    schedule = SparseUprightReward()
    for step in range(1, 11):
        reward, info = transform(schedule, step=step)
        assert reward == 1
        assert info["upright_streak"] == step
        assert info["stable_success"] is (step >= 10)
        assert info["entered_upright"] is (step == 1)
    reward, info = transform(schedule, theta=np.pi, step=11)
    assert reward == 0
    assert info["left_upright"]
    assert info["upright_streak"] == 0
    assert info["longest_upright_streak"] == 10
    assert info["stable_success"]
    assert info["first_upright_timestep"] == 1


def test_flythrough_is_not_stable_success_and_reset_clears_history():
    schedule = SparseUprightReward()
    transform(schedule, theta=np.pi)
    _, entered = transform(schedule, theta=0.0, step=2)
    _, left = transform(schedule, theta=0.0, velocity=2.0, step=3)
    assert entered["first_upright_timestep"] == 2
    assert left["left_upright"]
    assert not left["stable_success"]
    schedule.reset()
    _, info = transform(schedule, theta=np.pi)
    assert info["first_upright_timestep"] is None
    assert info["longest_upright_streak"] == 0


def test_sparse_snapshot_restores_streak():
    schedule = SparseUprightReward()
    for step in range(1, 10):
        transform(schedule, step=step)
    state = schedule.get_state()
    expected = transform(schedule, step=10)
    schedule.reset()
    schedule.set_state(state)
    assert transform(schedule, step=10) == expected


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("dense", DenseReward),
        ("delayed", BlockDelayedReward),
        ("sparse", SparseUprightReward),
    ],
)
def test_reward_factory(kind, expected):
    assert isinstance(make_reward(SimpleNamespace(kind=kind, delay_block_size=8), 0.995), expected)

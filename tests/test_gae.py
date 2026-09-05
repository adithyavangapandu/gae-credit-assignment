import numpy as np
import pytest

from gae_credit.estimators.gae import (
    compute_event_credit,
    compute_gae,
    compute_omitted_tail,
    compute_segment_credit,
    compute_sign_disagreement,
    compute_tail_fraction,
)


def test_one_step_is_td_residual_and_terminal_value_is_ignored():
    result = compute_gae([1, 2, 3], [10, 20, 30, 999], [False, False, True], 0.5, 0.9, 1)
    np.testing.assert_array_equal(result.td_errors, [1, -3, -27])
    np.testing.assert_array_equal(result.advantages, [1, -3, -27])
    np.testing.assert_array_equal(result.value_targets, [11, 17, 3])
    np.testing.assert_array_equal(result.effective_horizons, [1, 1, 1])


def test_three_step_has_exactly_three_residuals_hand_calculation():
    result = compute_gae([1, 2, 3, 4], np.zeros(5), [False] * 3 + [True], 0.5, 1, 3)
    # At t=0 omit q**3 * 4; near the boundary the horizon becomes 2, then 1.
    np.testing.assert_array_equal(result.advantages, [2.75, 4.5, 5, 4])
    np.testing.assert_array_equal(result.effective_horizon_at_each_timestep, [3, 3, 2, 1])


def test_full_reverse_matches_hand_expansion_and_bounded_iterative():
    arguments = ([1, 2, 3, 4], np.zeros(5), [False] * 3 + [True], 0.5, 1)
    full = compute_gae(*arguments, horizon="full")
    np.testing.assert_array_equal(full.advantages, [3.25, 4.5, 5, 4])
    np.testing.assert_array_equal(full.effective_horizons, [4, 3, 2, 1])
    for horizon in (4, 16, 200):
        iterative = compute_gae(*arguments, horizon=horizon)
        np.testing.assert_array_equal(iterative.advantages, full.advantages)


def test_gamma_and_lambda_both_weight_the_residual_tail():
    result = compute_gae([1, 2, 3, 4], np.zeros(5), [0, 0, 0, 1], 0.5, 0.5, 3)
    np.testing.assert_array_equal(result.advantages, [1.6875, 3, 4, 4])


@pytest.mark.parametrize("horizon", [3, 16, "full"])
def test_gae_never_crosses_an_episode_boundary(horizon):
    result = compute_gae([1, 2, 10, 20], [3, 4, 5, 6, 999], [0, 1, 0, 1], 0.5, 1, horizon)
    # Final residual is reward - value = 20 - 6 = 14, with zero bootstrap.
    np.testing.assert_array_equal(result.td_errors, [0, -2, 8, 14])
    np.testing.assert_array_equal(result.advantages, [-1, -2, 15, 14])
    np.testing.assert_array_equal(result.effective_horizons, [2, 1, 2, 1])


def test_full_lambda_one_telescopes_to_monte_carlo_return_minus_value():
    result = compute_gae([2, -1, 4], [1, 2, -3, 999], [0, 0, 1], 0.5, 1, "full")
    # Monte Carlo returns are [2-.5+1, -1+2, 4] = [2.5,1,4].
    np.testing.assert_array_equal(result.advantages, [1.5, -1, 7])
    np.testing.assert_array_equal(result.value_targets, [2.5, 1, 4])


def test_nonterminal_array_boundary_bootstraps_last_value():
    result = compute_gae([1], [2, 10], [False], 0.5, 1, "full")
    np.testing.assert_array_equal(result.advantages, [4])
    np.testing.assert_array_equal(result.value_targets, [6])


@pytest.mark.parametrize("horizon", [1, 3, 16, "full"])
def test_unit_event_credit_respects_actual_horizon(horizon):
    rewards = [0, 0, 0, 1, 0]
    values = [2, 1, -1, 0, 3, 999]
    credit = compute_event_credit(rewards, values, [0, 0, 0, 0, 1], 3, 0.5, 1, horizon)
    expected = {
        1: [0, 0, 0, 1, 0],
        3: [0, 0.25, 0.5, 1, 0],
        16: [0.125, 0.25, 0.5, 1, 0],
        "full": [0.125, 0.25, 0.5, 1, 0],
    }
    np.testing.assert_array_equal(credit, expected[horizon])


def test_nonunit_signed_event_credit_and_terminal_boundary():
    credit = compute_event_credit([1, 1, 0, -4], np.zeros(5), [0, 1, 0, 1], 3, 0.5, 1)
    np.testing.assert_array_equal(credit, [0, 0, -2, -4])


def test_segment_credit_is_sum_of_event_credits_with_fixed_values():
    credit = compute_segment_credit([1, 2, 4, 8], [3, 4, 5, 6, 7], [0, 0, 0, 1], 1, 3, 0.5, 1)
    # Remove rewards 2 and 4, but retain the final reward 8.
    np.testing.assert_array_equal(credit, [2, 4, 4, 0])


def test_omitted_tail_and_bounded_fraction_handle_cancellation():
    full = np.array([3.25, 0, -1, 0])
    short = np.array([2.75, 2, 1, 0])
    np.testing.assert_array_equal(compute_omitted_tail(full, short), [0.5, -2, -2, 0])
    np.testing.assert_allclose(compute_tail_fraction(full, short), [0.5 / 3.25, 0.5, 2 / 3, 0])
    np.testing.assert_array_equal(
        compute_sign_disagreement(full, short), [False, False, True, False]
    )
    np.testing.assert_array_equal(compute_omitted_tail(full, full), np.zeros(4))
    np.testing.assert_array_equal(compute_tail_fraction(full, full), np.zeros(4))


def test_inputs_are_not_mutated_by_estimation_or_event_diagnostics():
    rewards = np.array([1.0, 2, 3])
    values = np.array([0.5, 1, 2, 999])
    terminals = np.array([False, False, True])
    for array in (rewards, values, terminals):
        array.flags.writeable = False
    compute_gae(rewards, values, terminals)
    compute_event_credit(rewards, values, terminals, 1)
    compute_segment_credit(rewards, values, terminals, 0, 3)
    np.testing.assert_array_equal(rewards, [1, 2, 3])
    np.testing.assert_array_equal(values, [0.5, 1, 2, 999])
    np.testing.assert_array_equal(terminals, [False, False, True])


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True, "three", None])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(ValueError, match="horizon"):
        compute_gae([1], [0, 0], [True], horizon=horizon)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": -1},
        {"lam": 1.01},
        {"gamma": np.nan},
    ],
)
def test_invalid_discount_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        compute_gae([1], [0, 0], [True], **kwargs)


@pytest.mark.parametrize(
    "rewards,values,terminated",
    [
        ([1], [0], [True]),
        ([1], [0, 0], [0, 1]),
        ([1], [0, 0], [2]),
        ([np.nan], [0, 0], [True]),
        ([[1]], [0, 0], [True]),
    ],
)
def test_invalid_rollout_arrays_rejected(rewards, values, terminated):
    with pytest.raises(ValueError):
        compute_gae(rewards, values, terminated)


def test_saved_synthetic_trajectory_can_be_evaluated_under_every_horizon(tmp_path):
    path = tmp_path / "trajectory.npz"
    np.savez(path, rewards=[1, 2, 3, 4], values=np.zeros(5), terminated=[0, 0, 0, 1])
    with np.load(path) as trajectory:
        advantages = {
            str(horizon): compute_gae(**trajectory, horizon=horizon).advantages
            for horizon in (1, 3, 16, "full")
        }
    assert all(np.isfinite(array).all() for array in advantages.values())
    np.testing.assert_array_equal(advantages["16"], advantages["full"])

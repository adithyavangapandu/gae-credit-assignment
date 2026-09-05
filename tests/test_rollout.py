from dataclasses import replace

import numpy as np
import pytest
import torch

from gae_credit.algorithms.networks import Actor, Critic
from gae_credit.algorithms.ppo import PPO, prepare_batch
from gae_credit.algorithms.rollout import collect_rollout
from gae_credit.config import AlgorithmConfig, EnvironmentConfig, RewardConfig
from gae_credit.envs.pendulum import PendulumEnv
from gae_credit.envs.rewards import make_reward


def collect(kind="dense", deterministic=False):
    env = PendulumEnv(EnvironmentConfig(max_steps=20))
    reward = make_reward(RewardConfig(kind, 8 if kind == "delayed" else 1))
    actor, critic = Actor(4), Critic(4)
    episodes = collect_rollout(
        env, reward, actor, critic, [17, 29], torch.Generator().manual_seed(123), deterministic
    )
    return actor, critic, episodes


@pytest.mark.parametrize("kind", ["dense", "delayed", "sparse"])
def test_complete_rollouts_reproducible_and_keep_both_rewards(kind):
    _, _, episodes = collect(kind)
    _, _, repeated = collect(kind)
    for ep, again in zip(episodes, repeated):
        for field in (
            "observations",
            "actions",
            "pre_tanh_actions",
            "rewards",
            "base_dense_rewards",
            "values",
            "terminated",
            "old_log_probs",
        ):
            np.testing.assert_array_equal(getattr(ep, field), getattr(again, field))
        assert ep.observations.shape == (20, 4)
        assert ep.actions.shape == (20, 1)
        assert ep.values.shape == (21,)
        assert ep.values[-1] == 0
        assert ep.terminated.sum() == 1 and ep.terminated[-1]
        assert ep.metrics == again.metrics
        assert ep.metrics["stable_success"] == (ep.metrics["longest_upright_streak"] >= 10)
        if kind == "dense":
            np.testing.assert_array_equal(ep.rewards, ep.base_dense_rewards)
        elif kind == "delayed":
            weights = 0.995 ** np.arange(20)
            assert weights @ ep.rewards == pytest.approx(weights @ ep.base_dense_rewards)
        else:
            assert np.isin(ep.rewards, [0, 1]).all()


def test_initial_ppo_ratio_is_one_and_update_changes_both_networks():
    actor, critic, episodes = collect()
    batch = prepare_batch(episodes, 0.995, 0.95, 3, 3)
    ratio = (actor.log_prob(batch.observations, batch.pre_tanh_actions) - batch.old_log_probs).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-6, rtol=1e-6)
    old_probs = batch.old_log_probs.clone()
    targets = batch.value_targets.clone()
    old_actor = [p.clone() for p in actor.parameters()]
    old_critic = [p.clone() for p in critic.parameters()]
    optimizer = PPO(actor, critic, AlgorithmConfig(epochs=2, minibatch_size=20))
    metrics = optimizer.update(batch, torch.Generator().manual_seed(19))
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(x, y) for x, y in zip(old_actor, actor.parameters()))
    assert any(not torch.equal(x, y) for x, y in zip(old_critic, critic.parameters()))
    torch.testing.assert_close(batch.old_log_probs, old_probs, atol=0, rtol=0)
    torch.testing.assert_close(batch.value_targets, targets, atol=0, rtol=0)


def test_preparing_batch_cannot_mix_episodes_and_actor_critic_horizons():
    _, _, episodes = collect()
    combined = prepare_batch(episodes, 0.995, 0.95, 3, "full")
    for i, ep in enumerate(episodes):
        isolated = prepare_batch([ep], 0.995, 0.95, 3, "full")
        torch.testing.assert_close(combined.advantages[i * 20 : (i + 1) * 20], isolated.advantages)
        torch.testing.assert_close(
            combined.value_targets[i * 20 : (i + 1) * 20], isolated.value_targets
        )


def test_target_kl_stops_remaining_epochs():
    actor, critic, episodes = collect()
    batch = prepare_batch(episodes, 0.995, 0.95, 3, 3)
    config = replace(AlgorithmConfig(), epochs=10, minibatch_size=40, target_kl=1e-12)
    metrics = PPO(actor, critic, config).update(batch, torch.Generator().manual_seed(0))
    assert metrics["epochs_completed"] == 1

import math

import pytest
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from gae_credit.algorithms.networks import Actor, Critic


def test_actions_log_probs_and_deterministic_mean():
    actor = Actor(4)
    obs = torch.zeros(1000, 4)
    action, log_prob, latent = actor.sample(obs, torch.Generator().manual_seed(3))
    assert (action.abs() <= 2).all()
    assert torch.isfinite(log_prob).all()
    torch.testing.assert_close(actor.log_prob(obs, latent), log_prob)
    deterministic, _, _ = actor.sample(obs, torch.Generator(), deterministic=True)
    torch.testing.assert_close(deterministic, 2 * actor(obs).tanh())


def test_squashed_log_prob_matches_change_of_variables():
    actor = Actor(4, init_log_std=math.log(0.7))
    obs = torch.tensor([[1.0, 0, 0.5, 1]])
    latent = torch.tensor([[0.25]])
    distribution = TransformedDistribution(
        Normal(actor(obs), actor.std), [TanhTransform(), AffineTransform(0, 2)]
    )
    expected = distribution.log_prob(2 * latent.tanh()).sum(-1)
    torch.testing.assert_close(actor.log_prob(obs, latent), expected)


def test_saturated_actions_have_finite_corrected_log_prob_and_gradients():
    actor = Actor(4)
    obs = torch.zeros(2, 4)
    latent = torch.tensor([[30.0], [-30.0]])
    lp = actor.log_prob(obs, latent)
    assert torch.isfinite(lp).all()
    lp.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters())


@pytest.mark.parametrize("model", [Actor, Critic])
def test_model_seed_reproducibility_without_changing_global_rng(model):
    before = torch.random.get_rng_state().clone()
    a, b, c = model(4, seed=7), model(4, seed=7), model(4, seed=8)
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), c.parameters()))


def test_transformed_entropy_is_differentiable_and_finite():
    actor = Actor(4)
    entropy = actor.entropy(torch.zeros(32, 4), torch.Generator().manual_seed(0)).mean()
    entropy.backward()
    assert torch.isfinite(entropy)
    assert torch.isfinite(actor.log_std.grad).all()

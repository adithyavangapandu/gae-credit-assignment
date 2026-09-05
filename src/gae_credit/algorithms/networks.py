"""A bounded Gaussian actor and a separate finite-horizon value network."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _mlp(observation_dim: int, hidden_size: int, output_gain: float) -> nn.Sequential:
    network = nn.Sequential(
        nn.Linear(observation_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
    )
    for module in network:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(network[-1].weight, gain=output_gain)
    return network


class Actor(nn.Module):
    """Tanh-squashed Gaussian, with a learned state-independent base log std.

    Log probabilities consume the saved pre-tanh sample. Reconstructing it with
    ``atanh(action / max_torque)`` loses information when float32 tanh saturates.
    Initialization restores the caller's RNG state; sampling uses an explicit
    generator so environment resets never alter the action stream.
    """

    def __init__(
        self,
        observation_dim: int,
        hidden_size: int = 64,
        max_torque: float = 2.0,
        init_log_std: float = -0.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if observation_dim < 1 or hidden_size < 1:
            raise ValueError("Network dimensions must be positive")
        if not math.isfinite(max_torque) or max_torque <= 0:
            raise ValueError("max_torque must be finite and positive")
        if not math.isfinite(init_log_std):
            raise ValueError("init_log_std must be finite")
        self.max_torque = float(max_torque)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.mean_network = _mlp(observation_dim, hidden_size, output_gain=0.01)
        self.log_std = nn.Parameter(torch.tensor([init_log_std], dtype=torch.float32))

    def forward(self, observations: Tensor) -> Tensor:
        """Return the Gaussian mean before the squash transformation."""
        return self.mean_network(observations)

    @property
    def std(self) -> Tensor:
        """Base Gaussian std; the bounded action distribution is not Gaussian."""
        return self.log_std.clamp(-20.0, 2.0).exp()

    def deterministic_action(self, observations: Tensor) -> Tensor:
        return self.max_torque * self(observations).tanh()

    def log_prob(self, observations: Tensor, pre_tanh_actions: Tensor) -> Tensor:
        mean = self(observations)
        log_std = self.log_std.clamp(-20.0, 2.0)
        gaussian_log_prob = (
            -0.5 * ((pre_tanh_actions - mean) / log_std.exp()).square()
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        )
        # Stable log(1 - tanh(x)^2), including at saturated float32 actions.
        log_jacobian = math.log(self.max_torque) + 2.0 * (
            math.log(2.0) - pre_tanh_actions - F.softplus(-2.0 * pre_tanh_actions)
        )
        return (gaussian_log_prob - log_jacobian).sum(dim=-1)

    def sample(
        self,
        observations: Tensor,
        generator: torch.Generator,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean = self(observations)
        if deterministic:
            pre_tanh_actions = mean
        else:
            noise = torch.randn(
                mean.shape, dtype=mean.dtype, device=mean.device, generator=generator
            )
            pre_tanh_actions = mean + self.std * noise
        actions = self.max_torque * pre_tanh_actions.tanh()
        return actions, self.log_prob(observations, pre_tanh_actions), pre_tanh_actions

    def entropy(self, observations: Tensor, generator: torch.Generator) -> Tensor:
        """One reparameterized Monte Carlo sample of transformed differential entropy.

        This includes the scale and squash Jacobian, unlike Gaussian entropy.
        Its differentiable sample allows an entropy bonus for the actual policy.
        """
        _, log_prob, _ = self.sample(observations, generator)
        return -log_prob


class Critic(nn.Module):
    """Independent two-layer tanh state-value network."""

    def __init__(self, observation_dim: int, hidden_size: int = 64, seed: int = 1) -> None:
        super().__init__()
        if observation_dim < 1 or hidden_size < 1:
            raise ValueError("Network dimensions must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.value_network = _mlp(observation_dim, hidden_size, output_gain=1.0)

    def forward(self, observations: Tensor) -> Tensor:
        return self.value_network(observations).squeeze(-1)

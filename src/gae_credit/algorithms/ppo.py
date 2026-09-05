"""Clipped PPO using fixed advantages and value targets from complete episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from gae_credit.estimators.gae import compute_gae

from .networks import Actor, Critic
from .rollout import EpisodeRollout

if TYPE_CHECKING:
    from gae_credit.config import AlgorithmConfig


@dataclass(frozen=True)
class PPOBatch:
    observations: Tensor
    actions: Tensor
    pre_tanh_actions: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    value_targets: Tensor
    td_errors: Tensor
    old_values: Tensor

    def __len__(self) -> int:
        return len(self.observations)


def prepare_batch(
    episodes: list[EpisodeRollout],
    gamma: float,
    lam: float,
    actor_horizon: int | str,
    critic_horizon: int | str,
) -> PPOBatch:
    """Compute actor and critic estimators independently within each episode."""
    if not episodes:
        raise ValueError("At least one complete episode is required")
    actor_results = [
        compute_gae(ep.rewards, ep.values, ep.terminated, gamma, lam, actor_horizon)
        for ep in episodes
    ]
    critic_results = actor_results
    if critic_horizon != actor_horizon:
        critic_results = [
            compute_gae(ep.rewards, ep.values, ep.terminated, gamma, lam, critic_horizon)
            for ep in episodes
        ]

    def tensor(arrays: list[np.ndarray]) -> Tensor:
        return torch.tensor(np.concatenate(arrays), dtype=torch.float32)

    return PPOBatch(
        observations=tensor([ep.observations for ep in episodes]),
        actions=tensor([ep.actions for ep in episodes]),
        pre_tanh_actions=tensor([ep.pre_tanh_actions for ep in episodes]),
        old_log_probs=tensor([ep.old_log_probs for ep in episodes]),
        advantages=tensor([result.advantages for result in actor_results]),
        value_targets=tensor([result.value_targets for result in critic_results]),
        td_errors=tensor([result.td_errors for result in actor_results]),
        old_values=tensor([ep.values[:-1] for ep in episodes]),
    )


class PPO:
    """Separate actor/critic Adam optimizers and an explicit update RNG stream."""

    def __init__(self, actor: Actor, critic: Critic, config: AlgorithmConfig) -> None:
        self.actor = actor
        self.critic = critic
        self.config = config
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    def update(self, batch: PPOBatch, generator: torch.Generator) -> dict[str, float]:
        if len(batch) == 0:
            raise ValueError("Cannot update on an empty batch")
        device = next(self.actor.parameters()).device
        observations = batch.observations.to(device)
        pre_tanh_actions = batch.pre_tanh_actions.to(device)
        old_log_probs = batch.old_log_probs.to(device)
        targets = batch.value_targets.to(device)
        advantages = batch.advantages.to(device)
        for name, array in (
            ("observations", observations),
            ("pre_tanh_actions", pre_tanh_actions),
            ("old_log_probs", old_log_probs),
            ("targets", targets),
            ("advantages", advantages),
        ):
            if not torch.isfinite(array).all():
                raise ValueError(f"Non-finite {name} in PPO batch")
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        totals = dict.fromkeys(
            ("actor_loss", "critic_loss", "entropy", "actor_grad_norm", "critic_grad_norm"),
            0.0,
        )
        samples_seen = 0
        epochs_completed = 0
        for _ in range(self.config.epochs):
            indices = torch.randperm(len(batch), generator=generator, device=device)
            for start in range(0, len(batch), self.config.minibatch_size):
                idx = indices[start : start + self.config.minibatch_size]
                log_probs = self.actor.log_prob(observations[idx], pre_tanh_actions[idx])
                ratios = (log_probs - old_log_probs[idx]).exp()
                unclipped = ratios * advantages[idx]
                clipped = (
                    ratios.clamp(1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
                    * advantages[idx]
                )
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                entropy = self.actor.entropy(observations[idx], generator).mean()
                actor_objective = actor_loss - self.config.entropy_coef * entropy
                critic_loss = (self.critic(observations[idx]) - targets[idx]).square().mean()
                if not torch.isfinite(actor_objective) or not torch.isfinite(critic_loss):
                    raise FloatingPointError("Non-finite PPO loss")

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_objective.backward()
                actor_norm = clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
                )
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coef * critic_loss).backward()
                critic_norm = clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
                )
                self.critic_optimizer.step()
                count = len(idx)
                samples_seen += count
                for key, value in (
                    ("actor_loss", actor_loss),
                    ("critic_loss", critic_loss),
                    ("entropy", entropy),
                    ("actor_grad_norm", actor_norm),
                    ("critic_grad_norm", critic_norm),
                ):
                    totals[key] += count * float(value.detach().item())
            epochs_completed += 1
            with torch.no_grad():
                log_ratio = self.actor.log_prob(observations, pre_tanh_actions) - old_log_probs
                approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
            if self.config.target_kl is not None and approx_kl.item() > self.config.target_kl:
                break

        with torch.no_grad():
            clip_fraction = (
                ((log_ratio.exp() - 1.0).abs() > self.config.clip_epsilon).float().mean()
            )
            target_variance = batch.value_targets.var(unbiased=False).item()
            explained_variance = 0.0
            if target_variance > 1e-12:
                explained_variance = 1.0 - (
                    (batch.value_targets - batch.old_values).var(unbiased=False).item()
                    / target_variance
                )
            metrics = {key: value / samples_seen for key, value in totals.items()}
            metrics.update(
                action_std=float(self.actor.std.mean().item()),
                approx_kl=float(approx_kl.item()),
                clip_fraction=float(clip_fraction.item()),
                advantage_mean=float(batch.advantages.mean().item()),
                advantage_std=float(batch.advantages.std(unbiased=False).item()),
                advantage_min=float(batch.advantages.min().item()),
                advantage_max=float(batch.advantages.max().item()),
                td_error_mean=float(batch.td_errors.mean().item()),
                td_error_std=float(batch.td_errors.std(unbiased=False).item()),
                target_mean=float(batch.value_targets.mean().item()),
                target_std=float(batch.value_targets.std(unbiased=False).item()),
                prediction_mean=float(batch.old_values.mean().item()),
                prediction_std=float(batch.old_values.std(unbiased=False).item()),
                explained_variance=float(explained_variance),
                epochs_completed=float(epochs_completed),
            )
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Non-finite PPO metrics")
        return metrics

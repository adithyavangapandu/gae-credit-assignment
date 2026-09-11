"""Vanilla policy gradient with one on-policy actor step and fitted critic."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .ppo import PPOBatch


class VPG:
    """One full-batch policy-gradient step followed by configurable critic fitting."""

    def __init__(self, actor, critic, config):
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
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        log_probs = self.actor.log_prob(observations, pre_tanh_actions)
        actor_loss = -(log_probs * advantages).mean()
        entropy = self.actor.entropy(observations, generator).mean()
        actor_objective = actor_loss - self.config.entropy_coef * entropy
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_objective.backward()
        actor_norm = clip_grad_norm_(
            self.actor.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
        )
        self.actor_optimizer.step()

        critic_loss_total = 0.0
        critic_norm_total = 0.0
        samples_seen = 0
        for _ in range(self.config.epochs):
            indices = torch.randperm(len(batch), generator=generator, device=device)
            for start in range(0, len(batch), self.config.minibatch_size):
                selected = indices[start : start + self.config.minibatch_size]
                critic_loss = (
                    (self.critic(observations[selected]) - targets[selected]).square().mean()
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coef * critic_loss).backward()
                critic_norm = clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
                )
                self.critic_optimizer.step()
                count = len(selected)
                samples_seen += count
                critic_loss_total += count * float(critic_loss.detach())
                critic_norm_total += count * float(critic_norm.detach())

        with torch.no_grad():
            new_log_probs = self.actor.log_prob(observations, pre_tanh_actions)
            log_ratio = new_log_probs - old_log_probs
            approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
            target_variance = batch.value_targets.var(unbiased=False).item()
            explained_variance = 0.0
            if target_variance > 1e-12:
                explained_variance = 1.0 - (
                    (batch.value_targets - batch.old_values).var(unbiased=False).item()
                    / target_variance
                )
            metrics = {
                "actor_loss": float(actor_loss.detach()),
                "critic_loss": critic_loss_total / samples_seen,
                "entropy": float(entropy.detach()),
                "action_std": float(self.actor.std.mean()),
                "actor_grad_norm": float(actor_norm.detach()),
                "critic_grad_norm": critic_norm_total / samples_seen,
                "approx_kl": float(approx_kl),
                "clip_fraction": 0.0,
                "advantage_mean": float(batch.advantages.mean()),
                "advantage_std": float(batch.advantages.std(unbiased=False)),
                "advantage_min": float(batch.advantages.min()),
                "advantage_max": float(batch.advantages.max()),
                "td_error_mean": float(batch.td_errors.mean()),
                "td_error_std": float(batch.td_errors.std(unbiased=False)),
                "target_mean": float(batch.value_targets.mean()),
                "target_std": float(batch.value_targets.std(unbiased=False)),
                "prediction_mean": float(batch.old_values.mean()),
                "prediction_std": float(batch.old_values.std(unbiased=False)),
                "explained_variance": float(explained_variance),
                "epochs_completed": float(self.config.epochs),
            }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Non-finite VPG metrics")
        return metrics

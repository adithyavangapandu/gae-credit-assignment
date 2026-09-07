"""Complete-episode collection, independent of the policy optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from numpy.typing import NDArray

from .networks import Actor, Critic


@dataclass(frozen=True)
class EpisodeRollout:
    observations: NDArray[np.float32]
    actions: NDArray[np.float32]
    pre_tanh_actions: NDArray[np.float32]
    rewards: NDArray[np.float64]
    base_dense_rewards: NDArray[np.float64]
    values: NDArray[np.float64]
    terminated: NDArray[np.bool_]
    old_log_probs: NDArray[np.float32]
    metrics: dict[str, Any]
    reward_diagnostics: list[dict[str, Any]]


@torch.no_grad()
def collect_rollout(
    env: Any,
    reward: Any,
    actor: Actor,
    critic: Critic,
    episode_seeds: Iterable[int],
    action_generator: torch.Generator,
    deterministic: bool = False,
) -> list[EpisodeRollout]:
    """Collect full episodes without computing advantages or reseeding torch.

    Real terminal transitions bootstrap zero. A generic truncated transition
    retains its last value prediction; the study environment uses real terminal
    transitions at its finite time horizon. Arrays from distinct resets are kept
    separate so no estimator can accidentally propagate credit across episodes.
    """
    device = next(actor.parameters()).device
    episodes = []
    for seed in episode_seeds:
        observation, _ = env.reset(seed=int(seed))
        reward.reset()
        observations, actions, pre_tanh_actions, old_log_probs = [], [], [], []
        rewards, base_rewards, values, terminals, diagnostics = [], [], [], [], []
        angle_cost = velocity_cost = torque_cost = torque_energy = 0.0
        upright_count = upright_streak = longest_streak = 0
        first_upright_timestep = None
        while True:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            action, log_prob, pre_tanh_action = actor.sample(
                observation_tensor, action_generator, deterministic=deterministic
            )
            action_array = action.cpu().numpy().copy()
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            actions.append(action_array)
            pre_tanh_actions.append(pre_tanh_action.cpu().numpy().copy())
            old_log_probs.append(float(log_prob.item()))
            values.append(float(critic(observation_tensor).item()))
            next_observation, _, terminated, truncated, info = env.step(action_array)
            train_reward, reward_info = reward.transform(
                base_dense_reward=float(info["base_dense_reward"]),
                theta=float(info["theta"]),
                theta_dot=float(info["theta_dot"]),
                timestep=int(info["timestep"]),
                terminated=bool(terminated or truncated),
            )
            rewards.append(float(train_reward))
            base_rewards.append(float(info["base_dense_reward"]))
            terminals.append(bool(terminated))
            diagnostics.append(dict(reward_info))
            angle_cost += float(info["angle_cost"])
            velocity_cost += float(info["velocity_cost"])
            torque_cost += float(info["torque_cost"])
            torque_energy += float(info["torque"]) ** 2 * env.config.dt
            upright = bool(reward_info.get("upright", info["upright_indicator"]))
            if upright:
                upright_count += 1
                upright_streak += 1
                longest_streak = max(longest_streak, upright_streak)
                if first_upright_timestep is None:
                    first_upright_timestep = int(info["timestep"])
            else:
                upright_streak = 0
            observation = next_observation
            if terminated or truncated:
                bootstrap = 0.0
                if not terminated:
                    final_observation = torch.as_tensor(
                        observation, dtype=torch.float32, device=device
                    )
                    bootstrap = float(critic(final_observation).item())
                values.append(bootstrap)
                break
        length = len(rewards)
        metrics = {
            "episode_seed": int(seed),
            "train_return": float(np.sum(rewards)),
            "base_dense_return": float(np.sum(base_rewards)),
            "length": length,
            "first_upright_timestep": first_upright_timestep,
            "stable_success": longest_streak >= 10,
            "upright_fraction": upright_count / length,
            "longest_upright_streak": longest_streak,
            "angle_cost": angle_cost,
            "velocity_cost": velocity_cost,
            "torque_cost": torque_cost,
            "torque_energy": torque_energy,
        }
        episodes.append(
            EpisodeRollout(
                observations=np.asarray(observations, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                pre_tanh_actions=np.asarray(pre_tanh_actions, dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float64),
                base_dense_rewards=np.asarray(base_rewards, dtype=np.float64),
                values=np.asarray(values, dtype=np.float64),
                terminated=np.asarray(terminals, dtype=np.bool_),
                old_log_probs=np.asarray(old_log_probs, dtype=np.float32),
                metrics=metrics,
                reward_diagnostics=diagnostics,
            )
        )
    return episodes

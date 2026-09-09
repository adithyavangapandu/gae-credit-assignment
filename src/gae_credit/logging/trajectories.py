"""Typed per-step diagnostic trajectories from fixed evaluation episodes."""

from __future__ import annotations

import pyarrow as pa

TRAJECTORY_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("checkpoint_env_steps", pa.int64(), nullable=False),
        pa.field("trajectory_id", pa.int64(), nullable=False),
        pa.field("episode_seed", pa.int64(), nullable=False),
        pa.field("timestep", pa.int64(), nullable=False),
        *[
            pa.field(name, pa.float64(), nullable=False)
            for name in (
                "observation_cos",
                "observation_sin",
                "observation_velocity",
                "observation_time_remaining",
                "next_observation_cos",
                "next_observation_sin",
                "next_observation_velocity",
                "next_observation_time_remaining",
                "action",
                "pre_tanh_action",
                "value",
                "training_reward",
                "base_dense_reward",
                "theta",
                "theta_dot",
                "torque",
            )
        ],
        pa.field("upright", pa.bool_(), nullable=False),
        pa.field("terminated", pa.bool_(), nullable=False),
        pa.field("emitted_payout", pa.bool_(), nullable=False),
        pa.field("block_id", pa.int64(), nullable=True),
        pa.field("block_position", pa.int64(), nullable=True),
        pa.field("payout_count", pa.int64(), nullable=True),
    ]
)


def trajectory_rows(run_id, checkpoint_env_steps, episodes):
    rows = []
    for trajectory_id, episode in enumerate(episodes):
        for index in range(len(episode.rewards)):
            reward_info = episode.reward_diagnostics[index]
            observation = episode.observations[index]
            next_observation = episode.next_observations[index]
            rows.append(
                {
                    "run_id": run_id,
                    "checkpoint_env_steps": checkpoint_env_steps,
                    "trajectory_id": trajectory_id,
                    "episode_seed": episode.metrics["episode_seed"],
                    "timestep": index + 1,
                    "observation_cos": float(observation[0]),
                    "observation_sin": float(observation[1]),
                    "observation_velocity": float(observation[2]),
                    "observation_time_remaining": float(observation[3]),
                    "next_observation_cos": float(next_observation[0]),
                    "next_observation_sin": float(next_observation[1]),
                    "next_observation_velocity": float(next_observation[2]),
                    "next_observation_time_remaining": float(next_observation[3]),
                    "action": float(episode.actions[index, 0]),
                    "pre_tanh_action": float(episode.pre_tanh_actions[index, 0]),
                    "value": float(episode.values[index]),
                    "training_reward": float(episode.rewards[index]),
                    "base_dense_reward": float(episode.base_dense_rewards[index]),
                    "theta": float(episode.thetas[index]),
                    "theta_dot": float(episode.theta_dots[index]),
                    "torque": float(episode.torques[index]),
                    "upright": bool(episode.uprights[index]),
                    "terminated": bool(episode.terminated[index]),
                    "emitted_payout": bool(reward_info.get("emitted_payout", False)),
                    "block_id": reward_info.get("block_id"),
                    "block_position": reward_info.get("block_position"),
                    "payout_count": reward_info.get("payout_count"),
                }
            )
    return rows

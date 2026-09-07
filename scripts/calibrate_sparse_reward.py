"""Measure whether the frozen sparse-reward region is discoverable before pilots."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gae_credit.algorithms.networks import Actor
from gae_credit.config import derive_seeds
from gae_credit.envs.pendulum import PendulumEnv, is_upright
from gae_credit.pilot import load_pilot_spec

SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.int64(), nullable=False),
        pa.field("episode_seed", pa.int64(), nullable=False),
        pa.field("entered_upright", pa.bool_(), nullable=False),
        pa.field("first_entry_timestep", pa.int64(), nullable=True),
        pa.field("upright_timesteps", pa.int64(), nullable=False),
        pa.field("longest_upright_streak", pa.int64(), nullable=False),
        pa.field("stable_success", pa.bool_(), nullable=False),
    ]
)


@torch.no_grad()
def calibrate(
    config,
    episodes: int,
    angle_threshold: float,
    velocity_threshold: float,
    *,
    adjustment_already_used: bool = False,
):
    if not 2_000 <= episodes <= 5_000:
        raise ValueError("Calibration must use 2,000–5,000 episodes")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    streams = derive_seeds(config)
    model_seed = int(np.random.SeedSequence(streams["model"]).generate_state(2)[0])
    actor = Actor(
        4,
        config.optimizer.hidden_size,
        config.environment.max_torque,
        config.optimizer.init_log_std,
        seed=model_seed,
    )
    action_rng = torch.Generator().manual_seed(streams["action"])
    env_rng = np.random.default_rng(streams["environment"])
    episode_seeds = env_rng.integers(0, 2**32, size=episodes, dtype=np.uint32)
    env = PendulumEnv(config.environment)
    rows = []
    try:
        for episode_index, episode_seed in enumerate(episode_seeds):
            observation, _ = env.reset(seed=int(episode_seed))
            first_entry = None
            upright_timesteps = streak = longest = 0
            while True:
                tensor = torch.as_tensor(observation, dtype=torch.float32)
                action, _, _ = actor.sample(tensor, action_rng)
                observation, _, terminated, truncated, info = env.step(action.numpy())
                upright = is_upright(
                    info["theta"],
                    info["theta_dot"],
                    angle_threshold=angle_threshold,
                    velocity_threshold=velocity_threshold,
                )
                if upright:
                    upright_timesteps += 1
                    streak += 1
                    longest = max(longest, streak)
                    if first_entry is None:
                        first_entry = int(info["timestep"])
                else:
                    streak = 0
                if terminated or truncated:
                    break
            rows.append(
                {
                    "episode_index": episode_index,
                    "episode_seed": int(episode_seed),
                    "entered_upright": first_entry is not None,
                    "first_entry_timestep": first_entry,
                    "upright_timesteps": upright_timesteps,
                    "longest_upright_streak": longest,
                    "stable_success": longest >= 10,
                }
            )
    finally:
        env.close()
    entry_count = sum(row["entered_upright"] for row in rows)
    rate = entry_count / episodes
    decision = "freeze_after_single_adjustment" if adjustment_already_used else "keep"
    if not adjustment_already_used:
        if rate < 0.005:
            decision = "widen_once"
        elif rate > 0.10:
            decision = "tighten_once"
    summary = {
        "purpose": "exploration feasibility only; excluded from hypothesis tests",
        "policy": "untrained stochastic PPO actor",
        "training_seed": config.seed,
        "config_hash": config.config_hash(),
        "episodes": episodes,
        "episode_length": config.environment.max_steps,
        "angle_threshold_radians": angle_threshold,
        "velocity_threshold": velocity_threshold,
        "entry_count": entry_count,
        "entry_fraction": rate,
        "stable_success_count": sum(row["stable_success"] for row in rows),
        "stable_success_fraction": sum(row["stable_success"] for row in rows) / episodes,
        "first_entry_timestep_median": (
            float(
                np.median([row["first_entry_timestep"] for row in rows if row["entered_upright"]])
            )
            if entry_count
            else None
        ),
        "upright_timesteps_successful_median": (
            float(np.median([row["upright_timesteps"] for row in rows if row["entered_upright"]]))
            if entry_count
            else None
        ),
        "longest_upright_streak_max": max(row["longest_upright_streak"] for row in rows),
        "preregistered_decision": decision,
        "adjustment_already_used": adjustment_already_used,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot/study_v1.yaml")
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--angle-threshold", type=float, default=0.262)
    parser.add_argument("--velocity-threshold", type=float, default=1.0)
    parser.add_argument("--adjustment-already-used", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    _, config = load_pilot_spec(args.config)
    rows, summary = calibrate(
        config,
        args.episodes,
        args.angle_threshold,
        args.velocity_threshold,
        adjustment_already_used=args.adjustment_already_used,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMA),
        args.output_dir / "sparse_calibration.parquet",
        compression="zstd",
    )
    (args.output_dir / "sparse_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

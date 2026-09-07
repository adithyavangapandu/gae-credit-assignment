"""Deterministic excluded-pilot configuration and run-matrix contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from gae_credit.cloud.config import image_digest, run_id_for
from gae_credit.config import RewardConfig, config_from_dict

MATRIX_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.int64(), nullable=False),
        pa.field("phase", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("algorithm", pa.string(), nullable=False),
        pa.field("reward_kind", pa.string(), nullable=False),
        pa.field("delay_block_size", pa.int64(), nullable=False),
        pa.field("actor_horizon", pa.string(), nullable=False),
        pa.field("critic_horizon", pa.string(), nullable=False),
        pa.field("gamma", pa.float64(), nullable=False),
        pa.field("lambda", pa.float64(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("total_env_steps", pa.int64(), nullable=False),
        pa.field("config_hash", pa.string(), nullable=False),
        pa.field("image_digest", pa.string(), nullable=False),
        pa.field("expected_gcs_prefix", pa.string(), nullable=False),
    ]
)


def load_pilot_spec(path: str | Path):
    raw = yaml.safe_load(Path(path).read_text())
    if set(raw) != {"study_version", "purpose", "sparse_thresholds", "matrix", "base_config"}:
        raise ValueError("Pilot study file has unexpected fields")
    if raw["study_version"] != "study_v1" or "excluded" not in raw["purpose"]:
        raise ValueError("Pilot must be explicitly versioned and excluded from inference")
    base = config_from_dict(raw["base_config"])
    if base.training.total_env_steps != 250_000:
        raise ValueError("Pilot budget must be 250,000 environment steps")
    return raw, base


def condition_config(spec: dict, base, reward: dict, horizon, seed: int):
    thresholds = spec["sparse_thresholds"]
    reward_config = RewardConfig(kind=reward["kind"], delay_block_size=reward["delay_block_size"])
    if reward["kind"] == "sparse":
        reward_config = replace(
            reward_config,
            upright_angle_threshold=float(thresholds["angle_radians"]),
            upright_velocity_threshold=float(thresholds["velocity"]),
        )
    estimator = replace(
        base.estimator, horizon=horizon, actor_horizon=horizon, critic_horizon=horizon
    )
    return replace(base, seed=int(seed), reward=reward_config, estimator=estimator)


def generate_matrix(spec: dict, base, gcp, image_uri: str) -> list[dict]:
    image_digest(image_uri)
    matrix = spec["matrix"]
    if matrix["seeds"] != [100, 101, 102] or matrix["horizons"] != [3, "full"]:
        raise ValueError("Pilot seeds/horizons differ from the frozen design")
    if len(matrix["rewards"]) != 4:
        raise ValueError("Pilot requires four reward conditions")
    rows = []
    for reward in matrix["rewards"]:
        for horizon in matrix["horizons"]:
            for seed in matrix["seeds"]:
                config = condition_config(spec, base, reward, horizon, seed)
                value = "full" if horizon == "full" else str(horizon)
                rows.append(
                    {
                        "phase": "pilot",
                        "run_id": run_id_for(config, "pilot"),
                        "algorithm": config.algorithm,
                        "reward_kind": config.reward.kind,
                        "delay_block_size": config.reward.delay_block_size,
                        "actor_horizon": value,
                        "critic_horizon": value,
                        "gamma": config.estimator.gamma,
                        "lambda": config.estimator.lam,
                        "seed": config.seed,
                        "total_env_steps": config.training.total_env_steps,
                        "config_hash": config.config_hash(),
                        "image_digest": image_uri,
                        "expected_gcs_prefix": gcp.run_prefix(config, "pilot"),
                    }
                )
    rows.sort(
        key=lambda row: (
            row["reward_kind"],
            row["delay_block_size"],
            row["actor_horizon"],
            row["seed"],
        )
    )
    for task_id, row in enumerate(rows):
        row["task_id"] = task_id
    if len(rows) != 24 or len({row["run_id"] for row in rows}) != 24:
        raise ValueError("Pilot matrix must contain 24 unique runs")
    return rows


def write_matrix(rows: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=MATRIX_SCHEMA), path, compression="zstd")


def read_matrix(path: str | Path) -> list[dict]:
    table = pq.read_table(path)
    if not table.schema.equals(MATRIX_SCHEMA, check_metadata=True):
        raise ValueError("Pilot matrix schema mismatch")
    rows = table.to_pylist()
    if [row["task_id"] for row in rows] != list(range(len(rows))):
        raise ValueError("Pilot task IDs must be consecutive")
    if len(rows) != 24 or len({row["run_id"] for row in rows}) != 24:
        raise ValueError("Pilot matrix must contain 24 unique runs")
    return rows


def config_for_row(spec: dict, base, row: dict):
    horizon = "full" if row["actor_horizon"] == "full" else int(row["actor_horizon"])
    config = condition_config(
        spec,
        base,
        {"kind": row["reward_kind"], "delay_block_size": row["delay_block_size"]},
        horizon,
        row["seed"],
    )
    if config.config_hash() != row["config_hash"] or run_id_for(config, "pilot") != row["run_id"]:
        raise ValueError(f"Matrix identity mismatch for task {row['task_id']}")
    return config

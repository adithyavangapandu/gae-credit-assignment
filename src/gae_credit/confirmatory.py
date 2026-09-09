"""Frozen 160-run confirmatory PPO matrix and checksum contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from gae_credit.cloud.config import image_digest, run_id_for
from gae_credit.config import RewardConfig, config_from_dict

CONFIRMATORY_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.int64(), nullable=False),
        pa.field("phase", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("algorithm", pa.string(), nullable=False),
        pa.field("reward_type", pa.string(), nullable=False),
        pa.field("reward_delay", pa.int64(), nullable=False),
        pa.field("gae_horizon", pa.string(), nullable=False),
        pa.field("gamma", pa.float64(), nullable=False),
        pa.field("gae_lambda", pa.float64(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("training_steps", pa.int64(), nullable=False),
        pa.field("evaluation_interval", pa.int64(), nullable=False),
        pa.field("git_sha", pa.string(), nullable=False),
        pa.field("image_digest", pa.string(), nullable=False),
        pa.field("config_hash", pa.string(), nullable=False),
        pa.field("gcs_output_uri", pa.string(), nullable=False),
    ]
)


def load_confirmatory_spec(path):
    raw = yaml.safe_load(Path(path).read_text())
    expected = {"study_version", "purpose", "sparse_thresholds", "matrix", "base_config"}
    if set(raw) != expected or raw["study_version"] != "experiment-v1":
        raise ValueError("Confirmatory study file is not frozen experiment-v1")
    base = config_from_dict(raw["base_config"])
    if base.training.total_env_steps != 1_000_000:
        raise ValueError("Confirmatory budget must be one million environment steps")
    if base.logging.diagnostic_trajectories_per_checkpoint != 20:
        raise ValueError("Confirmatory runs require 20 checkpoint trajectories")
    return raw, base


def condition_config(spec, base, reward, horizon, seed):
    reward_config = RewardConfig(kind=reward["kind"], delay_block_size=reward["delay_block_size"])
    if reward["kind"] == "sparse":
        reward_config = replace(
            reward_config,
            upright_angle_threshold=float(spec["sparse_thresholds"]["angle_radians"]),
            upright_velocity_threshold=float(spec["sparse_thresholds"]["velocity"]),
        )
    estimator = replace(
        base.estimator, horizon=horizon, actor_horizon=horizon, critic_horizon=horizon
    )
    return replace(base, seed=int(seed), reward=reward_config, estimator=estimator)


def generate_manifest(spec, base, gcp, image_uri, git_sha):
    image_digest(image_uri)
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise ValueError("git_sha must be a full commit SHA")
    matrix = spec["matrix"]
    if matrix["horizons"] != [1, 3, 16, "full"] or matrix["seeds"] != list(range(10)):
        raise ValueError("Confirmatory horizons or seeds differ from the frozen design")
    rows = []
    for reward in matrix["rewards"]:
        for horizon in matrix["horizons"]:
            for seed in matrix["seeds"]:
                config = condition_config(spec, base, reward, horizon, seed)
                rows.append(
                    {
                        "phase": "confirmatory",
                        "run_id": run_id_for(config, "confirmatory"),
                        "algorithm": "ppo-clip",
                        "reward_type": reward["kind"],
                        "reward_delay": reward["delay_block_size"],
                        "gae_horizon": "full" if horizon == "full" else str(horizon),
                        "gamma": config.estimator.gamma,
                        "gae_lambda": config.estimator.lam,
                        "seed": seed,
                        "training_steps": config.training.total_env_steps,
                        "evaluation_interval": config.evaluation.interval_env_steps,
                        "git_sha": git_sha,
                        "image_digest": image_uri,
                        "config_hash": config.config_hash(),
                        "gcs_output_uri": gcp.run_prefix(config, "confirmatory"),
                    }
                )
    rows.sort(
        key=lambda row: (
            row["reward_type"],
            row["reward_delay"],
            row["gae_horizon"],
            row["seed"],
        )
    )
    for task_id, row in enumerate(rows):
        row["task_id"] = task_id
    validate_rows(rows)
    return rows


def validate_rows(rows):
    if len(rows) != 160:
        raise ValueError("Confirmatory manifest must contain exactly 160 rows")
    for key in ("run_id", "config_hash", "gcs_output_uri"):
        if len({row[key] for row in rows}) != 160:
            raise ValueError(f"Confirmatory manifest has duplicate {key}")
    if {row["seed"] for row in rows} != set(range(10)):
        raise ValueError("Confirmatory manifest contains invalid or pilot seeds")
    cells = {}
    for row in rows:
        cell = (row["reward_type"], row["reward_delay"], row["gae_horizon"])
        cells.setdefault(cell, set()).add(row["seed"])
    if len(cells) != 16 or any(seeds != set(range(10)) for seeds in cells.values()):
        raise ValueError("Every confirmatory cell must contain ten seeds")
    if (
        len({row["git_sha"] for row in rows}) != 1
        or len({row["image_digest"] for row in rows}) != 1
    ):
        raise ValueError("All confirmatory rows must use one code and image identity")


def config_for_row(spec, base, row):
    horizon = "full" if row["gae_horizon"] == "full" else int(row["gae_horizon"])
    config = condition_config(
        spec,
        base,
        {"kind": row["reward_type"], "delay_block_size": row["reward_delay"]},
        horizon,
        row["seed"],
    )
    if (
        config.config_hash() != row["config_hash"]
        or run_id_for(config, "confirmatory") != row["run_id"]
    ):
        raise ValueError(f"Confirmatory identity mismatch for task {row['task_id']}")
    return config


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=CONFIRMATORY_SCHEMA), path, compression="zstd")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256.json").write_text(
        json.dumps({"sha256": digest}, indent=2, sort_keys=True) + "\n"
    )
    return digest


def read_manifest(path):
    path = Path(path)
    expected = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Confirmatory manifest checksum mismatch")
    table = pq.read_table(path)
    if not table.schema.equals(CONFIRMATORY_SCHEMA, check_metadata=True):
        raise ValueError("Confirmatory manifest schema mismatch")
    rows = table.to_pylist()
    validate_rows(rows)
    return rows

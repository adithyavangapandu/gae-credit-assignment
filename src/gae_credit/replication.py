"""Frozen 60-run VPG replication matrix and checksum contract."""

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

REPLICATION_SCHEMA = pa.schema(
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
        pa.field("evaluation_seed", pa.int64(), nullable=False),
        pa.field("git_sha", pa.string(), nullable=False),
        pa.field("image_digest", pa.string(), nullable=False),
        pa.field("config_hash", pa.string(), nullable=False),
        pa.field("gcs_output_uri", pa.string(), nullable=False),
    ]
)


def load_replication_spec(path="configs/replication/vpg_v1.yaml"):
    raw = yaml.safe_load(Path(path).read_text())
    if set(raw) != {"study_version", "purpose", "matrix", "base_config"}:
        raise ValueError("Unexpected VPG replication fields")
    if raw["study_version"] != "vpg-replication-v1":
        raise ValueError("Unknown VPG replication version")
    base = config_from_dict(raw["base_config"])
    if base.algorithm != "vpg" or base.training.total_env_steps != 1_000_000:
        raise ValueError("VPG replication algorithm or budget is not frozen")
    if base.evaluation.seed != 20260905:
        raise ValueError("VPG must reuse the PPO evaluation seed")
    return raw, base


def condition_config(spec, base, reward, horizon, seed):
    reward_config = RewardConfig(kind=reward["kind"], delay_block_size=reward["delay_block_size"])
    if reward["kind"] == "sparse":
        reward_config = replace(
            reward_config,
            upright_angle_threshold=0.17453292519943295,
            upright_velocity_threshold=1.0,
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
    expected_rewards = [
        {"kind": "dense", "delay_block_size": 1},
        {"kind": "delayed", "delay_block_size": 32},
        {"kind": "sparse", "delay_block_size": 1},
    ]
    if (
        matrix["rewards"] != expected_rewards
        or matrix["horizons"] != [3, "full"]
        or matrix["seeds"] != list(range(10))
    ):
        raise ValueError("VPG replication matrix differs from the frozen six-cell design")
    rows = []
    for reward in matrix["rewards"]:
        for horizon in matrix["horizons"]:
            for seed in matrix["seeds"]:
                config = condition_config(spec, base, reward, horizon, seed)
                rows.append(
                    {
                        "phase": "replication",
                        "run_id": run_id_for(config, "replication"),
                        "algorithm": "vpg",
                        "reward_type": reward["kind"],
                        "reward_delay": reward["delay_block_size"],
                        "gae_horizon": str(horizon),
                        "gamma": config.estimator.gamma,
                        "gae_lambda": config.estimator.lam,
                        "seed": seed,
                        "training_steps": config.training.total_env_steps,
                        "evaluation_interval": config.evaluation.interval_env_steps,
                        "evaluation_seed": config.evaluation.seed,
                        "git_sha": git_sha,
                        "image_digest": image_uri,
                        "config_hash": config.config_hash(),
                        "gcs_output_uri": gcp.run_prefix(config, "replication"),
                    }
                )
    rows.sort(
        key=lambda row: (row["reward_type"], row["reward_delay"], row["gae_horizon"], row["seed"])
    )
    for task_id, row in enumerate(rows):
        row["task_id"] = task_id
    validate_rows(rows)
    return rows


def validate_rows(rows):
    if len(rows) != 60:
        raise ValueError("VPG replication manifest must contain exactly 60 rows")
    for key in ("run_id", "config_hash", "gcs_output_uri"):
        if len({row[key] for row in rows}) != 60:
            raise ValueError(f"VPG replication has duplicate {key}")
    cells = {}
    for row in rows:
        cell = (row["reward_type"], row["reward_delay"], row["gae_horizon"])
        cells.setdefault(cell, set()).add(row["seed"])
    if len(cells) != 6 or any(seeds != set(range(10)) for seeds in cells.values()):
        raise ValueError("Each VPG cell must contain exactly seeds 0-9")
    if {row["training_steps"] for row in rows} != {1_000_000}:
        raise ValueError("VPG training budget mismatch")
    if {row["evaluation_seed"] for row in rows} != {20260905}:
        raise ValueError("VPG evaluation seeds are inconsistent")
    if {row["algorithm"] for row in rows} != {"vpg"}:
        raise ValueError("Replication manifest contains a non-VPG algorithm")


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
        or run_id_for(config, "replication") != row["run_id"]
    ):
        raise ValueError("VPG replication row identity mismatch")
    return config


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=REPLICATION_SCHEMA), path, compression="zstd")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256.json").write_text(
        json.dumps({"sha256": digest}, indent=2, sort_keys=True) + "\n"
    )
    return digest


def read_manifest(path):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    expected = json.loads(sidecar.read_text())["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("VPG replication manifest checksum mismatch")
    table = pq.read_table(path)
    if not table.schema.equals(REPLICATION_SCHEMA, check_metadata=True):
        raise ValueError("VPG replication schema mismatch")
    rows = table.to_pylist()
    validate_rows(rows)
    return rows

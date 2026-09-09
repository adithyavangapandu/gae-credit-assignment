import json
from dataclasses import replace

import numpy as np
import pyarrow.parquet as pq
import pytest

from gae_credit.config import config_from_dict
from gae_credit.logging import validate_run
from gae_credit.train import run_training


def smoke_config(output_dir, horizon=3, reward="dense"):
    return config_from_dict(
        {
            "seed": 0,
            "environment": {"max_steps": 20},
            "training": {"total_env_steps": 80, "episodes_per_rollout": 2},
            "optimizer": {"epochs": 2, "minibatch_size": 20},
            "evaluation": {"episodes": 2, "interval_env_steps": 40},
            "logging": {"output_dir": str(output_dir)},
            "estimator": {"horizon": horizon},
            "reward": {"kind": reward, "delay_block_size": 8 if reward == "delayed" else 1},
        }
    )


@pytest.mark.parametrize(
    "horizon,reward", [(3, "dense"), ("full", "dense"), (3, "delayed"), ("full", "sparse")]
)
def test_end_to_end_smoke_artifacts(tmp_path, horizon, reward):
    cfg = smoke_config(tmp_path, horizon, reward)
    path = run_training(cfg, verbose=False)
    report = validate_run(path)
    assert report["valid"] and report["updates"] == 2
    assert report["actor_horizon"] == report["critic_horizon"] == horizon
    summary = json.loads((path / "summary.json").read_text())
    assert summary["initial_actor_hash"] != summary["final_actor_hash"]
    assert summary["initial_critic_hash"] != summary["final_critic_hash"]
    with pytest.raises(FileExistsError):
        run_training(cfg, verbose=False)


def test_identical_config_and_seed_reproduce_data_locally(tmp_path):
    cfg = smoke_config(tmp_path)
    path = run_training(cfg, verbose=False)
    tables = {
        name: pq.read_table(path / f"{name}.parquet").to_pydict()
        for name in ("updates", "episodes", "evaluations")
    }
    with np.load(path / "diagnostics" / "update-00001.npz") as data:
        diagnostics = {k: data[k].copy() for k in data.files}
    first = json.loads((path / "summary.json").read_text())
    run_training(cfg, overwrite=True, verbose=False)
    for name, expected in tables.items():
        actual = pq.read_table(path / f"{name}.parquet").to_pydict()
        if name == "updates":
            expected.pop("elapsed_seconds")
            actual.pop("elapsed_seconds")
        assert expected == actual
    with np.load(path / "diagnostics" / "update-00001.npz") as data:
        for key, expected in diagnostics.items():
            np.testing.assert_array_equal(data[key], expected)
    assert first == json.loads((path / "summary.json").read_text())


def test_checkpoint_trajectories_have_twenty_step_rows_per_episode(tmp_path):
    cfg = smoke_config(tmp_path)
    cfg = replace(
        cfg,
        training=replace(cfg.training, checkpoint_interval_env_steps=40),
        logging=replace(cfg.logging, diagnostic_trajectories_per_checkpoint=2),
    )
    path = run_training(cfg, verbose=False)
    files = sorted((path / "diagnostic_trajectories").glob("*.parquet"))
    assert [file.name for file in files] == [
        "checkpoint-0000040.parquet",
        "checkpoint-0000080.parquet",
    ]
    for file in files:
        table = pq.read_table(file)
        assert table.num_rows == 40
        assert set(table["trajectory_id"].to_pylist()) == {0, 1}
        assert set(table["timestep"].to_pylist()) == set(range(1, 21))

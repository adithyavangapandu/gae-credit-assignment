from dataclasses import replace

import numpy as np
import pytest
import torch
from test_smoke_training import smoke_config

from gae_credit.algorithms.networks import Actor, Critic
from gae_credit.algorithms.ppo import prepare_batch
from gae_credit.algorithms.rollout import collect_rollout
from gae_credit.algorithms.vpg import VPG
from gae_credit.cloud.config import GCPConfig, run_id_for
from gae_credit.config import AlgorithmConfig, EnvironmentConfig, RewardConfig
from gae_credit.envs.pendulum import PendulumEnv
from gae_credit.envs.rewards import make_reward
from gae_credit.logging import validate_run
from gae_credit.replication import (
    config_for_row,
    generate_manifest,
    load_replication_spec,
    read_manifest,
    write_manifest,
)
from gae_credit.train import run_training

IMAGE = "us-central1-docker.pkg.dev/example-project/repo/trainer@sha256:" + "d" * 64
GIT_SHA = "e" * 40


def test_vpg_update_is_finite_and_changes_actor_and_critic():
    actor, critic = Actor(4, seed=3), Critic(4, seed=4)
    env = PendulumEnv(EnvironmentConfig(max_steps=20))
    episodes = collect_rollout(
        env,
        make_reward(RewardConfig()),
        actor,
        critic,
        [1, 2],
        torch.Generator().manual_seed(5),
    )
    batch = prepare_batch(episodes, 0.995, 0.95, 3, 3)
    old_actor = [value.clone() for value in actor.parameters()]
    old_critic = [value.clone() for value in critic.parameters()]
    metrics = VPG(actor, critic, AlgorithmConfig(epochs=2, minibatch_size=20)).update(
        batch, torch.Generator().manual_seed(6)
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["clip_fraction"] == 0
    assert any(not torch.equal(old, new) for old, new in zip(old_actor, actor.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(old_critic, critic.parameters()))


def test_vpg_manifest_has_exact_six_cells_and_ten_seeds(tmp_path):
    spec, base = load_replication_spec()
    gcp = GCPConfig(project_id="example-project", artifact_bucket="gs://example-project-data")
    rows = generate_manifest(spec, base, gcp, IMAGE, GIT_SHA)
    assert len(rows) == 60
    assert len({(row["reward_type"], row["reward_delay"], row["gae_horizon"]) for row in rows}) == 6
    assert {row["seed"] for row in rows} == set(range(10))
    assert all(row["gcs_output_uri"].find("algorithm=vpg") >= 0 for row in rows)
    assert all(config_for_row(spec, base, row).algorithm == "vpg" for row in rows)
    path = tmp_path / "vpg.parquet"
    write_manifest(rows, path)
    assert read_manifest(path) == rows
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(path)


def test_vpg_run_identity_uses_replication_phase():
    _, base = load_replication_spec()
    assert run_id_for(base, "replication").startswith("replication-vpg-dense-h003-seed0-")


def test_vpg_training_writes_the_common_artifact_contract(tmp_path):
    config = replace(smoke_config(tmp_path), algorithm="vpg")
    path = run_training(config, verbose=False)
    report = validate_run(path)
    assert report["valid"] and report["updates"] == 2

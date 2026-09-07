from dataclasses import replace

import pytest

from gae_credit.cloud.config import GCPConfig
from gae_credit.config import RewardConfig, config_from_dict, load_config
from gae_credit.pilot import config_for_row, generate_matrix, load_pilot_spec

IMAGE = "us-central1-docker.pkg.dev/example-project/repo/trainer@sha256:" + "a" * 64


def test_day4_config_identity_is_backward_compatible():
    config = load_config("configs/cloud/smoke_dense_h3.yaml")
    assert (
        config.config_hash() == "cc025069deb52eba451ea3128f3d48f1a0e325b3a7d19858236b21197a834c86"
    )
    assert "upright_angle_threshold" not in config.to_dict()["reward"]


def test_sparse_thresholds_are_explicit_and_roundtrip():
    config = config_from_dict(
        {
            "reward": {
                "kind": "sparse",
                "upright_angle_threshold": 0.1,
                "upright_velocity_threshold": 0.5,
            }
        }
    )
    assert config_from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="both"):
        config_from_dict({"reward": {"kind": "sparse", "upright_angle_threshold": 0.1}})
    with pytest.raises(ValueError, match="only valid"):
        config_from_dict({"reward": {"kind": "dense", "upright_angle_threshold": 0.1}})


def test_pilot_matrix_has_exact_deterministic_factorial():
    spec, base = load_pilot_spec("configs/pilot/study_v1.yaml")
    gcp = GCPConfig(project_id="example-project", artifact_bucket="gs://example-bucket")
    rows = generate_matrix(spec, base, gcp, IMAGE)
    assert len(rows) == 24
    assert [row["task_id"] for row in rows] == list(range(24))
    assert len({row["config_hash"] for row in rows}) == 24
    assert {(row["reward_kind"], row["delay_block_size"]) for row in rows} == {
        ("dense", 1),
        ("delayed", 8),
        ("delayed", 32),
        ("sparse", 1),
    }
    assert {row["actor_horizon"] for row in rows} == {"3", "full"}
    assert {row["seed"] for row in rows} == {100, 101, 102}
    for row in rows:
        assert config_for_row(spec, base, row).config_hash() == row["config_hash"]


def test_non_sparse_threshold_override_is_rejected_directly():
    base = config_from_dict({})
    with pytest.raises(ValueError):
        replace(base, reward=RewardConfig(kind="dense", upright_angle_threshold=0.1))

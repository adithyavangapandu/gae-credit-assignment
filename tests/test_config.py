import pytest
import yaml

from gae_credit.config import config_from_dict, derive_seeds, load_config


@pytest.mark.parametrize(
    "raw",
    [{"estimator": {"horizon": h}} for h in (0, -1, True, 1.5, "bad")]
    + [
        {"reward": {"delay_block_size": 0}},
        {"environment": {"max_steps": -1}},
        {"environment": {"max_steps": 0}},
        {"environment": {"include_time_remaining": False}},
        {"estimator": {"gamma": float("nan")}},
        {"training": {"total_env_steps": 1001}},
        {"unexpected": 42},
        {"optimizer": {"actor_lrr": 1}},
        {"estimator": {"horizon": 3, "critic_horizon": "full"}},
        {"reward": {"kind": "delayed"}, "estimator": {"gamma": 0}},
        {"seeds": {"pilot_seeds": [0]}},
        {"evaluation": {"interval_env_steps": 1}},
    ],
)
def test_invalid_config_rejected(raw):
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_resolved_roundtrip_and_hash_are_stable(tmp_path):
    config = load_config("configs/smoke_dense_full.yaml")
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    assert load_config(path) == config
    assert load_config(path).config_hash() == config.config_hash()
    assert config.estimator.actor_horizon == config.estimator.critic_horizon == "full"
    assert config_from_dict({"seed": 1}).config_hash() != config_from_dict({}).config_hash()
    assert (
        config_from_dict({"seed": 0, "algorithm": "ppo"}).config_hash()
        == config_from_dict({}).config_hash()
    )


def test_separate_streams_and_shared_evaluation_seeds():
    a = derive_seeds(config_from_dict({"seed": 0}))
    b = derive_seeds(config_from_dict({"seed": 1}))
    assert len(set(a.values())) == len(a)
    assert a == derive_seeds(config_from_dict({"seed": 0}))
    assert a["evaluation"] == b["evaluation"]
    assert all(a[k] != b[k] for k in a if k != "evaluation")

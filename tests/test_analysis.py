import numpy as np
import pandas as pd

from gae_credit.analysis import (
    bootstrap_interval,
    distance_to_next_reward,
    load_analysis_config,
    normalized_auc,
    reward_condition,
    run_outcomes,
)
from gae_credit.analysis.day8 import holm_adjust, pre_reward_advantage_magnitude
from gae_credit.analysis.sync import ROOT_FILES, analysis_keys


def test_analysis_protocol_and_auc_are_frozen():
    config = load_analysis_config()
    assert config["primary_metric"] == "evaluation_auc"
    assert config["bootstrap_samples"] == 10_000
    assert normalized_auc([0, 500_000, 1_000_000], [-1000, -500, 0], 1_000_000) == -500
    assert reward_condition("delayed", 32) == "delayed_32"


def test_run_outcomes_preserve_censoring_and_fixed_windows():
    config = load_analysis_config()
    steps = np.arange(0, 1_000_001, 10_000)
    evaluations = pd.DataFrame(
        {
            "run_id": "run",
            "env_steps": steps,
            "mean_base_dense_return": np.linspace(-1000, -300, len(steps)),
            "mean_upright_fraction": np.linspace(0, 0.5, len(steps)),
            "stable_success_rate": np.linspace(0, 0.4, len(steps)),
        }
    )
    update_steps = np.arange(1_000, 1_000_001, 1_000)
    updates = pd.DataFrame(
        {
            "run_id": "run",
            "env_steps": update_steps,
            "approx_kl": np.zeros(len(update_steps)),
            "clip_fraction": np.zeros(len(update_steps)),
            "explained_variance": np.ones(len(update_steps)),
            "critic_loss": np.full(len(update_steps), 4.0),
        }
    )
    result = run_outcomes(evaluations, updates, config)
    assert result["steps_to_success"] is None
    assert result["first_successful_evaluation_step"] is None
    assert result["final_critic_mse"] == 4.0
    assert result["training_instability_rate"] == 0.0


def test_distance_bootstrap_and_holm_are_deterministic():
    np.testing.assert_array_equal(distance_to_next_reward([0, 0, 1, 0, 2]), [2, 1, 0, 1, 0])
    first = bootstrap_interval([1, 2, 3], 100, 0.95, 7)
    assert first == bootstrap_interval([1, 2, 3], 100, 0.95, 7)
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_analysis_mirror_excludes_model_checkpoints():
    files = {
        **{name: {} for name in ROOT_FILES},
        "diagnostics/update-00001.npz": {},
        "diagnostic_trajectories/checkpoint-0050000.parquet": {},
        "checkpoints/final.pt": {},
        "recovery/update-000050-deadbeef.pt": {},
    }
    keys = analysis_keys({"files": files})
    assert "diagnostics/update-00001.npz" in keys
    assert "diagnostic_trajectories/checkpoint-0050000.parquet" in keys
    assert "checkpoints/final.pt" not in keys
    assert not any(key.startswith("recovery/") for key in keys)


def test_pre_reward_advantage_is_reduced_at_run_level(tmp_path):
    np.savez_compressed(
        tmp_path / "update-00001.npz",
        rewards=np.array([0.0, 0.0, 1.0, 0.0]),
        advantages_3=np.array([-2.0, 1.0, 4.0, 100.0]),
    )
    magnitude, count = pre_reward_advantage_magnitude(tmp_path, "3")
    assert count == 2
    assert magnitude == 1.5

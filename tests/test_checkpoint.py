import json
from dataclasses import replace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from test_smoke_training import smoke_config

from gae_credit.checkpoint import TrainingInterrupted
from gae_credit.train import run_training


def tables(path):
    result = {
        name: pq.read_table(path / f"{name}.parquet").to_pydict()
        for name in ("updates", "episodes", "evaluations")
    }
    result["updates"].pop("elapsed_seconds")
    return result


@pytest.mark.parametrize("reward", ["dense", "delayed", "sparse"])
def test_interrupted_resume_matches_uninterrupted_training(tmp_path, reward):
    cfg = smoke_config(tmp_path, reward=reward)
    cfg = replace(cfg, training=replace(cfg.training, checkpoint_interval_env_steps=40))
    path = run_training(cfg, verbose=False)
    expected = tables(path)
    summary = json.loads((path / "summary.json").read_text())
    with np.load(path / "diagnostics" / "update-00002.npz") as arrays:
        diagnostics = {k: arrays[k].copy() for k in arrays.files}
    with pytest.raises(TrainingInterrupted):
        run_training(cfg, overwrite=True, verbose=False, stop_after_update=1)
    assert not (path / "_SUCCESS").exists()
    checkpoint = path / "checkpoints" / "update-000001.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["rng"].keys() >= {
        "python",
        "numpy_global",
        "torch_global",
        "action",
        "environment",
    }
    assert payload["reward_state"]["kind"] == reward
    run_training(cfg, resume_path=checkpoint, verbose=False)
    assert tables(path) == expected
    assert json.loads((path / "summary.json").read_text()) == summary
    with np.load(path / "diagnostics" / "update-00002.npz") as arrays:
        for key, value in diagnostics.items():
            np.testing.assert_array_equal(arrays[key], value)
    assert (path / "_SUCCESS").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "different"),
        ("config_hash", "wrong"),
        ("source_digest", "wrong"),
        ("image_digest", "sha256:wrong"),
        ("checkpoint_schema_version", 999),
        ("env_steps", 999),
        ("runtime_fingerprint", "wrong"),
    ],
)
def test_resume_mismatch_fails_before_changing_run(tmp_path, field, value):
    cfg = smoke_config(tmp_path)
    with pytest.raises(TrainingInterrupted):
        run_training(cfg, verbose=False, stop_after_update=1)
    path = tmp_path / cfg.run_id
    checkpoint = path / "checkpoints" / "update-000001.pt"
    data = torch.load(checkpoint, weights_only=True)
    data[field] = value
    bad = tmp_path / "bad.pt"
    torch.save(data, bad)
    before = (path / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="resume"):
        run_training(cfg, resume_path=bad, verbose=False)
    assert (path / "manifest.json").read_bytes() == before
    assert not (path / "_SUCCESS").exists()

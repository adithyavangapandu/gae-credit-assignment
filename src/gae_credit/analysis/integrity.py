"""Confirmatory identity checks shared by synchronization and analysis."""

import json

import numpy as np
import pyarrow.parquet as pq

from gae_credit.cloud.config import image_digest
from gae_credit.confirmatory import config_for_row
from gae_credit.logging import validate_run


def validate_confirmatory_run(path, row, spec, base):
    report = validate_run(path)
    config = config_for_row(spec, base, row)
    manifest = json.loads((path / "manifest.json").read_text())
    expected = {
        "phase": "confirmatory",
        "config_hash": row["config_hash"],
        "git_commit": row["git_sha"],
        "git_dirty": False,
        "image_digest": image_digest(row["image_digest"]),
        "gcp_project": "gae-experiment-507805",
        "gcp_region": "us-central1",
        "environment_version": "pendulum-poststep-v1",
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"run identity differs from frozen manifest: {mismatches}")
    updates = pq.read_table(path / "updates.parquet").to_pandas()
    if not np.isfinite(updates.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("training metrics contain nonfinite values")
    expected_updates = config.training.total_env_steps // (
        config.training.episodes_per_rollout * config.environment.max_steps
    )
    if len(updates) != expected_updates:
        raise ValueError("run terminated before the frozen training budget")
    return {**report, "status": "valid", "runtime_seconds": float(updates.elapsed_seconds.iloc[-1])}

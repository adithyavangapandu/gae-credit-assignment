import json

import pytest

from gae_credit.cloud.config import GCPConfig
from gae_credit.confirmatory import (
    config_for_row,
    generate_manifest,
    load_confirmatory_spec,
    read_manifest,
    write_manifest,
)

IMAGE = "us-central1-docker.pkg.dev/example-project/repo/trainer@sha256:" + "b" * 64
GIT_SHA = "c" * 40


def test_confirmatory_manifest_is_exact_factorial_and_checksummed(tmp_path):
    spec, base = load_confirmatory_spec("configs/confirmatory/study_v1.yaml")
    gcp = GCPConfig(project_id="example-project", artifact_bucket="gs://example-bucket")
    rows = generate_manifest(spec, base, gcp, IMAGE, GIT_SHA)
    assert len(rows) == 160
    assert {row["seed"] for row in rows} == set(range(10))
    assert {row["gae_horizon"] for row in rows} == {"1", "3", "16", "full"}
    assert len({row["run_id"] for row in rows}) == 160
    assert all(row["training_steps"] == 1_000_000 for row in rows)
    assert all(config_for_row(spec, base, row).seed == row["seed"] for row in rows)
    path = tmp_path / "matrix.parquet"
    digest = write_manifest(rows, path)
    assert read_manifest(path) == rows
    sidecar = json.loads((tmp_path / "matrix.parquet.sha256.json").read_text())
    assert sidecar == {"sha256": digest}
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(path)


def test_confirmatory_config_freezes_requested_diagnostics():
    _, base = load_confirmatory_spec("configs/confirmatory/study_v1.yaml")
    assert base.logging.diagnostic_trajectories_per_checkpoint == 20
    assert base.training.checkpoint_interval_env_steps == 50_000
    assert base.evaluation.interval_env_steps == 10_000

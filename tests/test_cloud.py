import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_smoke_training import smoke_config

from gae_credit.checkpoint import TrainingInterrupted
from gae_credit.cloud.config import (
    GCPConfig,
    gcp_from_dict,
    image_digest,
    run_id_for,
    source_digest,
)
from gae_credit.cloud.submit import build_job_spec, submit_job
from gae_credit.logging import validate_run
from gae_credit.logging.gcs import GCSArtifactLogger
from gae_credit.logging.vertex import VertexMetricsLogger
from gae_credit.storage.download import download_run
from gae_credit.storage.local import LocalArtifactStore
from gae_credit.train import run_training

IMAGE = "us-central1-docker.pkg.dev/example-project/gae-training/trainer@sha256:" + "a" * 64


def gcp():
    return GCPConfig(
        project_id="example-project",
        artifact_bucket="gs://example-project-data",
        tensorboard="projects/123456/locations/us-central1/tensorboards/123",
    )


def context(cfg):
    run_id = run_id_for(cfg, "smoke")
    return {
        "phase": "smoke",
        "image_digest": image_digest(IMAGE),
        "vertex_run_id": run_id,
        "artifact_prefix": gcp().run_prefix(cfg, "smoke"),
    }


def identity(cfg):
    return {
        "run_id": run_id_for(cfg, "smoke"),
        "config_hash": cfg.config_hash(),
        "source_digest": source_digest(),
        "image_digest": image_digest(IMAGE),
    }


class RecordingStore(LocalArtifactStore):
    def __init__(self, root, fail_key=None):
        super().__init__(root)
        self.writes = []
        self.fail_key = fail_key

    def write_bytes(self, key, data, *, overwrite=False):
        if key == self.fail_key:
            raise OSError("simulated network interruption")
        super().write_bytes(key, data, overwrite=overwrite)
        self.writes.append(key)


def test_deployment_overrides_identity_and_dry_run_have_no_clients(tmp_path):
    cfg = smoke_config(tmp_path)
    cloud = gcp_from_dict(
        gcp().to_dict(),
        environ={
            "GAE_GCP_REGION": "us-east1",
            "GAE_VERTEX_TENSORBOARD": "projects/123/locations/us-east1/tensorboards/1",
        },
    )
    assert cloud.region == "us-east1"
    spec = build_job_spec(
        cfg, gcp(), "smoke", IMAGE, "runner@example-project.iam.gserviceaccount.com"
    )
    assert len(spec["display_name"]) <= 63
    assert spec["labels"]["run_id"] == run_id_for(cfg, "smoke")
    assert len(spec["worker_pool_specs"]) == 1
    env = {
        item["name"]: item["value"]
        for item in spec["worker_pool_specs"][0]["container_spec"]["env"]
    }
    assert env["GAE_RUN_ID"] in env["GAE_GCS_PREFIX"]
    assert json.loads(env["GAE_RESOLVED_CONFIG"]) == cfg.to_dict()
    with pytest.raises(ValueError, match="Run ID"):
        build_job_spec(
            cfg,
            gcp(),
            "smoke",
            IMAGE,
            "runner@example-project.iam.gserviceaccount.com",
            "manual-test",
        )
    with pytest.raises(ValueError, match="immutable"):
        image_digest(IMAGE.split("@")[0] + ":latest")


def test_prefix_collision_and_generation_guard(tmp_path):
    cfg = smoke_config(tmp_path / "runs")
    store = RecordingStore(tmp_path / "bucket")
    first = GCSArtifactLogger(store, identity(cfg))
    with pytest.raises(FileExistsError):
        GCSArtifactLogger(store, identity(cfg))
    first.release()
    with pytest.raises(ValueError, match="different"):
        GCSArtifactLogger(store, {**identity(cfg), "image_digest": "changed"})
    assert not store.exists("_LEASE.json")


def test_completed_cloud_bundle_downloads_with_identical_schemas(tmp_path):
    cfg = smoke_config(tmp_path / "runs")
    store = RecordingStore(tmp_path / "bucket")
    artifacts = GCSArtifactLogger(store, identity(cfg))
    path = run_training(cfg, context=context(cfg), artifact_logger=artifacts, verbose=False)
    assert store.writes[-1] == "_SUCCESS"
    assert not store.exists("_LEASE.json")
    destination = tmp_path / "downloaded" / path.name
    assert download_run(store, destination)["valid"]
    assert validate_run(destination) == validate_run(path)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["image_digest"] == image_digest(IMAGE)
    with pytest.raises(FileExistsError):
        GCSArtifactLogger(store, identity(cfg))
    (destination / "checkpoints" / "final.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_run(destination)


def test_failed_upload_never_publishes_success_and_can_resume(tmp_path):
    cfg = smoke_config(tmp_path / "runs")
    store = RecordingStore(tmp_path / "bucket", fail_key="updates.parquet")
    with pytest.raises(OSError):
        run_training(
            cfg,
            context=context(cfg),
            artifact_logger=GCSArtifactLogger(store, identity(cfg)),
            verbose=False,
        )
    assert not store.exists("_SUCCESS")
    assert not store.exists("_LEASE.json")
    path = Path(cfg.logging.output_dir) / run_id_for(cfg, "smoke")
    assert not (path / "_SUCCESS").exists()
    assert json.loads((path / "manifest.json").read_text())["status"] == "failed"
    store.fail_key = None
    artifacts = GCSArtifactLogger(store, identity(cfg))
    checkpoint = artifacts.download_checkpoint(tmp_path / "recovery.pt")
    run_training(
        cfg, context=context(cfg), artifact_logger=artifacts, resume_path=checkpoint, verbose=False
    )
    assert store.exists("_SUCCESS")


def test_remote_checkpoint_restarts_on_fresh_local_directory(tmp_path):
    cfg = smoke_config(tmp_path / "runs")
    cfg = replace(cfg, training=replace(cfg.training, checkpoint_interval_env_steps=40))
    store = RecordingStore(tmp_path / "bucket")
    with pytest.raises(TrainingInterrupted):
        run_training(
            cfg,
            context=context(cfg),
            artifact_logger=GCSArtifactLogger(store, identity(cfg)),
            verbose=False,
            stop_after_update=1,
        )
    assert store.exists("_LATEST_CHECKPOINT.json") and not store.exists("_SUCCESS")
    path = Path(cfg.logging.output_dir) / run_id_for(cfg, "smoke")
    path.rename(tmp_path / "old-worker-files")
    artifacts = GCSArtifactLogger(store, identity(cfg))
    checkpoint = artifacts.download_checkpoint(tmp_path / "resume.pt")
    completed = run_training(
        cfg, context=context(cfg), artifact_logger=artifacts, resume_path=checkpoint, verbose=False
    )
    assert validate_run(completed)["updates"] == 2


def test_compact_vertex_metrics_and_parameters(tmp_path):
    pytest.importorskip("google.cloud.aiplatform_v1")
    sdk = MagicMock()
    cfg = smoke_config(tmp_path)
    logger = VertexMetricsLogger(gcp(), run_id_for(cfg, "smoke"), cfg, sdk=sdk)
    logger.log_update(
        dict.fromkeys(
            (
                "actor_loss",
                "critic_loss",
                "advantage_std",
                "explained_variance",
                "entropy",
                "actor_grad_norm",
                "critic_grad_norm",
                "approx_kl",
                "clip_fraction",
                "env_steps",
            ),
            1,
        )
    )
    logger.log_evaluation(
        {"mean_base_dense_return": -20, "stable_success_rate": 0.5, "env_steps": 2}
    )
    logger.close()
    assert sdk.log_time_series_metrics.call_count == 2
    assert sdk.log_time_series_metrics.call_args.kwargs["step"] == 2
    assert sdk.log_params.call_args.args[0]["config_hash"] == cfg.config_hash()


def test_submission_is_fixed_cpu_and_rejects_completed_or_duplicate(tmp_path):
    cfg = smoke_config(tmp_path / "runs")
    store = RecordingStore(tmp_path / "bucket")
    sdk = MagicMock()
    sdk.CustomJob.return_value.resource_name = "projects/123/locations/us-central1/customJobs/456"
    spec = build_job_spec(
        cfg, gcp(), "smoke", IMAGE, "runner@example-project.iam.gserviceaccount.com"
    )
    result = submit_job(spec, gcp(), store=store, sdk=sdk)
    assert result["job_resource_name"].endswith("/456")
    assert sdk.CustomJob.return_value.submit.call_args.kwargs["disable_retries"] is True
    with pytest.raises(FileExistsError):
        submit_job(spec, gcp(), store=store, sdk=sdk)
    assert not store.exists("_SUBMIT_LOCK.json")
    sdk.CustomJob.get.return_value.state = SimpleNamespace(name="JOB_STATE_RUNNING")
    store.write_json("_LATEST_CHECKPOINT.json", {})
    with pytest.raises(ValueError, match="stopped"):
        submit_job(spec, gcp(), resume=True, store=store, sdk=sdk)


@pytest.mark.parametrize("key", ["../escape", "/absolute", "a/../../b", "a\\b", "a//b"])
def test_store_rejects_unsafe_paths(tmp_path, key):
    with pytest.raises(ValueError):
        LocalArtifactStore(tmp_path).write_json(key, {})

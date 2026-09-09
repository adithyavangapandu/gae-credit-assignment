"""Construct one fixed CPU Custom Job; dry-run construction needs no ADC."""

import json
import os
import re

from gae_credit.cloud.config import image_digest, run_id_for
from gae_credit.storage.gcs import GCSArtifactStore


def build_job_spec(config, gcp, phase, image_uri, service_account, run_id=None, *, resume=False):
    gcp.validate()
    image_digest(image_uri)
    expected = run_id_for(config, phase)
    requested = run_id or os.environ.get("GAE_RUN_ID") or expected
    if requested != expected:
        raise ValueError(f"Run ID must match the resolved config/phase: {expected}")
    if not re.fullmatch(
        r"[a-z][a-z0-9-]*@" + re.escape(gcp.project_id) + r"\.iam\.gserviceaccount\.com",
        service_account,
    ):
        raise ValueError("Use a runtime service account from the configured project")
    if phase == "pilot" and config.seed not in config.seeds.pilot_seeds:
        raise ValueError("Pilot jobs must use an excluded pilot seed")
    if phase == "confirmatory" and config.seed not in config.seeds.final_training_seeds:
        raise ValueError("Confirmatory jobs must use a final training seed")
    if phase == "pilot" and config.training.checkpoint_interval_env_steps != 50_000:
        raise ValueError("Pilot jobs require checkpoints every 50,000 environment steps")
    prefix = gcp.run_prefix(config, phase)
    h = config.estimator.horizon
    env = {
        "GAE_RESOLVED_CONFIG": json.dumps(config.to_dict(), sort_keys=True),
        "GAE_GCP_CONFIG": json.dumps(gcp.to_dict(), sort_keys=True),
        "GAE_PHASE": phase,
        "GAE_RUN_ID": expected,
        "GAE_GCS_PREFIX": prefix,
        "GAE_IMAGE_URI": image_uri,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    return {
        "display_name": expected,
        "labels": {
            "run_id": expected,
            "phase": phase,
            "algorithm": config.algorithm,
            "reward": config.reward.kind
            + (f"-{config.reward.delay_block_size}" if config.reward.kind == "delayed" else ""),
            "horizon": "full" if h == "full" else f"h{h:03d}",
            "seed": str(config.seed),
        },
        "worker_pool_specs": [
            {
                "replica_count": 1,
                "machine_spec": {"machine_type": gcp.machine_type},
                "container_spec": {
                    "image_uri": image_uri,
                    "command": ["python", "-m", "gae_credit.train"],
                    "args": ["--cloud"] + (["--resume-latest"] if resume else []),
                    "env": [{"name": k, "value": v} for k, v in env.items()],
                },
            }
        ],
        "service_account": service_account,
        "artifact_prefix": prefix,
        "project": gcp.project_id,
        "location": gcp.region,
        "timeout_seconds": gcp.timeout_seconds,
        "config_hash": config.config_hash(),
    }


def submit_job(spec, gcp, *, resume=False, store=None, sdk=None):
    gcp.validate(deployed=True)
    if store is None:
        store = GCSArtifactStore(spec["artifact_prefix"], project=gcp.project_id)
    if store.exists("_SUCCESS"):
        raise FileExistsError("Run already completed; refusing duplicate submission")
    if sdk is None:
        from google.cloud import aiplatform as sdk
    # A separate submission lock also covers uncertain API outcomes. Never blindly retry.
    store.write_json("_SUBMIT_LOCK.json", {"run_id": spec["display_name"]})
    try:
        if resume:
            if not store.exists("_LATEST_CHECKPOINT.json") or not store.exists("_SUBMISSION.json"):
                raise ValueError("Resume requires a published checkpoint and previous job record")
            previous = json.loads(store.read_bytes("_SUBMISSION.json"))
            if previous.get("config_hash") != spec["config_hash"]:
                raise ValueError("Previous submission config mismatch")
            old_job = sdk.CustomJob.get(previous["job_resource_name"])
            if old_job.state.name not in {
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
                "JOB_STATE_EXPIRED",
            }:
                raise ValueError("Previous job must be stopped before resuming")
            # The old job is proven terminal; no old worker can still hold this lease.
            store.delete("_LEASE.json")
        elif store.exists("_SUBMISSION.json"):
            raise FileExistsError("Run already submitted; use explicit resume after failure")
        job = sdk.CustomJob(
            display_name=spec["display_name"],
            worker_pool_specs=spec["worker_pool_specs"],
            labels=spec["labels"],
            project=gcp.project_id,
            location=gcp.region,
            staging_bucket=gcp.artifact_bucket,
        )
        job.submit(
            service_account=spec["service_account"],
            timeout=spec["timeout_seconds"],
            restart_job_on_worker_restart=False,
            disable_retries=True,
        )
        result = {
            "job_resource_name": job.resource_name,
            "run_id": spec["display_name"],
            "config_hash": spec["config_hash"],
            "artifact_prefix": spec["artifact_prefix"],
            "attempt_count": previous.get("attempt_count", 1) + 1 if resume else 1,
        }
        store.write_json("_SUBMISSION.json", result, overwrite=resume)
    except (ValueError, FileExistsError, KeyError):
        store.delete("_SUBMIT_LOCK.json")
        raise
    # For network/API errors leave the lock: submission might have been accepted.
    store.delete("_SUBMIT_LOCK.json")
    return result

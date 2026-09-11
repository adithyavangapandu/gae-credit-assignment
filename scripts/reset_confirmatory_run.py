"""Safely clear one terminal failed confirmatory attempt for a step-zero restart."""

import argparse
import json

from gae_credit.cloud.config import image_digest, load_gcp_config
from gae_credit.confirmatory import read_manifest
from gae_credit.storage.gcs import GCSArtifactStore

TERMINAL_FAILURES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", default="data/manifests/ppo_confirmatory_v1.parquet")
    parser.add_argument("--gcp-config", default="configs/cloud/gcp.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm-run-id",
        help="Required with --execute; must exactly repeat --run-id",
    )
    args = parser.parse_args()
    if args.execute and args.confirm_run_id != args.run_id:
        parser.error("--execute requires an exact --confirm-run-id")
    rows = read_manifest(args.manifest)
    matches = [row for row in rows if row["run_id"] == args.run_id]
    if len(matches) != 1:
        raise ValueError("Run ID is not a unique frozen manifest row")
    row = matches[0]
    gcp = load_gcp_config(args.gcp_config).validate(deployed=True)
    store = GCSArtifactStore(row["gcs_output_uri"], project=gcp.project_id)
    if store.exists("_SUCCESS"):
        raise ValueError("Completed runs cannot be reset")
    if not store.exists("_SUBMISSION.json") or not store.exists("_IDENTITY.json"):
        raise ValueError("Reset requires an existing submitted run identity")
    receipt = json.loads(store.read_bytes("_SUBMISSION.json"))
    identity = json.loads(store.read_bytes("_IDENTITY.json"))
    expected_identity = {
        "run_id": row["run_id"],
        "config_hash": row["config_hash"],
        "image_digest": image_digest(row["image_digest"]),
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Stored GCS identity does not match the frozen manifest")
    from google.cloud import aiplatform

    job = aiplatform.CustomJob.get(receipt["job_resource_name"])
    if job.state.name not in TERMINAL_FAILURES:
        raise ValueError(f"Run is not a terminal failure: {job.state.name}")
    blobs = list(store.bucket.list_blobs(prefix=store.prefix + "/"))
    experiment_run = aiplatform.ExperimentRun.get(
        row["run_id"],
        experiment=gcp.vertex_experiment,
        project=gcp.project_id,
        location=gcp.region,
    )
    plan = {
        "run_id": row["run_id"],
        "job_resource_name": receipt["job_resource_name"],
        "job_state": job.state.name,
        "gcs_prefix": row["gcs_output_uri"],
        "gcs_objects": len(blobs),
        "vertex_experiment_run_exists": experiment_run is not None,
        "action": "execute-reset" if args.execute else "dry-run",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if experiment_run is not None:
        experiment_run.delete(delete_backing_tensorboard_run=True)
    for blob in blobs:
        blob.delete(if_generation_match=int(blob.generation))
    print(json.dumps({"run_id": row["run_id"], "status": "reset-complete"}, indent=2))


if __name__ == "__main__":
    main()

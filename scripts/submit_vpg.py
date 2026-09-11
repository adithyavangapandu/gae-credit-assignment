"""Submit checksum-frozen VPG replication jobs with a six-run canary gate."""

import argparse
import json
from pathlib import Path

from gae_credit.cloud.config import load_gcp_config
from gae_credit.cloud.submit import build_job_spec, submit_job
from gae_credit.replication import config_for_row, load_replication_spec, read_manifest
from gae_credit.storage.gcs import GCSArtifactStore

ACTIVE_STATES = {"JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_RUNNING"}
RETRYABLE_STATES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def job_state(row, project, sdk):
    store = GCSArtifactStore(row["gcs_output_uri"], project=project)
    if store.exists("_SUCCESS"):
        return "JOB_STATE_SUCCEEDED", store
    if not store.exists("_SUBMISSION.json"):
        return "NOT_SUBMITTED", store
    receipt = json.loads(store.read_bytes("_SUBMISSION.json"))
    return sdk.CustomJob.get(receipt["job_resource_name"]).state.name, store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/vpg_replication_v1.parquet")
    parser.add_argument("--study-config", default="configs/replication/vpg_v1.yaml")
    parser.add_argument("--gcp-config", default="configs/cloud/gcp_vpg.yaml")
    parser.add_argument("--preflight", default="reports/vpg/preflight_report.json")
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--batch", choices=("canary", "remaining"), default="canary")
    parser.add_argument("--resume-run")
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.offline and (args.submit or args.resume_run):
        parser.error("offline mode cannot submit or verify a resumed run")
    if not 1 <= args.max_concurrency <= 32:
        raise ValueError("VPG concurrency must be between one and 32")
    rows = read_manifest(args.manifest)
    preflight = json.loads(Path(args.preflight).read_text())
    sidecar = json.loads(Path(args.manifest + ".sha256.json").read_text())
    if preflight.get("status") != "READY" or preflight.get("manifest_sha256") != sidecar["sha256"]:
        raise RuntimeError("VPG preflight does not approve this frozen manifest")
    spec, base = load_replication_spec(args.study_config)
    gcp = load_gcp_config(args.gcp_config)
    stores = {}
    if args.offline:
        states = {row["run_id"]: "NOT_SUBMITTED" for row in rows}
    else:
        from google.cloud import aiplatform

        states = {}
        for row in rows:
            states[row["run_id"]], stores[row["run_id"]] = job_state(
                row, gcp.project_id, aiplatform
            )
    active = sum(state in ACTIVE_STATES for state in states.values())
    if args.resume_run:
        matches = [row for row in rows if row["run_id"] == args.resume_run]
        if len(matches) != 1 or states.get(args.resume_run) not in RETRYABLE_STATES:
            raise RuntimeError("Resume requires one frozen terminal failed run")
        selected = matches
        resume = True
    else:
        canaries = [row for row in rows if row["seed"] == 0]
        if len(canaries) != 6:
            raise ValueError("VPG canary must contain all six seed-zero cells")
        if args.batch == "remaining":
            unfinished = [
                row["run_id"] for row in canaries if states[row["run_id"]] != "JOB_STATE_SUCCEEDED"
            ]
            if unfinished:
                raise RuntimeError(f"VPG canary has not passed: {unfinished}")
            selected = [row for row in rows if row["seed"] != 0]
        else:
            selected = canaries
        resume = False
    results = []
    for row in selected:
        state = states[row["run_id"]]
        if not resume and state != "NOT_SUBMITTED":
            results.append({"run_id": row["run_id"], "action": "skip", "state": state})
            continue
        if active >= args.max_concurrency:
            results.append({"run_id": row["run_id"], "action": "defer", "state": state})
            continue
        config = config_for_row(spec, base, row)
        job_spec = build_job_spec(
            config,
            gcp,
            "replication",
            row["image_digest"],
            args.service_account,
            row["run_id"],
            resume=resume,
        )
        if args.submit:
            receipt = submit_job(
                job_spec,
                gcp,
                resume=resume,
                store=stores[row["run_id"]],
                sdk=aiplatform,
            )
            results.append(
                {"run_id": row["run_id"], "action": "resumed" if resume else "submitted", **receipt}
            )
            active += 1
        else:
            results.append(
                {"run_id": row["run_id"], "action": "ready-to-resume" if resume else "ready"}
            )
    print(json.dumps({"batch": args.batch, "active_jobs": active, "results": results}, indent=2))


if __name__ == "__main__":
    main()

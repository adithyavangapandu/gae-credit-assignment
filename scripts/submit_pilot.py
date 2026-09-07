"""Submit the excluded PPO pilot through its four-run quality gate."""

import argparse
import json

from gae_credit.cloud.config import load_gcp_config
from gae_credit.cloud.submit import build_job_spec, submit_job
from gae_credit.pilot import config_for_row, load_pilot_spec, read_matrix
from gae_credit.storage.gcs import GCSArtifactStore

INITIAL_GATE = {
    ("dense", 1, "3", 100),
    ("dense", 1, "full", 100),
    ("delayed", 32, "3", 100),
    ("sparse", 1, "full", 100),
}
ACTIVE_STATES = {"JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_RUNNING"}


def selected_rows(rows, batch):
    initial = [
        row
        for row in rows
        if (row["reward_kind"], row["delay_block_size"], row["actor_horizon"], row["seed"])
        in INITIAL_GATE
    ]
    if len(initial) != 4:
        raise ValueError("Pilot matrix does not contain the exact four-run initial gate")
    return initial if batch == "initial" else [row for row in rows if row not in initial]


def job_state(row, project, region, sdk):
    store = GCSArtifactStore(row["expected_gcs_prefix"], project=project)
    if store.exists("_SUCCESS"):
        return "JOB_STATE_SUCCEEDED", store
    if not store.exists("_SUBMISSION.json"):
        return "NOT_SUBMITTED", store
    receipt = json.loads(store.read_bytes("_SUBMISSION.json"))
    job = sdk.CustomJob.get(receipt["job_resource_name"])
    return job.state.name, store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="data/manifests/pilot_run_matrix.parquet")
    parser.add_argument("--study-config", default="configs/pilot/study_v1.yaml")
    parser.add_argument("--gcp-config", default="configs/cloud/gcp.yaml")
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--batch", choices=("initial", "remaining"), default="initial")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument(
        "--submit", action="store_true", help="Create jobs; otherwise preflight only"
    )
    args = parser.parse_args()
    if not 1 <= args.max_concurrency <= 8:
        raise ValueError("Pilot concurrency must be between one and eight")
    rows = read_matrix(args.matrix)
    if len({row["config_hash"] for row in rows}) != len(rows):
        raise ValueError("Duplicate configuration hashes in pilot matrix")
    spec, base = load_pilot_spec(args.study_config)
    gcp = load_gcp_config(args.gcp_config)
    from google.cloud import aiplatform

    states = {}
    stores = {}
    for row in rows:
        states[row["run_id"]], stores[row["run_id"]] = job_state(
            row, gcp.project_id, gcp.region, aiplatform
        )
    if args.batch == "remaining":
        gate = selected_rows(rows, "initial")
        unfinished = [
            row["run_id"] for row in gate if states[row["run_id"]] != "JOB_STATE_SUCCEEDED"
        ]
        if unfinished:
            raise RuntimeError(f"Initial quality gate has not passed: {unfinished}")
    active = sum(state in ACTIVE_STATES for state in states.values())
    results = []
    for row in selected_rows(rows, args.batch):
        state = states[row["run_id"]]
        if state != "NOT_SUBMITTED":
            results.append({"run_id": row["run_id"], "action": "skip", "state": state})
            continue
        if active >= args.max_concurrency:
            results.append({"run_id": row["run_id"], "action": "defer", "state": state})
            continue
        config = config_for_row(spec, base, row)
        job_spec = build_job_spec(
            config,
            gcp,
            "pilot",
            row["image_digest"],
            args.service_account,
            row["run_id"],
        )
        if args.submit:
            receipt = submit_job(job_spec, gcp, store=stores[row["run_id"]], sdk=aiplatform)
            results.append({"run_id": row["run_id"], "action": "submitted", **receipt})
            active += 1
        else:
            results.append({"run_id": row["run_id"], "action": "ready"})
    print(json.dumps({"batch": args.batch, "active_jobs": active, "results": results}, indent=2))


if __name__ == "__main__":
    main()

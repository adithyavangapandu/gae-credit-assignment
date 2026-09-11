"""Submit checksum-frozen confirmatory PPO jobs with staged concurrency."""

import argparse
import json

from gae_credit.cloud.config import load_gcp_config
from gae_credit.cloud.submit import build_job_spec, submit_job
from gae_credit.confirmatory import config_for_row, load_confirmatory_spec, read_manifest
from gae_credit.storage.gcs import GCSArtifactStore

ACTIVE_STATES = {"JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_RUNNING"}
RETRYABLE_STATES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def is_canary(row):
    return row["seed"] == 0 and row["gae_horizon"] in {"3", "full"}


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
    parser.add_argument("--manifest", default="data/manifests/ppo_confirmatory_v1.parquet")
    parser.add_argument("--study-config", default="configs/confirmatory/study_v1.yaml")
    parser.add_argument("--gcp-config", default="configs/cloud/gcp.yaml")
    parser.add_argument("--preflight", default="reports/day6/preflight_report.json")
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--batch", choices=("canary", "remaining"), default="canary")
    parser.add_argument("--resume-run", help="Resume one terminal failed run from its checkpoint")
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Validate job specs without remote state checks; incompatible with --submit",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.offline and args.submit:
        parser.error("--offline cannot be combined with --submit")
    if args.offline and args.resume_run:
        parser.error("--offline cannot verify a resumable cloud run")
    if not 1 <= args.max_concurrency <= 32:
        raise ValueError("Day 6 concurrency must be between one and 32")
    rows = read_manifest(args.manifest)
    frozen = json.loads(open(args.preflight).read())
    if frozen.get("status") != "READY":
        raise RuntimeError("Day 6 preflight is not READY; refusing submission")
    expected_digest = json.loads(open(args.manifest + ".sha256.json").read())["sha256"]
    if frozen.get("manifest_sha256") != expected_digest:
        raise RuntimeError("Preflight report does not approve this manifest checksum")
    spec, base = load_confirmatory_spec(args.study_config)
    gcp = load_gcp_config(args.gcp_config)
    states, stores = {}, {}
    if args.offline:
        states = {row["run_id"]: "NOT_SUBMITTED" for row in rows}
    else:
        from google.cloud import aiplatform

        for row in rows:
            states[row["run_id"]], stores[row["run_id"]] = job_state(
                row, gcp.project_id, aiplatform
            )
    canaries = [row for row in rows if is_canary(row)]
    if len(canaries) != 8:
        raise ValueError("Confirmatory canary must contain eight representative runs")
    active = sum(state in ACTIVE_STATES for state in states.values())
    if args.resume_run:
        matches = [row for row in rows if row["run_id"] == args.resume_run]
        if len(matches) != 1:
            raise ValueError("--resume-run must name exactly one frozen manifest row")
        row = matches[0]
        state = states[row["run_id"]]
        if state not in RETRYABLE_STATES:
            raise RuntimeError(f"Run is not terminal and retryable: {state}")
        if active >= args.max_concurrency:
            raise RuntimeError("Concurrency cap is full; wait before resuming")
        config = config_for_row(spec, base, row)
        job_spec = build_job_spec(
            config,
            gcp,
            "confirmatory",
            row["image_digest"],
            args.service_account,
            row["run_id"],
            resume=True,
        )
        result = {"run_id": row["run_id"], "action": "ready-to-resume", "state": state}
        if args.submit:
            receipt = submit_job(
                job_spec,
                gcp,
                resume=True,
                store=stores[row["run_id"]],
                sdk=aiplatform,
            )
            result = {"run_id": row["run_id"], "action": "resumed", **receipt}
            active += 1
        print(json.dumps({"active_jobs": active, "results": [result]}, indent=2))
        return
    if args.batch == "remaining":
        failed_gate = [
            row["run_id"] for row in canaries if states[row["run_id"]] != "JOB_STATE_SUCCEEDED"
        ]
        if failed_gate:
            raise RuntimeError(f"Confirmatory canary has not passed: {failed_gate}")
        selected = [row for row in rows if not is_canary(row)]
    else:
        selected = canaries
    results = []
    for row in selected:
        state = states[row["run_id"]]
        if state != "NOT_SUBMITTED":
            results.append({"run_id": row["run_id"], "action": "skip", "state": state})
            continue
        if active >= args.max_concurrency:
            results.append({"run_id": row["run_id"], "action": "defer"})
            continue
        config = config_for_row(spec, base, row)
        job_spec = build_job_spec(
            config,
            gcp,
            "confirmatory",
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
    print(
        json.dumps(
            {
                "batch": args.batch,
                "offline": args.offline,
                "active_jobs": active,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Produce the hard-gate READY/NOT_READY report for confirmatory submission."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from gae_credit.cloud.config import image_digest, load_gcp_config
from gae_credit.confirmatory import read_manifest
from gae_credit.storage.gcs import GCSArtifactStore


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/ppo_confirmatory_v1.parquet")
    parser.add_argument("--pilot-qc", default="analysis/pilot_qc_data.json")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--freeze-tag", default="experiment-v1")
    parser.add_argument("--regional-cpu-quota", type=int, required=True)
    parser.add_argument("--concurrent-job-quota", type=int, required=True)
    parser.add_argument("--project", default="gae-experiment-507805")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/day6"))
    args = parser.parse_args()
    checks, errors = {}, []

    def check(name, condition, detail):
        checks[name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            errors.append(f"{name}: {detail}")

    try:
        rows = read_manifest(args.manifest)
        manifest_sha = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
        check("manifest", len(rows) == 160, f"{len(rows)} rows, sha256={manifest_sha}")
    except Exception as error:
        rows, manifest_sha = [], None
        check("manifest", False, str(error))
    head = command("git", "rev-parse", "HEAD")
    try:
        tag = command("git", "rev-list", "-n", "1", args.freeze_tag)
    except subprocess.CalledProcessError:
        tag = ""
    check("git_clean", command("git", "status", "--porcelain") == "", head)
    tag_is_ancestor = (
        bool(tag)
        and subprocess.run(
            ["git", "merge-base", "--is-ancestor", tag, head], check=False
        ).returncode
        == 0
    )
    check("freeze_tag", tag_is_ancestor, f"tag={tag or 'missing'}, head={head}")
    check("dependency_lock", Path("uv.lock").is_file(), "uv.lock exists")
    try:
        image_digest(args.image_uri)
        check("image_digest", True, args.image_uri)
    except ValueError as error:
        check("image_digest", False, str(error))
    qc = json.loads(Path(args.pilot_qc).read_text())
    check(
        "pilot_complete",
        qc.get("complete") and qc.get("valid_runs") == 24 and qc.get("invalid_runs") == 0,
        f"valid={qc.get('valid_runs')}, invalid={qc.get('invalid_runs')}",
    )
    if rows:
        check("seed_separation", {row["seed"] for row in rows} == set(range(10)), "seeds 0–9")
        check(
            "single_image", {row["image_digest"] for row in rows} == {args.image_uri}, "one image"
        )
        check("git_identity", {row["git_sha"] for row in rows} == {tag}, "frozen tag commit")
    gcp = load_gcp_config("configs/cloud/gcp.yaml").validate(deployed=True)
    check("gcp_target", gcp.project_id == args.project, f"{gcp.project_id}/{gcp.region}")
    try:
        account = command(
            "gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"
        )
        check("gcp_auth", bool(account), account or "no active account")
    except subprocess.CalledProcessError as error:
        check("gcp_auth", False, str(error))
    if rows:
        try:
            collisions = []
            for row in rows:
                store = GCSArtifactStore(row["gcs_output_uri"], project=args.project)
                if any(
                    store.exists(key) for key in ("_SUCCESS", "_SUBMISSION.json", "_IDENTITY.json")
                ):
                    collisions.append(row["run_id"])
            check(
                "gcs_collisions",
                not collisions,
                "all 160 output paths are unused"
                if not collisions
                else f"{len(collisions)} collisions",
            )
        except Exception as error:
            check("gcs_collisions", False, str(error))
    try:
        command(
            "gcloud",
            "artifacts",
            "docker",
            "images",
            "describe",
            args.image_uri,
            "--project",
            args.project,
        )
        check("artifact_image", True, "immutable image is readable")
    except subprocess.CalledProcessError as error:
        check("artifact_image", False, error.stderr.strip() or str(error))
    max_jobs = args.regional_cpu_quota // 4
    check(
        "quota",
        max_jobs >= 8,
        f"CPU={args.regional_cpu_quota}, pipelines={args.concurrent_job_quota}, max_jobs={max_jobs}",
    )
    pytest = subprocess.run([".venv/bin/pytest", "-q"], capture_output=True, text=True)
    check("tests", pytest.returncode == 0, pytest.stdout.strip().splitlines()[-1])
    median = float(qc["median_runtime_seconds"])
    full_seconds = median * 4
    estimate = {
        "pilot_median_seconds": median,
        "estimated_seconds_per_full_run": full_seconds,
        "estimated_total_compute_hours": full_seconds * 160 / 3600,
        "operational_cap_jobs": min(10, max_jobs),
        "requested_target_jobs": 32,
    }
    status = "READY" if not errors else "NOT_READY"
    report = {
        "schema_version": 1,
        "head_sha": head,
        "freeze_sha": tag,
        "freeze_tag": args.freeze_tag,
        "image_uri": args.image_uri,
        "manifest_sha256": manifest_sha,
        "checks": checks,
        "errors": errors,
        "runtime_estimate": estimate,
        "billing_check": "waived by user on 2026-09-09",
        "status": status,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    lines = ["# Day 6 preflight", "", f"Frozen commit: `{tag}`", ""]
    lines.extend(
        f"- [{'x' if value['passed'] else ' '}] {name}: {value['detail']}"
        for name, value in checks.items()
    )
    lines.extend(
        ["", f"Operational concurrency cap: {estimate['operational_cap_jobs']}", "", status]
    )
    (args.output_dir / "preflight_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if status == "READY" else 1)


if __name__ == "__main__":
    main()

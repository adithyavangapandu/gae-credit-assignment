"""Create the hard-gate report for the 60-run VPG replication."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from gae_credit.cloud.config import image_digest, load_gcp_config
from gae_credit.replication import read_manifest
from gae_credit.storage.gcs import GCSArtifactStore


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/vpg_replication_v1.parquet")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--freeze-tag", default="vpg-replication-v1")
    parser.add_argument("--regional-cpu-quota", type=int, default=42)
    parser.add_argument("--concurrent-job-quota", type=int, default=200)
    parser.add_argument("--project", default="gae-experiment-507805")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/vpg"))
    args = parser.parse_args()
    checks, errors = {}, []

    def check(name, condition, detail):
        checks[name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            errors.append(f"{name}: {detail}")

    rows = read_manifest(args.manifest)
    digest = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    check("six_cells", len(rows) == 60, f"{len(rows)} rows across six cells")
    check("ten_seeds", {row["seed"] for row in rows} == set(range(10)), "seeds 0-9")
    check("training_budget", {row["training_steps"] for row in rows} == {1_000_000}, "1,000,000")
    check("evaluation_seed", {row["evaluation_seed"] for row in rows} == {20260905}, "20260905")
    head = command("git", "rev-parse", "HEAD")
    tag = command("git", "rev-list", "-n", "1", args.freeze_tag)
    clean = command("git", "status", "--porcelain") == ""
    tag_is_ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", tag, head], check=False).returncode
        == 0
    )
    check("git_clean", clean, head)
    check("freeze_tag", tag_is_ancestor and {row["git_sha"] for row in rows} == {tag}, f"tag={tag}")
    try:
        image_digest(args.image_uri)
        check(
            "immutable_image",
            {row["image_digest"] for row in rows} == {args.image_uri},
            args.image_uri,
        )
    except ValueError as error:
        check("immutable_image", False, str(error))
    gcp = load_gcp_config("configs/cloud/gcp.yaml").validate(deployed=True)
    check("gcp_target", gcp.project_id == args.project, f"{gcp.project_id}/{gcp.region}")
    account = command("gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)")
    check("gcp_auth", bool(account), account or "no active account")
    collisions = []
    for row in rows:
        store = GCSArtifactStore(row["gcs_output_uri"], project=args.project)
        if any(store.exists(key) for key in ("_SUCCESS", "_SUBMISSION.json", "_IDENTITY.json")):
            collisions.append(row["run_id"])
    check("gcs_paths_unused", not collisions, f"{len(collisions)} collisions")
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
    max_jobs = args.regional_cpu_quota // 4
    check(
        "quota",
        max_jobs >= 6,
        f"CPU={args.regional_cpu_quota}, pipelines={args.concurrent_job_quota}, max_jobs={max_jobs}",
    )
    tests = subprocess.run([".venv/bin/pytest", "-q"], capture_output=True, text=True)
    check("tests", tests.returncode == 0, tests.stdout.strip().splitlines()[-1])
    status = "READY" if not errors else "NOT_READY"
    report = {
        "schema_version": 1,
        "head_sha": head,
        "freeze_sha": tag,
        "freeze_tag": args.freeze_tag,
        "manifest_sha256": digest,
        "image_uri": args.image_uri,
        "operational_concurrency_cap": min(10, max_jobs),
        "checks": checks,
        "errors": errors,
        "status": status,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    lines = ["# VPG replication preflight", ""]
    lines.extend(
        f"- [{'x' if value['passed'] else ' '}] {name}: {value['detail']}"
        for name, value in checks.items()
    )
    lines.extend(["", status])
    (args.output_dir / "preflight_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if status == "READY" else 1)


if __name__ == "__main__":
    main()

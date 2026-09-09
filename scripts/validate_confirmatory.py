"""Download and validate confirmatory artifacts without computing treatment contrasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from gae_credit.cloud.config import image_digest
from gae_credit.confirmatory import config_for_row, load_confirmatory_spec, read_manifest
from gae_credit.logging import validate_run
from gae_credit.storage.download import download_run
from gae_credit.storage.gcs import GCSArtifactStore


def validate_confirmatory_run(path, row, spec, base):
    report = validate_run(path)
    config = config_for_row(spec, base, row)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("phase") != "confirmatory":
        raise ValueError("run phase is not confirmatory")
    if manifest.get("config_hash") != row["config_hash"]:
        raise ValueError("manifest config hash does not match frozen matrix")
    if manifest.get("git_commit") != row["git_sha"] or manifest.get("git_dirty") is not False:
        raise ValueError("run does not use the frozen clean source commit")
    if manifest.get("image_digest") != image_digest(row["image_digest"]):
        raise ValueError("run does not use the frozen image digest")
    if manifest.get("gcp_project") != "gae-experiment-507805":
        raise ValueError("run used the wrong GCP project")
    if manifest.get("gcp_region") != "us-central1":
        raise ValueError("run used the wrong GCP region")
    if manifest.get("environment_version") != "pendulum-poststep-v1":
        raise ValueError("run used an unexpected environment version")
    updates = pq.read_table(path / "updates.parquet").to_pandas()
    if not np.isfinite(updates.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("training metrics contain nonfinite values")
    expected_updates = config.training.total_env_steps // (
        config.training.episodes_per_rollout * config.environment.max_steps
    )
    if len(updates) != expected_updates:
        raise ValueError("run terminated before the frozen training budget")
    return {**report, "status": "valid", "runtime_seconds": float(updates.elapsed_seconds.iloc[-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/ppo_confirmatory_v1.parquet")
    parser.add_argument("--study-config", default="configs/confirmatory/study_v1.yaml")
    parser.add_argument("--download-root", type=Path, default=Path("runs/confirmatory_downloads"))
    parser.add_argument("--project")
    parser.add_argument("--fetch-gcs", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/day6/confirmatory_validation.json")
    )
    args = parser.parse_args()
    if args.fetch_gcs and not args.project:
        parser.error("--project is required with --fetch-gcs")
    rows = read_manifest(args.manifest)
    spec, base = load_confirmatory_spec(args.study_config)
    reports = []
    for row in rows:
        destination = args.download_root / row["run_id"]
        if args.fetch_gcs and not destination.exists():
            store = GCSArtifactStore(row["gcs_output_uri"], project=args.project)
            if store.exists("_SUCCESS"):
                download_run(store, destination)
        if not destination.exists():
            reports.append({"run_id": row["run_id"], "status": "missing"})
            continue
        try:
            reports.append(validate_confirmatory_run(destination, row, spec, base))
        except Exception as error:
            reports.append({"run_id": row["run_id"], "status": "invalid", "error": str(error)})
    result = {
        "purpose": "confirmatory integrity only; no treatment comparisons",
        "expected_runs": 160,
        "valid_runs": sum(item["status"] == "valid" for item in reports),
        "missing_runs": sum(item["status"] == "missing" for item in reports),
        "invalid_runs": sum(item["status"] == "invalid" for item in reports),
        "complete": all(item["status"] == "valid" for item in reports),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, indent=2))
    if args.require_complete and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

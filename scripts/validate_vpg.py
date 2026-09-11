"""Download and validate all 60 VPG replication runs without outcome analysis."""

import argparse
import json
from pathlib import Path

from gae_credit.cloud.config import image_digest
from gae_credit.logging import validate_run
from gae_credit.replication import config_for_row, load_replication_spec, read_manifest
from gae_credit.storage.download import download_run
from gae_credit.storage.gcs import GCSArtifactStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/vpg_replication_v1.parquet")
    parser.add_argument("--study-config", default="configs/replication/vpg_v1.yaml")
    parser.add_argument("--download-root", type=Path, default=Path("runs/vpg_downloads"))
    parser.add_argument("--project", default="gae-experiment-507805")
    parser.add_argument("--fetch-gcs", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/vpg/validation.json"))
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    spec, base = load_replication_spec(args.study_config)
    reports = []
    for row in rows:
        path = args.download_root / row["run_id"]
        if args.fetch_gcs and not path.exists():
            store = GCSArtifactStore(row["gcs_output_uri"], project=args.project)
            if store.exists("_SUCCESS"):
                download_run(store, path)
        if not path.exists():
            reports.append({"run_id": row["run_id"], "status": "missing"})
            continue
        try:
            report = validate_run(path)
            config = config_for_row(spec, base, row)
            manifest = json.loads((path / "manifest.json").read_text())
            expected = {
                "phase": "replication",
                "config_hash": row["config_hash"],
                "git_commit": row["git_sha"],
                "git_dirty": False,
                "image_digest": image_digest(row["image_digest"]),
                "environment_version": "pendulum-poststep-v1",
                "gcp_project": args.project,
                "gcp_region": "us-central1",
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError("VPG artifact identity mismatch")
            if config.evaluation.seed != row["evaluation_seed"]:
                raise ValueError("VPG evaluation seed mismatch")
            reports.append({**report, "status": "valid"})
        except Exception as error:
            reports.append({"run_id": row["run_id"], "status": "invalid", "error": str(error)})
    result = {
        "expected_cells": 6,
        "expected_runs": 60,
        "valid_runs": sum(row["status"] == "valid" for row in reports),
        "missing_runs": sum(row["status"] == "missing" for row in reports),
        "invalid_runs": sum(row["status"] == "invalid" for row in reports),
        "reports": reports,
    }
    result["complete"] = result["valid_runs"] == 60
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, indent=2))
    if args.require_complete and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

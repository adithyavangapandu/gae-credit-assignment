"""Download and validate confirmatory artifacts without computing treatment contrasts."""

import argparse
import json
from pathlib import Path

from gae_credit.analysis.integrity import validate_confirmatory_run
from gae_credit.confirmatory import load_confirmatory_spec, read_manifest
from gae_credit.storage.download import download_run
from gae_credit.storage.gcs import GCSArtifactStore


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

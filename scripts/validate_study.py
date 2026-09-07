"""Download completed pilot runs and apply study-level quality control."""

import argparse
import json
from pathlib import Path

from gae_credit.pilot import load_pilot_spec, read_matrix
from gae_credit.storage.download import download_run
from gae_credit.storage.gcs import GCSArtifactStore
from gae_credit.study_validation import validate_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="data/manifests/pilot_run_matrix.parquet")
    parser.add_argument("--study-config", default="configs/pilot/study_v1.yaml")
    parser.add_argument("--download-root", type=Path, default=Path("runs/pilot_downloads"))
    parser.add_argument("--project")
    parser.add_argument("--fetch-gcs", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("analysis/pilot_qc_data.json"))
    args = parser.parse_args()
    rows = read_matrix(args.matrix)
    spec, base = load_pilot_spec(args.study_config)
    if args.fetch_gcs:
        if not args.project:
            parser.error("--project is required with --fetch-gcs")
        args.download_root.mkdir(parents=True, exist_ok=True)
        for row in rows:
            destination = args.download_root / row["run_id"]
            if destination.exists():
                continue
            store = GCSArtifactStore(row["expected_gcs_prefix"], project=args.project)
            if store.exists("_SUCCESS"):
                download_run(store, destination)
    result = validate_study(rows, spec, base, args.download_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, indent=2))
    if args.require_complete and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

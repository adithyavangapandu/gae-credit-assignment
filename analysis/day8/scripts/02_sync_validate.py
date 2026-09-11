"""Download completed GCS bundles and store run-level integrity results."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from gae_credit.analysis.sync import download_analysis_bundle, validate_analysis_bundle
from gae_credit.confirmatory import read_manifest
from gae_credit.storage.gcs import GCSArtifactStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/ppo_confirmatory_v1.parquet")
    parser.add_argument("--project", default="gae-experiment-507805")
    parser.add_argument("--raw-root", type=Path, default=Path("analysis/day8/data/raw_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/day8/data"))
    parser.add_argument("--workers", type=int, default=12, help="Object downloads per run")
    parser.add_argument("--run-workers", type=int, default=4, help="Runs synchronized in parallel")
    parser.add_argument("--run-id", action="append", help="Sync only these run IDs")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if args.run_id:
        requested = set(args.run_id)
        rows = [row for row in rows if row["run_id"] in requested]
        if {row["run_id"] for row in rows} != requested:
            raise ValueError("A requested run ID is absent from the frozen manifest")
    args.raw_root.mkdir(parents=True, exist_ok=True)

    def synchronize(row):
        destination = args.raw_root / row["run_id"]
        store = GCSArtifactStore(row["gcs_output_uri"], project=args.project)
        if not destination.exists() and store.exists("_SUCCESS"):
            download_analysis_bundle(store, destination, row, workers=args.workers)
        if not destination.exists():
            return {"run_id": row["run_id"], "status": "missing"}
        try:
            return validate_analysis_bundle(destination, row)
        except Exception as error:
            return {"run_id": row["run_id"], "status": "invalid", "error": str(error)}

    with ThreadPoolExecutor(max_workers=args.run_workers) as executor:
        reports = list(executor.map(synchronize, rows))
    frame = pd.DataFrame(reports)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "validation_registry.parquet", index=False)
    counts = {str(key): int(value) for key, value in frame.status.value_counts().items()}
    summary = {
        "expected_runs": 160,
        "selected_runs": len(rows),
        "status_counts": counts,
        "complete": not args.run_id and counts.get("valid", 0) == 160,
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

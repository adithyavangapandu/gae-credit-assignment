"""Discover every frozen PPO run prefix and store a GCS inventory."""

import argparse
import json
from pathlib import Path

import pandas as pd

from gae_credit.analysis import reward_condition
from gae_credit.cloud.config import load_gcp_config
from gae_credit.confirmatory import read_manifest
from gae_credit.storage.gcs import GCSArtifactStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/ppo_confirmatory_v1.parquet")
    parser.add_argument("--gcp-config", default="configs/cloud/gcp.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/day8/data"))
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    gcp = load_gcp_config(args.gcp_config).validate(deployed=True)
    inventory = []
    for row in rows:
        store = GCSArtifactStore(row["gcs_output_uri"], project=gcp.project_id)
        markers = {
            key: store.exists(key)
            for key in ("_SUCCESS", "_SUBMISSION.json", "_LATEST_CHECKPOINT.json")
        }
        state = (
            "complete"
            if markers["_SUCCESS"]
            else "submitted"
            if markers["_SUBMISSION.json"]
            else "absent"
        )
        receipt = (
            json.loads(store.read_bytes("_SUBMISSION.json")) if markers["_SUBMISSION.json"] else {}
        )
        inventory.append(
            {
                **row,
                "reward_condition": reward_condition(row["reward_type"], row["reward_delay"]),
                "gcs_state": state,
                "has_checkpoint": markers["_LATEST_CHECKPOINT.json"],
                "job_resource_name": receipt.get("job_resource_name"),
                "attempt_count": receipt.get("attempt_count"),
            }
        )
    frame = pd.DataFrame(inventory).sort_values("task_id")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "gcs_inventory.parquet", index=False)
    counts = {str(key): int(value) for key, value in frame.gcs_state.value_counts().items()}
    summary = {
        "expected_runs": 160,
        "state_counts": counts,
        "complete": counts.get("complete", 0) == 160,
    }
    (args.output_dir / "gcs_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Build analysis-ready metric tables from validated local run bundles."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from gae_credit.analysis import reward_condition
from gae_credit.analysis.sync import validate_analysis_bundle
from gae_credit.confirmatory import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/ppo_confirmatory_v1.parquet")
    parser.add_argument("--raw-root", type=Path, default=Path("analysis/day8/data/raw_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/day8/data/core"))
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    tables = {name: [] for name in ("evaluations", "updates", "episodes")}
    registry = []
    for row in rows:
        path = args.raw_root / row["run_id"]
        if not path.exists():
            registry.append({**row, "status": "missing"})
            continue
        try:
            validate_analysis_bundle(path, row)
            condition = reward_condition(row["reward_type"], row["reward_delay"])
            metadata = {
                "task_id": row["task_id"],
                "reward_condition": condition,
                "gae_horizon": row["gae_horizon"],
                "seed": row["seed"],
            }
            for name in tables:
                frame = pq.read_table(path / f"{name}.parquet").to_pandas()
                for key, value in metadata.items():
                    frame[key] = value
                tables[name].append(frame)
            registry.append({**row, "reward_condition": condition, "status": "valid"})
        except Exception as error:
            registry.append({**row, "status": "invalid", "error": str(error)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registry).to_parquet(args.output_dir / "run_registry.parquet", index=False)
    for name, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_parquet(
            args.output_dir / f"{name}.parquet", index=False
        ) if frames else pd.DataFrame().to_parquet(args.output_dir / f"{name}.parquet", index=False)
    valid = sum(row["status"] == "valid" for row in registry)
    summary = {"expected_runs": 160, "valid_runs": valid, "complete": valid == 160}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Reduce every complete run to the frozen primary and secondary outcomes."""

import argparse
import json
from pathlib import Path

import pandas as pd

from gae_credit.analysis import load_analysis_config, run_outcomes
from gae_credit.analysis.day8 import pre_reward_advantage_magnitude


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", default="analysis/day8/analysis_v1.yaml")
    parser.add_argument("--core-dir", type=Path, default=Path("analysis/day8/data/core"))
    parser.add_argument("--raw-root", type=Path, default=Path("analysis/day8/data/raw_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/day8/results"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = load_analysis_config(args.analysis_config)
    registry = pd.read_parquet(args.core_dir / "run_registry.parquet")
    valid = registry.loc[registry.status == "valid"]
    if len(valid) != config["expected_runs"] and not args.allow_incomplete:
        raise RuntimeError("Refusing outcome analysis until all 160 runs are valid")
    evaluations = pd.read_parquet(args.core_dir / "evaluations.parquet")
    updates = pd.read_parquet(args.core_dir / "updates.parquet")
    rows = []
    for run_id, evaluation_group in evaluations.groupby("run_id", sort=True):
        update_group = updates.loc[updates.run_id == run_id]
        identity = evaluation_group.iloc[0]
        pre_reward, pre_reward_count = pre_reward_advantage_magnitude(
            args.raw_root / run_id / "diagnostics", identity.gae_horizon
        )
        rows.append(
            {
                "run_id": run_id,
                "reward_condition": identity.reward_condition,
                "gae_horizon": identity.gae_horizon,
                "seed": int(identity.seed),
                **run_outcomes(evaluation_group, update_group, config),
                "pre_reward_advantage_magnitude": pre_reward,
                "pre_reward_transitions": pre_reward_count,
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("steps_to_success", "first_successful_evaluation_step"):
        frame[column] = frame[column].astype("Int64")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "run_outcomes.parquet", index=False)
    summary = {
        "analysis_protocol": str(args.analysis_config),
        "runs": len(frame),
        "complete": len(frame) == config["expected_runs"],
        "interim_override": bool(args.allow_incomplete),
    }
    (args.output_dir / "run_outcomes_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

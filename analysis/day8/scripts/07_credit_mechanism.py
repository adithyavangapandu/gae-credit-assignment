"""Stream transition diagnostics into reward-distance mechanism summaries."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gae_credit.analysis import distance_to_next_reward, load_analysis_config, reward_condition
from gae_credit.analysis.day8 import discounted_returns, distance_bin
from gae_credit.confirmatory import read_manifest


def empty_stats():
    return defaultdict(float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", default="analysis/day8/analysis_v1.yaml")
    parser.add_argument("--manifest", default="data/manifests/ppo_confirmatory_v1.parquet")
    parser.add_argument("--raw-root", type=Path, default=Path("analysis/day8/data/raw_runs"))
    parser.add_argument("--core-dir", type=Path, default=Path("analysis/day8/data/core"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/day8/results"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = load_analysis_config(args.analysis_config)
    registry = pd.read_parquet(args.core_dir / "run_registry.parquet")
    valid_ids = set(registry.loc[registry.status == "valid", "run_id"])
    if len(valid_ids) != config["expected_runs"] and not args.allow_incomplete:
        raise RuntimeError("Refusing mechanism analysis until all 160 runs are valid")
    rows = [row for row in read_manifest(args.manifest) if row["run_id"] in valid_ids]
    aggregates = defaultdict(empty_stats)
    run_pre_reward = defaultdict(lambda: [0.0, 0])
    for row in rows:
        path = args.raw_root / row["run_id"]
        updates = pd.read_parquet(path / "updates.parquet").set_index("update_index")
        horizon = row["gae_horizon"]
        condition = reward_condition(row["reward_type"], row["reward_delay"])
        for diagnostic in sorted((path / "diagnostics").glob("update-*.npz")):
            update_index = int(diagnostic.stem.split("-")[1])
            env_steps = update_index * 1000
            step_bin = math.ceil(env_steps / config["credit_step_bin"]) * config["credit_step_bin"]
            with np.load(diagnostic, allow_pickle=False) as arrays:
                rewards = arrays["rewards"].astype(float)
                values = arrays["values"][:-1].astype(float)
                advantages = arrays[f"advantages_{horizon}"].astype(float)
                full = arrays["advantages_full"].astype(float)
                later_returns = discounted_returns(rewards, 0.995)
                distances = distance_to_next_reward(rewards)
                labels = distance_bin(distances, config["distance_bins"])
                actor_gradient = float(updates.loc[update_index, "actor_grad_norm"])
                for label in set(labels):
                    selected = labels == label
                    x = advantages[selected]
                    y = later_returns[selected]
                    critic_error = y - values[selected]
                    key = (row["run_id"], condition, horizon, row["seed"], step_bin, label)
                    stats = aggregates[key]
                    stats["n"] += len(x)
                    stats["sum_abs_advantage"] += np.abs(x).sum()
                    stats["sum_positive_advantage"] += (x > 0).sum()
                    stats["sum_reference_error"] += np.abs(x - full[selected]).sum()
                    stats["sum_actor_gradient"] += actor_gradient * len(x)
                    stats["sum_critic_squared_error"] += np.square(critic_error).sum()
                    stats["sum_meaningful"] += (
                        np.abs(x) >= config["meaningful_advantage_threshold"]
                    ).sum()
                    stats["sum_x"] += x.sum()
                    stats["sum_y"] += y.sum()
                    stats["sum_x2"] += np.square(x).sum()
                    stats["sum_y2"] += np.square(y).sum()
                    stats["sum_xy"] += (x * y).sum()
                pre_reward = np.isfinite(distances) & (distances > 0)
                run_pre_reward[row["run_id"]][0] += np.abs(advantages[pre_reward]).sum()
                run_pre_reward[row["run_id"]][1] += int(pre_reward.sum())
    output = []
    for key, stats in aggregates.items():
        run_id, condition, horizon, seed, step_bin, label = key
        n = stats["n"]
        covariance = stats["sum_xy"] - stats["sum_x"] * stats["sum_y"] / n
        variance_x = stats["sum_x2"] - stats["sum_x"] ** 2 / n
        variance_y = stats["sum_y2"] - stats["sum_y"] ** 2 / n
        correlation = (
            covariance / math.sqrt(variance_x * variance_y)
            if variance_x > 0 and variance_y > 0
            else np.nan
        )
        output.append(
            {
                "run_id": run_id,
                "reward_condition": condition,
                "gae_horizon": horizon,
                "seed": seed,
                "env_step_bin": step_bin,
                "reward_distance_bin": label,
                "transitions": int(n),
                "mean_absolute_raw_advantage": stats["sum_abs_advantage"] / n,
                "positive_advantage_fraction": stats["sum_positive_advantage"] / n,
                "mean_absolute_full_reference_error": stats["sum_reference_error"] / n,
                "mean_actor_gradient_norm": stats["sum_actor_gradient"] / n,
                "advantage_later_return_correlation": correlation,
                "critic_prediction_rmse": math.sqrt(stats["sum_critic_squared_error"] / n),
                "meaningful_signal_fraction": stats["sum_meaningful"] / n,
            }
        )
    credit_runs = []
    for row in rows:
        total, count = run_pre_reward[row["run_id"]]
        credit_runs.append(
            {
                "run_id": row["run_id"],
                "reward_condition": reward_condition(row["reward_type"], row["reward_delay"]),
                "gae_horizon": row["gae_horizon"],
                "seed": row["seed"],
                "pre_reward_advantage_magnitude": total / count if count else np.nan,
                "pre_reward_transitions": count,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_parquet(
        args.output_dir / "credit_distance_summaries.parquet", index=False
    )
    pd.DataFrame(credit_runs).to_parquet(
        args.output_dir / "credit_run_outcomes.parquet", index=False
    )
    report = {
        "runs": len(rows),
        "groups": len(output),
        "interim_override": bool(args.allow_incomplete),
    }
    (args.output_dir / "credit_analysis_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

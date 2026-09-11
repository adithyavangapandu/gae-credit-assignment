"""Estimate paired reward-by-horizon interactions and a seed fixed-effect model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gae_credit.analysis import bootstrap_interval, load_analysis_config


def fixed_effect_model(frame, metric):
    data = frame[[metric, "reward_condition", "gae_horizon", "seed"]].dropna().copy()
    data["reward_condition"] = pd.Categorical(
        data.reward_condition,
        categories=["dense", "delayed_8", "delayed_32", "sparse"],
    )
    data["gae_horizon"] = pd.Categorical(data.gae_horizon, categories=["full", "1", "3", "16"])
    reward = pd.get_dummies(data.reward_condition, prefix="reward", drop_first=True, dtype=float)
    horizon = pd.get_dummies(data.gae_horizon, prefix="horizon", dtype=float).drop(
        columns=["horizon_full"], errors="ignore"
    )
    interaction = {
        f"{reward_column}:{horizon_column}": reward[reward_column] * horizon[horizon_column]
        for reward_column in reward
        for horizon_column in horizon
    }
    seeds = pd.get_dummies(data.seed, prefix="seed", drop_first=True, dtype=float)
    design = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            reward,
            horizon,
            pd.DataFrame(interaction),
            seeds,
        ],
        axis=1,
    )
    x = design.to_numpy(dtype=float)
    y = data[metric].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    degrees = len(y) - rank
    variance = float(residuals @ residuals / degrees) if degrees > 0 else np.nan
    covariance = variance * np.linalg.pinv(x.T @ x)
    standard_errors = np.sqrt(np.diag(covariance))
    return pd.DataFrame(
        {
            "metric": metric,
            "term": design.columns,
            "estimate": coefficients,
            "standard_error": standard_errors,
            "n": len(y),
            "rank": rank,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", default="analysis/day8/analysis_v1.yaml")
    parser.add_argument("--outcomes", default="analysis/day8/results/run_outcomes.parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/day8/results"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = load_analysis_config(args.analysis_config)
    outcomes = pd.read_parquet(args.outcomes)
    if len(outcomes) != config["expected_runs"] and not args.allow_incomplete:
        raise RuntimeError("Refusing interaction analysis until all 160 outcomes exist")
    metrics = [
        config["primary_metric"],
        "final_evaluation_return",
        "final_upright_fraction",
        "upright_fraction_auc",
        "final_success_probability",
    ]
    seed_rows, summary_rows, models = [], [], []
    pivot = outcomes.pivot_table(
        index="seed", columns=["reward_condition", "gae_horizon"], values=metrics
    )
    index = 0
    for metric in metrics:
        models.append(fixed_effect_model(outcomes, metric))
        for reward in ("delayed_8", "delayed_32", "sparse"):
            for horizon in ("1", "3", "16"):
                required = [
                    (metric, reward, "full"),
                    (metric, reward, horizon),
                    (metric, "dense", "full"),
                    (metric, "dense", horizon),
                ]
                available = pivot.dropna(subset=required)
                values = (
                    available[(metric, reward, "full")]
                    - available[(metric, reward, horizon)]
                    - available[(metric, "dense", "full")]
                    + available[(metric, "dense", horizon)]
                )
                for seed, value in values.items():
                    seed_rows.append(
                        {
                            "interaction_id": index,
                            "metric": metric,
                            "reward_condition": reward,
                            "gae_horizon": horizon,
                            "seed": int(seed),
                            "difference_in_differences": float(value),
                        }
                    )
                if len(values):
                    low, high = bootstrap_interval(
                        values,
                        config["bootstrap_samples"],
                        config["confidence_level"],
                        config["bootstrap_seed"] + index,
                    )
                    summary_rows.append(
                        {
                            "interaction_id": index,
                            "metric": metric,
                            "reward_condition": reward,
                            "gae_horizon": horizon,
                            "n_pairs": len(values),
                            "mean_interaction": float(values.mean()),
                            "median_interaction": float(values.median()),
                            "ci_lower": low,
                            "ci_upper": high,
                            "is_preregistered_primary": metric == "evaluation_auc"
                            and reward == "delayed_32"
                            and horizon == "3",
                        }
                    )
                index += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seed_rows).to_parquet(
        args.output_dir / "interaction_seed_values.parquet", index=False
    )
    pd.DataFrame(summary_rows).to_parquet(
        args.output_dir / "interaction_summaries.parquet", index=False
    )
    pd.concat(models, ignore_index=True).to_parquet(
        args.output_dir / "interaction_fixed_effect_coefficients.parquet", index=False
    )
    monotonic = []
    for reward, group in outcomes.groupby("reward_condition", sort=True):
        horizon_auc = group.pivot(index="seed", columns="gae_horizon", values="evaluation_auc")
        for seed, values in horizon_auc.dropna(subset=["1", "3", "16", "full"]).iterrows():
            monotonic.append(
                {
                    "reward_condition": reward,
                    "seed": int(seed),
                    "h1": float(values["1"]),
                    "h3": float(values["3"]),
                    "h16": float(values["16"]),
                    "full": float(values["full"]),
                    "monotonic_non_decreasing": bool(
                        values["1"] <= values["3"] <= values["16"] <= values["full"]
                    ),
                }
            )
    pd.DataFrame(monotonic).to_parquet(
        args.output_dir / "horizon_response_monotonicity.parquet", index=False
    )
    report = {"interactions": len(summary_rows), "interim_override": bool(args.allow_incomplete)}
    (args.output_dir / "interaction_analysis_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

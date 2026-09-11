"""Compute seed-paired full-minus-truncated contrasts and bootstrap intervals."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gae_credit.analysis import bootstrap_interval, load_analysis_config
from gae_credit.analysis.day8 import holm_adjust, sign_flip_pvalue

LOWER_IS_BETTER = {"steps_to_success", "first_successful_evaluation_step"}


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
        raise RuntimeError("Refusing paired analysis until all 160 outcomes exist")
    excluded = {"run_id", "reward_condition", "gae_horizon", "seed", "pre_reward_transitions"}
    metrics = [column for column in outcomes if column not in excluded]
    individual = []
    summaries = []
    comparison_index = 0
    for reward, reward_rows in outcomes.groupby("reward_condition", sort=True):
        full = reward_rows.loc[reward_rows.gae_horizon == "full"].set_index("seed")
        for horizon in ("1", "3", "16"):
            truncated = reward_rows.loc[reward_rows.gae_horizon == horizon].set_index("seed")
            for metric in metrics:
                paired = (
                    full[[metric]]
                    .join(truncated[[metric]], how="inner", lsuffix="_full", rsuffix="_truncated")
                    .dropna()
                )
                differences = paired[f"{metric}_full"] - paired[f"{metric}_truncated"]
                for seed, difference in differences.items():
                    individual.append(
                        {
                            "comparison_id": comparison_index,
                            "reward_condition": reward,
                            "gae_horizon": horizon,
                            "metric": metric,
                            "seed": int(seed),
                            "full_minus_truncated": float(difference),
                        }
                    )
                values = differences.to_numpy(dtype=float)
                if len(values):
                    low, high = bootstrap_interval(
                        values,
                        config["bootstrap_samples"],
                        config["confidence_level"],
                        config["bootstrap_seed"] + comparison_index,
                    )
                    standard_deviation = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                    )
                    effect_size = (
                        float(values.mean() / standard_deviation) if standard_deviation else np.nan
                    )
                    full_favored = values < 0 if metric in LOWER_IS_BETTER else values > 0
                    summaries.append(
                        {
                            "comparison_id": comparison_index,
                            "reward_condition": reward,
                            "gae_horizon": horizon,
                            "metric": metric,
                            "n_pairs": len(values),
                            "n_censored_or_missing": 10 - len(values),
                            "mean_difference": float(values.mean()),
                            "median_difference": float(np.median(values)),
                            "ci_lower": low,
                            "ci_upper": high,
                            "paired_standardized_effect": effect_size,
                            "seeds_favoring_full": int(full_favored.sum()),
                            "seeds_favoring_truncated": int((~full_favored & (values != 0)).sum()),
                            "ties": int((values == 0).sum()),
                            "sign_flip_pvalue": sign_flip_pvalue(values),
                            "is_dense_h3_auc_noninferiority": metric == "evaluation_auc"
                            and reward == "dense"
                            and horizon == "3",
                            "noninferiority_pass": (
                                bool(
                                    high
                                    < -config["planned_comparisons"]["dense"][
                                        "noninferiority_margin"
                                    ]
                                )
                                if metric == "evaluation_auc"
                                and reward == "dense"
                                and horizon == "3"
                                else None
                            ),
                        }
                    )
                comparison_index += 1
    summary_frame = pd.DataFrame(summaries)
    if len(summary_frame):
        summary_frame["holm_pvalue"] = holm_adjust(summary_frame.sign_flip_pvalue)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(individual).to_parquet(
        args.output_dir / "paired_seed_differences.parquet", index=False
    )
    summary_frame.to_parquet(args.output_dir / "paired_summaries.parquet", index=False)
    report = {"comparisons": len(summary_frame), "interim_override": bool(args.allow_incomplete)}
    (args.output_dir / "paired_analysis_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

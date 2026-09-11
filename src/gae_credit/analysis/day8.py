"""Pure functions for the preregistered PPO analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_analysis_config(path="analysis/day8/analysis_v1.yaml"):
    config = yaml.safe_load(Path(path).read_text())
    required = {
        "schema_version",
        "primary_algorithm",
        "primary_metric",
        "secondary_metrics",
        "pairing_variable",
        "confidence_level",
        "bootstrap_samples",
        "bootstrap_seed",
        "expected_runs",
        "training_budget",
        "planned_comparisons",
        "primary_interaction",
        "multiplicity",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Analysis config is missing fields: {sorted(missing)}")
    if config["schema_version"] != 1 or config["primary_algorithm"] != "ppo":
        raise ValueError("Unsupported analysis protocol")
    if config["primary_metric"] != "evaluation_auc":
        raise ValueError("Primary endpoint differs from the frozen protocol")
    if config["pairing_variable"] != "seed" or config["expected_runs"] != 160:
        raise ValueError("Pairing or run count differs from the frozen design")
    if config["bootstrap_samples"] != 10_000 or config["training_budget"] != 1_000_000:
        raise ValueError("Bootstrap or training budget differs from the frozen design")
    return config


def reward_condition(reward_type, reward_delay):
    if reward_type == "delayed":
        if int(reward_delay) not in {8, 32}:
            raise ValueError("Unknown delayed reward condition")
        return f"delayed_{int(reward_delay)}"
    if reward_type not in {"dense", "sparse"}:
        raise ValueError("Unknown reward condition")
    return reward_type


def normalized_auc(steps, values, budget):
    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if (
        len(steps) < 2
        or len(steps) != len(values)
        or not np.isfinite(steps).all()
        or not np.isfinite(values).all()
        or steps[0] != 0
        or steps[-1] != budget
        or np.any(np.diff(steps) <= 0)
    ):
        raise ValueError("AUC requires finite, increasing evaluations from zero through budget")
    return float(np.trapezoid(values, steps) / budget)


def _first_step(frame, column, threshold):
    reached = frame.loc[frame[column] >= threshold, "env_steps"]
    return None if reached.empty else int(reached.iloc[0])


def run_outcomes(evaluations: pd.DataFrame, updates: pd.DataFrame, config: dict):
    evaluations = evaluations.sort_values("env_steps")
    updates = updates.sort_values("env_steps")
    budget = int(config["training_budget"])
    if evaluations.run_id.nunique() != 1 or updates.run_id.nunique() != 1:
        raise ValueError("Run outcomes accept exactly one run")
    if int(updates.env_steps.iloc[-1]) != budget:
        raise ValueError("Run has not reached the frozen training budget")
    final_evaluations = evaluations.tail(int(config["final_evaluation_checkpoints"]))
    final_updates = updates.tail(int(config["final_critic_updates"]))
    unstable = (updates.approx_kl > config["instability"]["approximate_kl_threshold"]) | (
        updates.clip_fraction > config["instability"]["clip_fraction_threshold"]
    )
    return {
        "evaluation_auc": normalized_auc(
            evaluations.env_steps, evaluations.mean_base_dense_return, budget
        ),
        "final_evaluation_return": float(final_evaluations.mean_base_dense_return.mean()),
        "best_evaluation_return_exploratory": float(evaluations.mean_base_dense_return.max()),
        "steps_to_success": _first_step(
            evaluations, "mean_base_dense_return", config["performance_return_threshold"]
        ),
        "final_upright_fraction": float(final_evaluations.mean_upright_fraction.mean()),
        "upright_fraction_auc": normalized_auc(
            evaluations.env_steps, evaluations.mean_upright_fraction, budget
        ),
        "final_success_probability": float(final_evaluations.stable_success_rate.mean()),
        "first_successful_evaluation_step": _first_step(
            evaluations,
            "stable_success_rate",
            config["successful_evaluation_probability"],
        ),
        "final_critic_explained_variance": float(final_updates.explained_variance.mean()),
        "final_critic_mse": float(final_updates.critic_loss.mean()),
        "training_instability_rate": float(unstable.mean()),
    }


def bootstrap_interval(values, samples, confidence, seed):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be nonempty and finite")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    alpha = 1.0 - confidence
    return tuple(float(x) for x in np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def sign_flip_pvalue(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        return math.nan
    observed = abs(values.mean())
    if len(values) <= 20:
        patterns = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
        bits = (patterns >> np.arange(len(values), dtype=np.uint64)) & 1
        permuted = ((bits * 2 - 1).astype(np.int8) * values).mean(axis=1)
    else:
        rng = np.random.default_rng(20260906)
        permuted = (rng.choice([-1, 1], size=(100_000, len(values))) * values).mean(axis=1)
    return float(np.mean(np.abs(permuted) >= observed))


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    result = np.full(len(pvalues), np.nan)
    finite = np.flatnonzero(np.isfinite(pvalues))
    order = finite[np.argsort(pvalues[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * pvalues[index])
        result[index] = min(1.0, running)
    return result


def distance_to_next_reward(rewards):
    rewards = np.asarray(rewards)
    distances = np.full(len(rewards), np.inf)
    next_index = None
    for index in range(len(rewards) - 1, -1, -1):
        if rewards[index] != 0:
            next_index = index
        if next_index is not None:
            distances[index] = next_index - index
    return distances


def distance_bin(distances, bins):
    labels = np.empty(len(distances), dtype=object)
    for index, distance in enumerate(distances):
        labels[index] = "33+"
        for definition in bins:
            maximum = definition["maximum"]
            if distance >= definition["minimum"] and (maximum is None or distance <= maximum):
                labels[index] = definition["label"]
                break
    return labels


def discounted_returns(rewards, gamma):
    rewards = np.asarray(rewards, dtype=np.float64)
    result = np.empty_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        result[index] = running
    return result


def pre_reward_advantage_magnitude(diagnostics_dir, horizon):
    total = 0.0
    count = 0
    for diagnostic in sorted(Path(diagnostics_dir).glob("update-*.npz")):
        with np.load(diagnostic, allow_pickle=False) as arrays:
            distances = distance_to_next_reward(arrays["rewards"])
            selected = np.isfinite(distances) & (distances > 0)
            total += float(np.abs(arrays[f"advantages_{horizon}"][selected]).sum())
            count += int(selected.sum())
    return (math.nan if count == 0 else total / count), count

"""Study-level quality checks that never compute pilot hypothesis contrasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from gae_credit.logging.run_logger import validate_run
from gae_credit.pilot import config_for_row


def _diagnostic_checks(path: Path, row: dict, gamma: float):
    checked = 0
    for diagnostic in sorted((path / "diagnostics").glob("update-*.npz")):
        with np.load(diagnostic, allow_pickle=False) as arrays:
            for name in arrays.files:
                if not np.isfinite(arrays[name]).all():
                    raise ValueError(f"nonfinite diagnostic {diagnostic.name}:{name}")
            if row["reward_kind"] == "delayed":
                rewards = arrays["rewards"]
                base = arrays["base_dense_rewards"]
                emitted = arrays["emitted_payout"].astype(bool)
                if np.any(rewards[~emitted] != 0):
                    raise ValueError("delayed reward is nonzero off payout steps")
                discounts = gamma ** np.arange(len(rewards))
                if not np.isclose(rewards @ discounts, base @ discounts, rtol=1e-10, atol=1e-8):
                    raise ValueError("discounted dense/delayed diagnostic totals differ")
            checked += 1
    if checked != 250:
        raise ValueError(f"expected 250 diagnostic updates, found {checked}")


def validate_study(rows: list[dict], spec: dict, base, download_root: str | Path):
    root = Path(download_root)
    reports, initial_hashes = [], {}
    for row in rows:
        path = root / row["run_id"]
        if not path.exists():
            reports.append({"run_id": row["run_id"], "status": "missing"})
            continue
        try:
            report = validate_run(path)
            config = config_for_row(spec, base, row)
            manifest = json.loads((path / "manifest.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            if manifest.get("phase") != "pilot" or manifest["config_hash"] != row["config_hash"]:
                raise ValueError("manifest does not match pilot matrix")
            updates = pq.read_table(path / "updates.parquet").to_pandas()
            episodes = pq.read_table(path / "episodes.parquet").to_pandas()
            evaluations = pq.read_table(path / "evaluations.parquet").to_pandas()
            expected_eval_steps = list(range(0, 250_001, 25_000))
            if evaluations["env_steps"].tolist() != expected_eval_steps:
                raise ValueError("evaluation checkpoints do not match the pilot schedule")
            if not (updates["action_std"] > 0).all():
                raise ValueError("action standard deviation is not positive")
            train = episodes[episodes["split"] == "train"]
            if row["reward_kind"] == "sparse" and not (train["train_return"] > 0).any():
                raise ValueError("sparse training never encountered positive reward")
            _diagnostic_checks(path, row, config.estimator.gamma)
            hashes = (summary["initial_actor_hash"], summary["initial_critic_hash"])
            previous = initial_hashes.setdefault(row["seed"], hashes)
            if previous != hashes:
                raise ValueError("same seed produced different initial model parameters")
            reports.append(
                {
                    **report,
                    "status": "valid",
                    "runtime_seconds": float(updates["elapsed_seconds"].iloc[-1]),
                    "checkpoint_bytes": sum(
                        item.stat().st_size for item in (path / "checkpoints").glob("*.pt")
                    ),
                    "rewarded_training_episodes": int((train["train_return"] > 0).sum()),
                    "upright_entries": int(train["first_upright_timestep"].notna().sum()),
                }
            )
        except Exception as error:
            reports.append({"run_id": row["run_id"], "status": "invalid", "error": str(error)})
    valid = [item for item in reports if item["status"] == "valid"]
    return {
        "purpose": "pilot quality control only; excluded from final inference",
        "expected_runs": len(rows),
        "valid_runs": len(valid),
        "missing_runs": sum(item["status"] == "missing" for item in reports),
        "invalid_runs": sum(item["status"] == "invalid" for item in reports),
        "complete": len(valid) == len(rows),
        "median_runtime_seconds": float(np.median([r["runtime_seconds"] for r in valid]))
        if valid
        else None,
        "reports": reports,
    }

"""Extrapolate measured run duration; price is supplied, never guessed."""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from gae_credit.logging import validate_run


def estimate(path, target_steps=1_000_000, hourly_price=None):
    validated = validate_run(path)
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    seconds = (
        datetime.fromisoformat(manifest["finished_at"])
        - datetime.fromisoformat(manifest["started_at"])
    ).total_seconds()
    seconds += manifest.get("elapsed_before_resume_seconds", 0.0)
    if seconds <= 0 or target_steps <= 0:
        raise ValueError("Measured duration and target steps must be positive")
    if hourly_price is not None and (not math.isfinite(hourly_price) or hourly_price < 0):
        raise ValueError("hourly-price must be finite and nonnegative")
    projected = seconds * target_steps / validated["total_env_steps"]
    return {
        "measured_seconds": seconds,
        "measured_env_steps": validated["total_env_steps"],
        "target_env_steps": target_steps,
        "linear_estimated_seconds": projected,
        "assumed_hourly_price": hourly_price,
        "estimated_compute_cost": None if hourly_price is None else projected / 3600 * hourly_price,
        "caveat": "Linear estimate includes this run's logging/evaluation overhead; excludes VM provisioning, image build/storage, GCS, TensorBoard, retries and taxes. Use a cloud run on the target machine; smoke extrapolation is approximate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--target-env-steps", type=int, default=1_000_000)
    parser.add_argument("--hourly-price", type=float)
    args = parser.parse_args()
    print(json.dumps(estimate(args.run_dir, args.target_env_steps, args.hourly_price), indent=2))


if __name__ == "__main__":
    main()

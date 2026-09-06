"""Exact restart at completed PPO update boundaries, never mid-episode."""

import hashlib
import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_VERSION = 1


class TrainingInterrupted(RuntimeError):
    """An intentional stop after a durable checkpoint, used to exercise recovery."""


def runtime_fingerprint():
    data = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": {k: version(k) for k in ("torch", "numpy", "gymnasium")},
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def capture_rng(action_rng, update_rng, evaluation_rng, env_rng, diagnostic_rng):
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy_global": [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
        "torch_global": torch.random.get_rng_state(),
        "action": action_rng.get_state(),
        "optimization": update_rng.get_state(),
        "evaluation": evaluation_rng.get_state(),
        "environment": env_rng.bit_generator.state,
        "diagnostics": diagnostic_rng.bit_generator.state,
    }


def restore_rng(state, action_rng, update_rng, evaluation_rng, env_rng, diagnostic_rng):
    random.setstate(state["python"])
    n = state["numpy_global"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.random.set_rng_state(state["torch_global"])
    for generator, key in (
        (action_rng, "action"),
        (update_rng, "optimization"),
        (evaluation_rng, "evaluation"),
    ):
        generator.set_state(state[key])
    env_rng.bit_generator.state = state["environment"]
    diagnostic_rng.bit_generator.state = state["diagnostics"]


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, config, context):
    # Tensor/primitive-only payload; do not enable arbitrary pickle execution.
    data = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_VERSION,
        "run_id": context["run_id"],
        "config_hash": config.config_hash(),
        "source_digest": context["source_digest"],
        "image_digest": context["image_digest"],
        "runtime_fingerprint": runtime_fingerprint(),
    }
    for field, value in expected.items():
        if data.get(field) != value:
            raise ValueError(f"Cannot resume: checkpoint {field} mismatch")
    if data.get("config") != config.to_dict():
        raise ValueError("Cannot resume: checkpoint resolved configuration mismatch")
    update = data.get("update_index")
    rollout_size = config.environment.max_steps * config.training.episodes_per_rollout
    if (
        isinstance(update, bool)
        or not isinstance(update, int)
        or not 1 <= update <= config.training.total_env_steps // rollout_size
    ):
        raise ValueError("Cannot resume: invalid update counter")
    if data.get("env_steps") != update * rollout_size:
        raise ValueError("Cannot resume: inconsistent step counter")
    rows = data["logger_snapshot"]["rows"]
    if len(rows["updates"]) != update or len(rows["episodes"]) != data["episode_id"]:
        raise ValueError("Cannot resume: checkpoint log counts mismatch")
    if rows["evaluations"] != [{"run_id": context["run_id"], **r} for r in data["evaluation_rows"]]:
        raise ValueError("Cannot resume: checkpoint evaluation rows mismatch")
    if [r["update_index"] for r in rows["updates"]] != list(range(1, update + 1)):
        raise ValueError("Cannot resume: checkpoint update sequence mismatch")
    return data

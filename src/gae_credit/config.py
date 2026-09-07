"""Strict typed configuration and stable identities for experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

Horizon = int | Literal["full"]


def _positive(value, name: str, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value <= 0 or (integer and not isinstance(value, int)):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")


def _horizon(value: Horizon) -> None:
    if value != "full":
        _positive(value, "horizon", integer=True)


@dataclass(frozen=True)
class EnvironmentConfig:
    max_steps: int = 200
    dt: float = 0.05
    max_torque: float = 2.0
    max_speed: float = 8.0
    g: float = 10.0
    m: float = 1.0
    l: float = 1.0  # noqa: E741 - conventional pendulum length symbol
    include_time_remaining: bool = True


@dataclass(frozen=True)
class RewardConfig:
    kind: Literal["dense", "delayed", "sparse"] = "dense"
    delay_block_size: int = 1
    upright_angle_threshold: float | None = None
    upright_velocity_threshold: float | None = None

    @property
    def resolved_upright_angle_threshold(self) -> float:
        return 0.262 if self.upright_angle_threshold is None else self.upright_angle_threshold

    @property
    def resolved_upright_velocity_threshold(self) -> float:
        return 1.0 if self.upright_velocity_threshold is None else self.upright_velocity_threshold


@dataclass(frozen=True)
class EstimatorConfig:
    gamma: float = 0.995
    lam: float = 0.95
    horizon: Horizon = 3
    actor_horizon: Horizon | None = None
    critic_horizon: Horizon | None = None

    def __post_init__(self):
        if self.actor_horizon is None:
            object.__setattr__(self, "actor_horizon", self.horizon)
        if self.critic_horizon is None:
            object.__setattr__(self, "critic_horizon", self.horizon)


@dataclass(frozen=True)
class AlgorithmConfig:
    hidden_size: int = 64
    actor_lr: float = 0.0003
    critic_lr: float = 0.001
    epochs: int = 4
    minibatch_size: int = 250
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    normalize_advantages: bool = True
    init_log_std: float = -0.5


@dataclass(frozen=True)
class TrainingConfig:
    total_env_steps: int = 1_000_000
    episodes_per_rollout: int = 5
    checkpoint_interval_env_steps: int = 0


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str = "runs"
    save_diagnostics: bool = True
    save_checkpoint: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 20
    interval_env_steps: int = 10_000
    seed: int = 20260905


@dataclass(frozen=True)
class SeedConfig:
    final_training_seeds: tuple[int, ...] = tuple(range(10))
    pilot_seeds: tuple[int, ...] = (100, 101, 102)


@dataclass(frozen=True)
class StudyConfig:
    study_id: str = "gae-pendulum-v1"
    algorithm: str = "ppo"
    seed: int = 0
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    optimizer: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)

    def __post_init__(self):
        if self.algorithm != "ppo":
            raise ValueError("Only PPO is implemented")
        if not isinstance(self.study_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.study_id):
            raise ValueError("study_id must contain letters, digits, underscores or hyphens")
        for name, value in [("seed", self.seed), ("evaluation.seed", self.evaluation.seed)]:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be a uint32 integer")
        for name in ("max_steps",):
            _positive(getattr(self.environment, name), name, integer=True)
        for name in ("dt", "max_torque", "max_speed", "g", "m", "l"):
            _positive(getattr(self.environment, name), name)
        if self.environment.include_time_remaining is not True:
            raise ValueError("Finite task requires include_time_remaining: true")
        if self.reward.kind not in ("dense", "delayed", "sparse"):
            raise ValueError("reward.kind must be dense, delayed or sparse")
        _positive(self.reward.delay_block_size, "delay_block_size", integer=True)
        if self.reward.kind != "delayed" and self.reward.delay_block_size != 1:
            raise ValueError("delay_block_size must be 1 for dense and sparse rewards")
        thresholds = (
            self.reward.upright_angle_threshold,
            self.reward.upright_velocity_threshold,
        )
        if self.reward.kind != "sparse" and any(value is not None for value in thresholds):
            raise ValueError("Upright threshold overrides are only valid for sparse reward")
        if any(value is not None for value in thresholds):
            if any(value is None for value in thresholds):
                raise ValueError("Specify both sparse upright thresholds")
            for name, value in zip(
                ("upright_angle_threshold", "upright_velocity_threshold"), thresholds
            ):
                _positive(value, name)
        for value in (
            self.estimator.horizon,
            self.estimator.actor_horizon,
            self.estimator.critic_horizon,
        ):
            _horizon(value)
        if self.estimator.actor_horizon != self.estimator.horizon:
            raise ValueError("actor_horizon must match horizon")
        if self.estimator.critic_horizon != self.estimator.horizon:
            raise ValueError("Days 1–3 require matching actor and critic horizons")
        for name, value in (("gamma", self.estimator.gamma), ("lambda", self.estimator.lam)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.reward.kind == "delayed" and self.estimator.gamma <= 0:
            raise ValueError("Delayed reward correction requires gamma > 0")
        for name in ("hidden_size", "epochs", "minibatch_size"):
            _positive(getattr(self.optimizer, name), name, integer=True)
        for name in (
            "actor_lr",
            "critic_lr",
            "clip_epsilon",
            "value_coef",
            "max_grad_norm",
            "target_kl",
        ):
            _positive(getattr(self.optimizer, name), name)
        if self.optimizer.clip_epsilon >= 1:
            raise ValueError("clip_epsilon must be less than 1")
        if not isinstance(self.optimizer.normalize_advantages, bool):
            raise ValueError("normalize_advantages must be boolean")
        if not math.isfinite(self.optimizer.entropy_coef) or self.optimizer.entropy_coef < 0:
            raise ValueError("entropy_coef must be finite and nonnegative")
        if not math.isfinite(self.optimizer.init_log_std):
            raise ValueError("init_log_std must be finite")
        _positive(self.training.total_env_steps, "total_env_steps", integer=True)
        _positive(self.training.episodes_per_rollout, "episodes_per_rollout", integer=True)
        _positive(self.evaluation.episodes, "evaluation.episodes", integer=True)
        _positive(self.evaluation.interval_env_steps, "interval_env_steps", integer=True)
        rollout_size = self.environment.max_steps * self.training.episodes_per_rollout
        checkpoint_interval = self.training.checkpoint_interval_env_steps
        if (
            isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, int)
            or checkpoint_interval < 0
            or checkpoint_interval % rollout_size
        ):
            raise ValueError(
                "checkpoint interval must be zero or a positive multiple of rollout size"
            )
        if checkpoint_interval and not self.logging.save_checkpoint:
            raise ValueError("Periodic checkpoints require save_checkpoint")
        if self.training.total_env_steps % rollout_size:
            raise ValueError("total_env_steps must be divisible by complete rollout size")
        if self.evaluation.interval_env_steps % rollout_size:
            raise ValueError("evaluation interval must be divisible by rollout size")
        if not isinstance(self.logging.output_dir, str) or not self.logging.output_dir.strip():
            raise ValueError("logging.output_dir must be a nonempty path")
        for name in ("save_diagnostics", "save_checkpoint"):
            if not isinstance(getattr(self.logging, name), bool):
                raise ValueError(f"{name} must be boolean")
        for seed_set in (self.seeds.final_training_seeds, self.seeds.pilot_seeds):
            if not seed_set or len(set(seed_set)) != len(seed_set):
                raise ValueError("seed sets must be nonempty and unique")
            for seed in seed_set:
                if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                    raise ValueError("seed sets must contain uint32 integers")
        if set(self.seeds.final_training_seeds) & set(self.seeds.pilot_seeds):
            raise ValueError("Pilot and final training seeds must not overlap")

    def to_dict(self) -> dict:
        result = asdict(self)
        # Preserve identities and validation of the already-produced Day 3 data.
        if self.training.checkpoint_interval_env_steps == 0:
            result["training"].pop("checkpoint_interval_env_steps")
        for name in ("upright_angle_threshold", "upright_velocity_threshold"):
            if result["reward"][name] is None:
                result["reward"].pop(name)
        result["estimator"]["lambda"] = result["estimator"].pop("lam")
        result["seeds"] = {k: list(v) for k, v in result["seeds"].items()}
        return result

    def config_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def run_id(self) -> str:
        reward = self.reward.kind
        if reward == "delayed":
            reward += f"-d{self.reward.delay_block_size:03d}"
        h = self.estimator.horizon
        horizon = "full" if h == "full" else f"h{h:03d}"
        return f"{self.algorithm}__{reward}__{horizon}__seed-{self.seed:03d}__{self.config_hash()[:12]}"


def config_from_dict(raw: dict) -> StudyConfig:
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    data = dict(raw)
    section_types = {
        "environment": EnvironmentConfig,
        "reward": RewardConfig,
        "estimator": EstimatorConfig,
        "optimizer": AlgorithmConfig,
        "training": TrainingConfig,
        "logging": LoggingConfig,
        "evaluation": EvaluationConfig,
        "seeds": SeedConfig,
    }
    try:
        for name, cls in section_types.items():
            if name not in data:
                continue
            if not isinstance(data[name], dict):
                raise ValueError(f"{name} must be a mapping")
            section = dict(data[name])
            if name == "estimator" and "lambda" in section:
                if "lam" in section:
                    raise ValueError("Specify only lambda, not both lambda and lam")
                section["lam"] = section.pop("lambda")
            if name == "seeds":
                section = {k: tuple(v) for k, v in section.items()}
            data[name] = cls(**section)
        return StudyConfig(**data)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc


def load_config(path: str | Path) -> StudyConfig:
    """Load one standalone YAML mapping; no implicit inheritance or ambient env vars."""
    return config_from_dict(yaml.safe_load(Path(path).read_text()))


def derive_seeds(config: StudyConfig) -> dict[str, int]:
    names = ("model", "environment", "action", "diagnostics", "optimization")
    children = np.random.SeedSequence(config.seed).spawn(len(names))
    streams = {name: int(child.generate_state(1)[0]) for name, child in zip(names, children)}
    # Fixed evaluation initial states are shared across training seeds and conditions.
    streams["evaluation"] = int(np.random.SeedSequence(config.evaluation.seed).generate_state(1)[0])
    return streams

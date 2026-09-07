"""Reward delivery mechanisms, independent of dynamics and neural networks."""

from copy import deepcopy
from typing import Any

import numpy as np

from gae_credit.envs.pendulum import is_upright


def _finite_reward(reward: float) -> float:
    value = float(reward)
    if not np.isfinite(value):
        raise ValueError("base reward must be finite")
    return value


class DenseReward:
    def reset(self) -> None:
        pass

    def transform(self, base_dense_reward, theta, theta_dot, timestep, terminated):
        return _finite_reward(base_dense_reward), {}

    def get_state(self) -> dict[str, Any]:
        return {"kind": "dense"}

    def set_state(self, snapshot: dict[str, Any]) -> None:
        if snapshot != {"kind": "dense"}:
            raise ValueError("incompatible dense reward snapshot")


class BlockDelayedReward:
    """Emit discount-corrected blocks and flush every finite episode boundary."""

    def __init__(self, delay_block_size: int = 8, gamma: float = 0.995):
        if isinstance(delay_block_size, bool) or not isinstance(delay_block_size, int):
            raise ValueError("delay_block_size must be a positive integer")
        if delay_block_size < 1:
            raise ValueError("delay_block_size must be positive")
        if not np.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("delayed rewards require 0 < gamma <= 1")
        self.delay_block_size = delay_block_size
        self.gamma = float(gamma)
        self.reset()

    def reset(self) -> None:
        self.block_id = 0
        self.count = 0
        self.pending_reward = 0.0
        self.pending_corrected_reward = 0.0

    def transform(self, base_dense_reward, theta, theta_dot, timestep, terminated):
        reward = _finite_reward(base_dense_reward)
        count = self.count + 1
        pending = self.pending_reward + reward
        corrected = self.pending_corrected_reward / self.gamma + reward
        if not np.isfinite(corrected):
            raise ValueError("discount correction overflowed; reduce delay or increase gamma")
        emitted = count == self.delay_block_size or bool(terminated)
        payout = corrected if emitted else 0.0
        diagnostics = {
            "block_id": self.block_id,
            "block_position": count,
            "emitted_payout": emitted,
            "payout_count": count if emitted else 0,
            "pending_reward": 0.0 if emitted else pending,
            "discount_corrected_payout": payout,
            "pending_corrected_reward": 0.0 if emitted else corrected,
        }
        self.count = 0 if emitted else count
        self.pending_reward = diagnostics["pending_reward"]
        self.pending_corrected_reward = diagnostics["pending_corrected_reward"]
        if emitted:
            self.block_id += 1
        return payout, diagnostics

    def get_state(self) -> dict[str, Any]:
        return deepcopy(
            {
                "kind": "delayed",
                "delay_block_size": self.delay_block_size,
                "gamma": self.gamma,
                "block_id": self.block_id,
                "count": self.count,
                "pending_reward": self.pending_reward,
                "pending_corrected_reward": self.pending_corrected_reward,
            }
        )

    def set_state(self, snapshot: dict[str, Any]) -> None:
        state = deepcopy(snapshot)
        if (
            state["kind"] != "delayed"
            or state["delay_block_size"] != self.delay_block_size
            or state["gamma"] != self.gamma
        ):
            raise ValueError("incompatible delayed reward snapshot")
        if (
            isinstance(state["count"], bool)
            or not isinstance(state["count"], int)
            or not 0 <= state["count"] < self.delay_block_size
            or isinstance(state["block_id"], bool)
            or not isinstance(state["block_id"], int)
            or state["block_id"] < 0
        ):
            raise ValueError("invalid delayed reward counters")
        pending = _finite_reward(state["pending_reward"])
        corrected = _finite_reward(state["pending_corrected_reward"])
        if state["count"] == 0 and (pending != 0 or corrected != 0):
            raise ValueError("empty reward buffer must have zero pending reward")
        self.block_id = state["block_id"]
        self.count = state["count"]
        self.pending_reward = pending
        self.pending_corrected_reward = corrected


class SparseUprightReward:
    """Reward every qualifying post-transition state; success needs a 10-step streak."""

    def __init__(self, angle_threshold: float = 0.262, velocity_threshold: float = 1.0):
        if not np.all(np.isfinite([angle_threshold, velocity_threshold])):
            raise ValueError("upright thresholds must be finite")
        if angle_threshold <= 0 or velocity_threshold <= 0:
            raise ValueError("upright thresholds must be positive")
        self.angle_threshold = float(angle_threshold)
        self.velocity_threshold = float(velocity_threshold)
        self.reset()

    def reset(self) -> None:
        self.previous_upright = False
        self.upright_streak = 0
        self.longest_upright_streak = 0
        self.first_upright_timestep = None

    def transform(self, base_dense_reward, theta, theta_dot, timestep, terminated):
        if not np.all(np.isfinite([theta, theta_dot])):
            raise ValueError("physical state must be finite")
        upright = is_upright(
            theta,
            theta_dot,
            angle_threshold=self.angle_threshold,
            velocity_threshold=self.velocity_threshold,
        )
        entered = upright and not self.previous_upright
        left = self.previous_upright and not upright
        self.upright_streak = self.upright_streak + 1 if upright else 0
        self.longest_upright_streak = max(self.longest_upright_streak, self.upright_streak)
        if upright and self.first_upright_timestep is None:
            self.first_upright_timestep = int(timestep)
        self.previous_upright = upright
        return float(upright), {
            "upright": upright,
            "entered_upright": entered,
            "left_upright": left,
            "upright_streak": self.upright_streak,
            "longest_upright_streak": self.longest_upright_streak,
            "first_upright_timestep": self.first_upright_timestep,
            "stable_success": self.longest_upright_streak >= 10,
        }

    def get_state(self) -> dict[str, Any]:
        return deepcopy(
            {
                "kind": "sparse",
                "angle_threshold": self.angle_threshold,
                "velocity_threshold": self.velocity_threshold,
                "previous_upright": self.previous_upright,
                "upright_streak": self.upright_streak,
                "longest_upright_streak": self.longest_upright_streak,
                "first_upright_timestep": self.first_upright_timestep,
            }
        )

    def set_state(self, snapshot: dict[str, Any]) -> None:
        state = deepcopy(snapshot)
        if (
            state["kind"] != "sparse"
            or state.get("angle_threshold", 0.262) != self.angle_threshold
            or state.get("velocity_threshold", 1.0) != self.velocity_threshold
        ):
            raise ValueError("incompatible sparse reward snapshot")
        for key in ("upright_streak", "longest_upright_streak"):
            if isinstance(state[key], bool) or not isinstance(state[key], int) or state[key] < 0:
                raise ValueError("upright streak counters must be nonnegative integers")
        if state["upright_streak"] > state["longest_upright_streak"]:
            raise ValueError("current streak exceeds longest streak")
        if bool(state["previous_upright"]) != (state["upright_streak"] > 0):
            raise ValueError("upright state and streak disagree")
        first = state["first_upright_timestep"]
        if first is not None and (
            isinstance(first, bool) or not isinstance(first, int) or first < 1
        ):
            raise ValueError("first upright timestep must be a positive integer or None")
        self.previous_upright = bool(state["previous_upright"])
        self.upright_streak = state["upright_streak"]
        self.longest_upright_streak = state["longest_upright_streak"]
        self.first_upright_timestep = first


def make_reward(config: Any, gamma: float = 0.995):
    if config.kind == "dense":
        return DenseReward()
    if config.kind == "delayed":
        return BlockDelayedReward(config.delay_block_size, gamma)
    if config.kind == "sparse":
        return SparseUprightReward(
            getattr(config, "resolved_upright_angle_threshold", 0.262),
            getattr(config, "resolved_upright_velocity_threshold", 1.0),
        )
    raise ValueError(f"unknown reward kind: {config.kind!r}")

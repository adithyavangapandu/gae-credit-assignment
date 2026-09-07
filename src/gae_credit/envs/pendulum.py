"""Pendulum dynamics with an observed, genuine finite task horizon."""

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def normalize_angle(theta: float) -> float:
    """Map an angle to [-pi, pi), with zero denoting upright."""
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def is_upright(
    theta: float,
    theta_dot: float,
    *,
    angle_threshold: float = 0.262,
    velocity_threshold: float = 1.0,
) -> bool:
    return abs(normalize_angle(theta)) < angle_threshold and abs(theta_dot) < velocity_threshold


class PendulumEnv(gym.Env):
    """A 200-step task by default; its final transition is terminal, not truncated.

    Dense reward is computed from the *next* physical state and the applied
    torque. This intentionally differs from Gymnasium's pre-transition cost.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Any = None):
        if config is None:
            from gae_credit.config import EnvironmentConfig

            config = EnvironmentConfig()
        self.config = config
        self.max_steps = config.max_steps
        self.dt = config.dt
        self.max_torque = config.max_torque
        self.max_speed = config.max_speed
        self.g = config.g
        self.m = config.m
        self.l = config.l
        self.include_time_remaining = config.include_time_remaining
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise ValueError("max_steps must be a positive integer")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        parameters = [self.dt, self.max_torque, self.max_speed, self.g, self.m, self.l]
        if not np.all(np.isfinite(parameters)) or min(parameters) <= 0:
            raise ValueError("physical parameters must be finite and positive")
        high = [1.0, 1.0, self.max_speed]
        low = [-1.0, -1.0, -self.max_speed]
        if self.include_time_remaining:
            high.append(1.0)
            low.append(0.0)
        self.observation_space = spaces.Box(
            np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
        )
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.theta = 0.0
        self.theta_dot = 0.0
        self.timestep = 0
        self._has_reset = False

    def _observation(self) -> np.ndarray:
        observation = [np.cos(self.theta), np.sin(self.theta), self.theta_dot]
        if self.include_time_remaining:
            observation.append((self.max_steps - self.timestep) / self.max_steps)
        return np.asarray(observation, dtype=np.float32)

    def _info(self, torque: float) -> dict[str, Any]:
        angle_cost = normalize_angle(self.theta) ** 2
        velocity_cost = 0.1 * self.theta_dot**2
        torque_cost = 0.001 * torque**2
        return {
            "theta": self.theta,
            "theta_dot": self.theta_dot,
            "timestep": self.timestep,
            "torque": torque,
            "angle_cost": angle_cost,
            "velocity_cost": velocity_cost,
            "torque_cost": torque_cost,
            "base_dense_reward": -(angle_cost + velocity_cost + torque_cost),
            "upright_indicator": is_upright(self.theta, self.theta_dot),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("custom initial states use set_state; reset options are unsupported")
        self.theta = float(self.np_random.uniform(-np.pi, np.pi))
        self.theta_dot = float(self.np_random.uniform(-1.0, 1.0))
        self.timestep = 0
        self._has_reset = True
        return self._observation(), self._info(0.0)

    def step(self, action):
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        if self.timestep >= self.max_steps:
            raise RuntimeError("episode has terminated; call reset")
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.size != 1 or not np.all(np.isfinite(action_array)):
            raise ValueError("action must contain one finite torque")
        torque = float(np.clip(action_array.item(), -self.max_torque, self.max_torque))
        acceleration = 3.0 * self.g / (2.0 * self.l) * np.sin(self.theta) + 3.0 * torque / (
            self.m * self.l**2
        )
        self.theta_dot = float(
            np.clip(self.theta_dot + acceleration * self.dt, -self.max_speed, self.max_speed)
        )
        self.theta = normalize_angle(self.theta + self.theta_dot * self.dt)
        self.timestep += 1
        info = self._info(torque)
        return (
            self._observation(),
            info["base_dense_reward"],
            self.timestep == self.max_steps,
            False,
            info,
        )

    def get_state(self) -> dict[str, Any]:
        if not self._has_reset:
            raise RuntimeError("reset must be called before taking a snapshot")
        return deepcopy(
            {
                "theta": self.theta,
                "theta_dot": self.theta_dot,
                "timestep": self.timestep,
                "rng_state": self.np_random.bit_generator.state,
            }
        )

    def set_state(self, snapshot: dict[str, Any]) -> None:
        state = deepcopy(snapshot)
        theta, theta_dot, timestep = state["theta"], state["theta_dot"], state["timestep"]
        if not np.all(np.isfinite([theta, theta_dot])) or abs(theta_dot) > self.max_speed:
            raise ValueError("snapshot contains an invalid physical state")
        if isinstance(timestep, bool) or not isinstance(timestep, int):
            raise ValueError("snapshot timestep must be an integer")
        if not 0 <= timestep <= self.max_steps:
            raise ValueError("snapshot timestep is outside the episode")
        # Validate the RNG snapshot before mutating the environment.
        generator = np.random.default_rng()
        generator.bit_generator.state = state["rng_state"]
        self.theta = float(theta)
        self.theta_dot = float(theta_dot)
        self.timestep = timestep
        self._np_random = generator
        self._has_reset = True

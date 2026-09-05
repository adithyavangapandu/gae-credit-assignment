"""Shared finite-horizon physics and independently stateful reward schedules."""

from gae_credit.envs.pendulum import PendulumEnv, is_upright, normalize_angle
from gae_credit.envs.rewards import (
    BlockDelayedReward,
    DenseReward,
    SparseUprightReward,
    make_reward,
)

__all__ = [
    "BlockDelayedReward",
    "DenseReward",
    "PendulumEnv",
    "SparseUprightReward",
    "is_upright",
    "make_reward",
    "normalize_angle",
]

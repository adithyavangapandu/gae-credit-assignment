"""Finite and full generalized advantage estimation with terminal boundaries."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GAEResult:
    advantages: np.ndarray
    value_targets: np.ndarray
    td_errors: np.ndarray
    effective_horizon_at_each_timestep: np.ndarray

    @property
    def effective_horizons(self) -> np.ndarray:
        return self.effective_horizon_at_each_timestep


def _array(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


def compute_gae(
    rewards,
    values,
    terminated,
    gamma: float = 0.995,
    lam: float = 0.95,
    horizon: int | str = 3,
) -> GAEResult:
    """Use at most H residuals, stopping at the first terminal transition.

    A final nonterminal transition bootstraps values[T], but the residual sum
    still ends at the array boundary. Complete finite episodes end terminal.
    Inputs are never modified, and returned arrays use float64 arithmetic.
    """
    rewards = _array(rewards, "rewards")
    values = _array(values, "values")
    terminals = np.asarray(terminated)
    size = rewards.size
    if values.size != size + 1:
        raise ValueError("values must have exactly len(rewards) + 1 elements")
    if terminals.shape != (size,) or not np.all(np.isin(terminals, [False, True])):
        raise ValueError("terminated must contain one boolean mask per reward")
    terminals = terminals.astype(bool)
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not np.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("lambda must be finite and in [0, 1]")
    if horizon != "full" and (
        isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1
    ):
        raise ValueError("horizon must be a positive integer or 'full'")
    td_errors = rewards + gamma * (~terminals) * values[1:] - values[:-1]
    advantages = np.zeros(size, dtype=np.float64)
    effective = np.zeros(size, dtype=np.int64)
    q = gamma * lam
    if horizon == "full":
        running = 0.0
        remaining = 0
        for t in range(size - 1, -1, -1):
            if terminals[t]:
                running, remaining = 0.0, 0
            running = td_errors[t] + q * running
            remaining += 1
            advantages[t] = running
            effective[t] = remaining
    else:
        for t in range(size):
            weight = 1.0
            for index in range(t, min(t + int(horizon), size)):
                advantages[t] += weight * td_errors[index]
                effective[t] += 1
                if terminals[index]:
                    break
                weight *= q
    return GAEResult(advantages, values[:-1] + advantages, td_errors, effective)


def _paired(full_advantages, truncated_advantages):
    full = _array(full_advantages, "full_advantages")
    truncated = _array(truncated_advantages, "truncated_advantages")
    if full.shape != truncated.shape:
        raise ValueError("advantage arrays must have identical shapes")
    return full, truncated


def compute_omitted_tail(full_advantages, truncated_advantages) -> np.ndarray:
    full, truncated = _paired(full_advantages, truncated_advantages)
    return full - truncated


def compute_tail_fraction(full_advantages, truncated_advantages, eps=1e-12) -> np.ndarray:
    """Bounded omitted magnitude / (retained magnitude + omitted magnitude).

    This is not a percentage of the full advantage, which can vanish because
    the retained and omitted signed components cancel.
    """
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    full, truncated = _paired(full_advantages, truncated_advantages)
    tail = np.abs(full - truncated)
    return tail / (np.abs(truncated) + tail + eps)


def compute_sign_disagreement(full_advantages, truncated_advantages) -> np.ndarray:
    """Opposite strictly nonzero signs; zero versus nonzero is not disagreement."""
    full, truncated = _paired(full_advantages, truncated_advantages)
    return ((full < 0) & (truncated > 0)) | ((full > 0) & (truncated < 0))


def compute_event_credit(
    rewards,
    values,
    terminated,
    event_index: int,
    gamma: float = 0.995,
    lam: float = 0.95,
    horizon: int | str = "full",
) -> np.ndarray:
    """Direct credit from one observed reward, holding all critic values fixed."""
    return compute_segment_credit(
        rewards, values, terminated, event_index, event_index + 1, gamma, lam, horizon
    )


def compute_segment_credit(
    rewards,
    values,
    terminated,
    start: int,
    stop: int,
    gamma: float = 0.995,
    lam: float = 0.95,
    horizon: int | str = "full",
) -> np.ndarray:
    """Credit from zeroing the reward interval [start, stop), with fixed values."""
    rewards = _array(rewards, "rewards")
    if any(
        isinstance(index, bool) or not isinstance(index, (int, np.integer))
        for index in (start, stop)
    ):
        raise ValueError("segment indices must be integers")
    if not 0 <= start <= stop <= len(rewards):
        raise ValueError("segment must lie within rewards")
    original = compute_gae(rewards, values, terminated, gamma, lam, horizon).advantages
    masked = rewards.copy()
    masked[start:stop] = 0.0
    counterfactual = compute_gae(masked, values, terminated, gamma, lam, horizon).advantages
    return original - counterfactual

"""Authoritative, versioned schemas for local experiment tables."""

import pyarrow as pa

SCHEMA_VERSION = 1


def _field(name: str, dtype: pa.DataType, *, nullable: bool = False) -> pa.Field:
    return pa.field(name, dtype, nullable=nullable)


UPDATES_SCHEMA = pa.schema(
    [
        _field("run_id", pa.string()),
        _field("update_index", pa.int64()),
        _field("env_steps", pa.int64()),
    ]
    + [
        _field(name, pa.float64())
        for name in (
            "actor_loss",
            "critic_loss",
            "entropy",
            "action_std",
            "actor_grad_norm",
            "critic_grad_norm",
            "approx_kl",
            "clip_fraction",
            "advantage_mean",
            "advantage_std",
            "advantage_min",
            "advantage_max",
            "td_error_mean",
            "td_error_std",
            "target_mean",
            "target_std",
            "prediction_mean",
            "prediction_std",
            "explained_variance",
            "mean_train_return",
            "mean_base_dense_return",
            "elapsed_seconds",
            "epochs_completed",
        )
    ]
)

EPISODES_SCHEMA = pa.schema(
    [
        _field("run_id", pa.string()),
        _field("episode_id", pa.int64()),
        _field("episode_seed", pa.int64()),
        _field("split", pa.string()),
        _field("update_index", pa.int64()),
        _field("env_steps", pa.int64()),
        _field("train_return", pa.float64()),
        _field("base_dense_return", pa.float64()),
        _field("length", pa.int64()),
        _field("first_upright_timestep", pa.int64(), nullable=True),
        _field("stable_success", pa.bool_()),
        _field("upright_fraction", pa.float64()),
        _field("longest_upright_streak", pa.int64()),
        _field("angle_cost", pa.float64()),
        _field("velocity_cost", pa.float64()),
        _field("torque_cost", pa.float64()),
        _field("torque_energy", pa.float64()),
    ]
)

EVALUATIONS_SCHEMA = pa.schema(
    [
        _field("run_id", pa.string()),
        _field("eval_index", pa.int64()),
        _field("env_steps", pa.int64()),
        _field("mean_train_return", pa.float64()),
        _field("mean_base_dense_return", pa.float64()),
        _field("stable_success_rate", pa.float64()),
        _field("mean_upright_fraction", pa.float64()),
        _field("episodes", pa.int64()),
    ]
)

TABLE_SCHEMAS = {
    "updates": UPDATES_SCHEMA,
    "episodes": EPISODES_SCHEMA,
    "evaluations": EVALUATIONS_SCHEMA,
}

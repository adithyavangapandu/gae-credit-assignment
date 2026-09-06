"""Compact Vertex Experiments metrics, with no artifact ownership."""


class VertexMetricsLogger:
    def __init__(self, gcp, run_id, config, *, resume=False, sdk=None):
        if sdk is None:
            from google.cloud import aiplatform as sdk
        self.sdk = sdk
        sdk.init(
            project=gcp.project_id,
            location=gcp.region,
            experiment=gcp.vertex_experiment,
            experiment_tensorboard=gcp.tensorboard,
        )
        sdk.start_run(run_id, resume=resume)
        sdk.log_params(
            {
                "run_id": run_id,
                "config_hash": config.config_hash(),
                "algorithm": config.algorithm,
                "reward": config.reward.kind,
                "delay": config.reward.delay_block_size,
                "actor_horizon": str(config.estimator.actor_horizon),
                "critic_horizon": str(config.estimator.critic_horizon),
                "seed": config.seed,
            }
        )

    def log_update(self, row):
        keys = (
            "actor_loss",
            "critic_loss",
            "advantage_std",
            "explained_variance",
            "entropy",
            "actor_grad_norm",
            "critic_grad_norm",
            "approx_kl",
            "clip_fraction",
            "env_steps",
        )
        metrics = {key: float(row[key]) for key in keys}
        self.sdk.log_metrics(metrics)
        self.sdk.log_time_series_metrics(metrics, step=int(row["env_steps"]))

    def log_evaluation(self, row):
        metrics = {
            "evaluation_dense_return": float(row["mean_base_dense_return"]),
            "stable_upright_success_rate": float(row["stable_success_rate"]),
        }
        self.sdk.log_metrics(metrics)
        self.sdk.log_time_series_metrics(metrics, step=int(row["env_steps"]))

    def close(self, *, failed=False):
        # Use the SDK's documented protobuf enum rather than a free-form string.
        from google.cloud.aiplatform_v1.types import Execution

        self.sdk.end_run(state=Execution.State.FAILED if failed else Execution.State.COMPLETE)

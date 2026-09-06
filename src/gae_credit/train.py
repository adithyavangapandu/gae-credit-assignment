"""Run the local PPO pipeline: python -m gae_credit.train --config CONFIG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from gae_credit.algorithms.networks import Actor, Critic
from gae_credit.algorithms.ppo import PPO, prepare_batch
from gae_credit.algorithms.rollout import collect_rollout
from gae_credit.checkpoint import (
    CHECKPOINT_VERSION,
    TrainingInterrupted,
    capture_rng,
    load_checkpoint,
    restore_rng,
    runtime_fingerprint,
    save_checkpoint,
)
from gae_credit.cloud.config import run_id_for, source_digest
from gae_credit.config import StudyConfig, config_from_dict, derive_seeds, load_config
from gae_credit.envs.pendulum import PendulumEnv
from gae_credit.envs.rewards import make_reward
from gae_credit.estimators.gae import (
    compute_event_credit,
    compute_gae,
    compute_omitted_tail,
    compute_sign_disagreement,
    compute_tail_fraction,
)
from gae_credit.logging import RunLogger, validate_run


def _model_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_training(
    config: StudyConfig,
    *,
    overwrite: bool = False,
    verbose: bool = True,
    context=None,
    resume_path=None,
    artifact_logger=None,
    metrics_logger=None,
    stop_after_update=None,
) -> Path:
    """Collect whole episodes, update PPO, evaluate, and close validated artifacts.

    CPU and one thread are intentional for the local reproducibility contract.
    Evaluation uses its own environment and the same fixed seeds at each point.
    """
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    context = {
        "phase": "local",
        "image_digest": None,
        **(context or {}),
        "source_digest": source_digest(),
    }
    context["run_id"] = run_id_for(config, context["phase"])
    if overwrite and artifact_logger is not None:
        raise ValueError("Cloud runs cannot be overwritten; use checkpoint resume")
    restored = load_checkpoint(resume_path, config, context) if resume_path else None
    if restored:
        context["resumed_from_update"] = restored["update_index"]
        context["elapsed_before_resume_seconds"] = restored["elapsed_seconds"]
    seeds = derive_seeds(config)
    model_seeds = np.random.SeedSequence(seeds["model"]).generate_state(2)
    actor = Actor(
        4,
        config.optimizer.hidden_size,
        config.environment.max_torque,
        config.optimizer.init_log_std,
        seed=int(model_seeds[0]),
    )
    critic = Critic(4, config.optimizer.hidden_size, seed=int(model_seeds[1]))
    optimizer = PPO(actor, critic, config.optimizer)
    action_rng = torch.Generator().manual_seed(seeds["action"])
    update_rng = torch.Generator().manual_seed(seeds["optimization"])
    evaluation_rng = torch.Generator().manual_seed(seeds["evaluation"])
    env_rng = np.random.default_rng(seeds["environment"])
    diagnostic_rng = np.random.default_rng(seeds["diagnostics"])
    evaluation_seeds = np.random.default_rng(seeds["evaluation"]).integers(
        0, 2**32, size=config.evaluation.episodes, dtype=np.uint32
    )
    env = PendulumEnv(config.environment)
    eval_env = PendulumEnv(config.environment)
    reward = make_reward(config.reward, config.estimator.gamma)
    eval_reward = make_reward(config.reward, config.estimator.gamma)
    logger = RunLogger(
        config,
        overwrite=overwrite,
        seeds=seeds,
        context=context,
        resume_snapshot=restored["logger_snapshot"] if restored else None,
        artifact_logger=artifact_logger,
        metrics_logger=metrics_logger,
    )
    initial_actor_hash = _model_hash(actor)
    initial_critic_hash = _model_hash(critic)
    start_time = time.perf_counter()
    episode_id = 0
    evaluation_rows = []
    rollout_size = config.environment.max_steps * config.training.episodes_per_rollout
    update_count = config.training.total_env_steps // rollout_size
    first_update = 1
    elapsed_before = 0.0
    if restored:
        actor.load_state_dict(restored["actor"])
        critic.load_state_dict(restored["critic"])
        optimizer.actor_optimizer.load_state_dict(restored["actor_optimizer"])
        optimizer.critic_optimizer.load_state_dict(restored["critic_optimizer"])
        restore_rng(
            restored["rng"], action_rng, update_rng, evaluation_rng, env_rng, diagnostic_rng
        )
        env.set_state(restored["environment_state"])
        eval_env.set_state(restored["evaluation_environment_state"])
        reward.set_state(restored["reward_state"])
        eval_reward.set_state(restored["evaluation_reward_state"])
        first_update = restored["update_index"] + 1
        episode_id = restored["episode_id"]
        evaluation_rows = restored["evaluation_rows"]
        initial_actor_hash = restored["initial_actor_hash"]
        initial_critic_hash = restored["initial_critic_hash"]
        elapsed_before = restored["elapsed_seconds"]
        start_time = time.perf_counter()

    def checkpoint(update_index, filename):
        payload = {
            "checkpoint_schema_version": CHECKPOINT_VERSION,
            "run_id": context["run_id"],
            "config": config.to_dict(),
            "config_hash": config.config_hash(),
            "source_digest": context["source_digest"],
            "image_digest": context["image_digest"],
            "runtime_fingerprint": runtime_fingerprint(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_optimizer": optimizer.actor_optimizer.state_dict(),
            "critic_optimizer": optimizer.critic_optimizer.state_dict(),
            "update_index": update_index,
            "env_steps": update_index * rollout_size,
            "rng": capture_rng(action_rng, update_rng, evaluation_rng, env_rng, diagnostic_rng),
            "environment_state": env.get_state(),
            "evaluation_environment_state": eval_env.get_state(),
            "reward_state": reward.get_state(),
            "evaluation_reward_state": eval_reward.get_state(),
            "episode_id": episode_id,
            "evaluation_rows": evaluation_rows,
            "initial_actor_hash": initial_actor_hash,
            "initial_critic_hash": initial_critic_hash,
            "elapsed_seconds": elapsed_before + time.perf_counter() - start_time,
            "logger_snapshot": logger.snapshot(),
        }
        path = logger.path / "checkpoints" / filename
        save_checkpoint(path, payload)
        logger.checkpoint(path, update_index)
        return path

    def record_episodes(episodes, split, update_index, env_steps):
        nonlocal episode_id
        for index, episode in enumerate(episodes):
            step = env_steps
            if split == "train":
                step = env_steps - rollout_size + (index + 1) * config.environment.max_steps
            logger.log_episode(
                {
                    **episode.metrics,
                    "episode_id": episode_id,
                    "split": split,
                    "update_index": update_index,
                    "env_steps": step,
                }
            )
            episode_id += 1

    def evaluate(update_index, env_steps):
        episodes = collect_rollout(
            eval_env,
            eval_reward,
            actor,
            critic,
            evaluation_seeds,
            evaluation_rng,
            deterministic=True,
        )
        record_episodes(episodes, "evaluation", update_index, env_steps)
        row = {
            "eval_index": len(evaluation_rows),
            "env_steps": env_steps,
            "episodes": len(episodes),
            "mean_train_return": float(np.mean([e.metrics["train_return"] for e in episodes])),
            "mean_base_dense_return": float(
                np.mean([e.metrics["base_dense_return"] for e in episodes])
            ),
            "stable_success_rate": float(np.mean([e.metrics["stable_success"] for e in episodes])),
            "mean_upright_fraction": float(
                np.mean([e.metrics["upright_fraction"] for e in episodes])
            ),
        }
        logger.log_evaluation(row)
        evaluation_rows.append(row)

    try:
        if not restored:
            evaluate(0, 0)
        for update_index in range(first_update, update_count + 1):
            episode_seeds = env_rng.integers(
                0, 2**32, size=config.training.episodes_per_rollout, dtype=np.uint32
            )
            episodes = collect_rollout(env, reward, actor, critic, episode_seeds, action_rng)
            env_steps = update_index * rollout_size
            record_episodes(episodes, "train", update_index, env_steps)
            batch = prepare_batch(
                episodes,
                config.estimator.gamma,
                config.estimator.lam,
                config.estimator.actor_horizon,
                config.estimator.critic_horizon,
            )
            metrics = optimizer.update(batch, update_rng)
            row = {
                **metrics,
                "update_index": update_index,
                "env_steps": env_steps,
                "mean_train_return": float(np.mean([e.metrics["train_return"] for e in episodes])),
                "mean_base_dense_return": float(
                    np.mean([e.metrics["base_dense_return"] for e in episodes])
                ),
                "elapsed_seconds": elapsed_before + time.perf_counter() - start_time,
            }
            logger.log_update(row)
            if config.logging.save_diagnostics:
                episode = episodes[int(diagnostic_rng.integers(len(episodes)))]
                arrays = {
                    "rewards": episode.rewards,
                    "values": episode.values,
                    "terminated": episode.terminated,
                }
                results = {}
                for horizon in (1, 3, 16, "full"):
                    results[horizon] = compute_gae(
                        episode.rewards,
                        episode.values,
                        episode.terminated,
                        config.estimator.gamma,
                        config.estimator.lam,
                        horizon,
                    )
                    arrays[f"advantages_{horizon}"] = results[horizon].advantages
                    arrays[f"effective_horizon_{horizon}"] = results[horizon].effective_horizons
                full, short = results["full"].advantages, results[3].advantages
                arrays.update(
                    omitted_tail_h3=compute_omitted_tail(full, short),
                    tail_fraction_h3=compute_tail_fraction(full, short),
                    sign_disagreement_h3=compute_sign_disagreement(full, short),
                    final_event_credit_h3=compute_event_credit(
                        episode.rewards,
                        episode.values,
                        episode.terminated,
                        len(episode.rewards) - 1,
                        config.estimator.gamma,
                        config.estimator.lam,
                        3,
                    ),
                )
                logger.write_diagnostics(f"update-{update_index:05d}", arrays)
            if (
                env_steps % config.evaluation.interval_env_steps == 0
                or update_index == update_count
            ):
                evaluate(update_index, env_steps)
            interval = config.training.checkpoint_interval_env_steps
            if (interval and env_steps % interval == 0) or stop_after_update == update_index:
                checkpoint(update_index, f"update-{update_index:06d}.pt")
            if stop_after_update == update_index:
                raise TrainingInterrupted(f"Stopped after checkpoint for update {update_index}")
            if verbose:
                print(
                    json.dumps(
                        {
                            "run_id": context["run_id"],
                            "update": update_index,
                            "env_steps": env_steps,
                            "approx_kl": metrics["approx_kl"],
                        }
                    ),
                    flush=True,
                )
        final_actor_hash = _model_hash(actor)
        final_critic_hash = _model_hash(critic)
        if initial_actor_hash == final_actor_hash or initial_critic_hash == final_critic_hash:
            raise RuntimeError("Smoke pipeline did not change both actor and critic parameters")
        if config.logging.save_checkpoint:
            checkpoint(update_count, "final.pt")
        # Recorded for artifact completeness; smoke AUC is not experimental evidence.
        auc = float(
            np.trapezoid(
                [r["mean_base_dense_return"] for r in evaluation_rows],
                [r["env_steps"] for r in evaluation_rows],
            )
            / config.training.total_env_steps
        )
        logger.complete(
            {
                "total_env_steps": config.training.total_env_steps,
                "updates": update_count,
                "episodes": episode_id,
                "evaluation_count": len(evaluation_rows),
                "actor_parameters_changed": True,
                "critic_parameters_changed": True,
                "initial_actor_hash": initial_actor_hash,
                "final_actor_hash": final_actor_hash,
                "initial_critic_hash": initial_critic_hash,
                "final_critic_hash": final_critic_hash,
                "evaluation_base_dense_auc": auc,
                "purpose": "pipeline validation; no performance claim",
            }
        )
    except BaseException as error:
        logger.fail(error)
        raise
    finally:
        env.close()
        eval_env.close()
    validate_run(logger.path)
    return logger.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output-dir", type=str, help="Override logging.output_dir (changes run hash)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace this exact existing run")
    parser.add_argument(
        "--cloud", action="store_true", help="Enable ADC-backed GCS and Vertex adapters"
    )
    parser.add_argument("--resume", type=Path, help="Resume a matching local checkpoint")
    parser.add_argument(
        "--resume-latest", action="store_true", help="Resume the cloud prefix's latest checkpoint"
    )
    parser.add_argument(
        "--stop-after-update",
        type=int,
        help="Checkpoint and exit unsuccessfully to exercise recovery",
    )
    args = parser.parse_args()
    if args.config is not None:
        config = load_config(args.config)
    elif args.cloud and "GAE_RESOLVED_CONFIG" in os.environ:
        config = config_from_dict(json.loads(os.environ["GAE_RESOLVED_CONFIG"]))
    else:
        parser.error("--config is required unless a cloud job supplies GAE_RESOLVED_CONFIG")
    if args.cloud and (args.overwrite or args.output_dir or args.resume):
        parser.error(
            "Cloud runs use the submitted config and --resume-latest; local overrides are disallowed"
        )
    if not args.cloud and args.resume_latest:
        parser.error("--resume-latest requires --cloud")
    if args.resume and args.overwrite:
        parser.error("Use either --resume or --overwrite, not both")
    if args.stop_after_update is not None and args.stop_after_update < 1:
        parser.error("--stop-after-update must be positive")
    if args.output_dir is not None:
        config = replace(config, logging=replace(config.logging, output_dir=args.output_dir))
    if args.cloud:
        from gae_credit.cloud.runner import run_cloud

        path = run_cloud(
            config, resume_latest=args.resume_latest, stop_after_update=args.stop_after_update
        )
    else:
        path = run_training(
            config,
            overwrite=args.overwrite,
            resume_path=args.resume,
            stop_after_update=args.stop_after_update,
        )
    print(f"Validated run: {path}")


if __name__ == "__main__":
    main()

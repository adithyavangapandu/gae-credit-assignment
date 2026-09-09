# Local experiment data dictionary

Schema version 1 is defined in `src/gae_credit/logging/schemas.py`. Every field
is nonnullable except `episodes.first_upright_timestep`. Numeric values must be
finite. Tables use Zstandard-compressed Parquet. Failed runs are marked failed
and rejected by validation. Logging is independent of Google Cloud.

## Identity and counters

`run_id` appears in every row: algorithm, reward (including D when delayed),
horizon, training seed, and first 12 characters of the SHA-256 config hash.
The full hash covers sorted compact JSON of all resolved settings, including
the output directory. YAML order and omitted defaults do not change identity.

`env_steps` always counts training transitions, excluding evaluation. Update
indices start at 1; evaluation indices at 0; episode IDs are globally consecutive
from 0 across all splits. Initial evaluation has update index 0.

## updates.parquet

One row per update. `run_id` is string, `update_index` and `env_steps` are int64;
all other fields are float64. Standard deviations and variances are population statistics.

| Column | Meaning |
|---|---|
| run_id | Shared identity |
| update_index | Consecutive PPO update number from 1 |
| env_steps | Training transitions after collection for this update |
| actor_loss | Sample-weighted mean negative clipped surrogate over visited minibatches/epochs, before entropy bonus |
| critic_loss | Sample-weighted mean squared target error, before value coefficient |
| entropy | Monte Carlo differential entropy of the scaled tanh-transformed policy; can be negative |
| action_std | Learned base Gaussian standard deviation before tanh, after update |
| actor_grad_norm | Mean actor objective gradient norm before clipping |
| critic_grad_norm | Mean coefficient-scaled critic gradient norm before clipping |
| approx_kl | Final rollout mean of `(ratio-1)-log(ratio)` |
| clip_fraction | Final fraction with `abs(ratio-1)>clip_epsilon` |
| advantage_mean | Mean raw actor advantage, before normalization |
| advantage_std | Standard deviation of raw actor advantage |
| advantage_min | Minimum raw advantage |
| advantage_max | Maximum raw advantage |
| td_error_mean | Mean rollout TD residual |
| td_error_std | Standard deviation of TD residuals |
| target_mean | Mean fixed critic targets from critic GAE |
| target_std | Standard deviation of critic targets |
| prediction_mean | Mean pre-update rollout critic prediction |
| prediction_std | Standard deviation of pre-update critic prediction |
| explained_variance | `1-var(target-old_prediction)/var(target)`; defined as zero when target variance <=1e-12 |
| mean_train_return | Mean undiscounted delivered return of training episodes |
| mean_base_dense_return | Mean undiscounted physical dense return of training episodes |
| elapsed_seconds | Cumulative wall time since starting evaluation/collection, measured at update logging |
| epochs_completed | PPO epochs performed; can be below maximum due to target KL |

Advantages logged here remain raw even when normalized for the actor objective.
KL and clip fraction describe the final policy on the whole rollout; loss and
gradient metrics average all visited samples, including repeated epochs.

## episodes.parquet

One row per full training, evaluation, or diagnostic episode.

| Column | Type | Meaning |
|---|---|---|
| run_id | string | Shared identity |
| episode_id | int64 | Globally unique consecutive ID |
| episode_seed | int64 | uint32 seed passed to environment reset |
| split | string | train, evaluation, or diagnostic (reserved) |
| update_index | int64 | Owning update; zero for initial evaluation |
| env_steps | int64 | Training step at episode end; checkpoint step for evaluation |
| train_return | float64 | Sum of delivered rewards, not discounted |
| base_dense_return | float64 | Sum of untransformed dense rewards |
| length | int64 | Transition count, exactly configured finite horizon |
| first_upright_timestep | nullable int64 | First qualifying post-step state, 1-based; null means never upright |
| stable_success | bool | At least ten consecutive qualifying post-step states |
| upright_fraction | float64 | Qualifying transitions divided by length |
| longest_upright_streak | int64 | Maximum consecutive qualifying steps |
| angle_cost | float64 | Sum of squared wrapped post-step angles |
| velocity_cost | float64 | Sum of 0.1 times squared post-step angular velocity |
| torque_cost | float64 | Sum of 0.001 times squared applied torque |
| torque_energy | float64 | Sum of applied torque squared times dt; effort proxy, not mechanical work |

Base dense return is minus the sum of the three cost columns, up to rounding.
Delayed rewards preserve discounted return; undiscounted train_return can differ.

## evaluations.parquet

One row per initial, interval, or final evaluation. `run_id` is string;
`eval_index`, `env_steps`, and `episodes` are int64; the means/rates are float64.

| Column | Meaning |
|---|---|
| run_id | Shared identity |
| eval_index | Consecutive checkpoint number from 0 |
| env_steps | Training step at evaluation |
| mean_train_return | Mean delivered return of fixed evaluation episodes |
| mean_base_dense_return | Mean dense return, the primary learning-curve ordinate |
| stable_success_rate | Fraction with a ten-step upright streak |
| mean_upright_fraction | Mean episode upright fraction |
| episodes | Number of evaluation episodes |

Evaluation uses fixed reset seeds and the transformed policy mean. It never
advances the training environment or action RNG.

## Manifest and configuration

`resolved_config.yaml` contains every default/override including both horizons
and seed lists. Manifest fields: `schema_version`, `run_id`, full `config_hash`,
`status` (running/completed/failed), UTC ISO `started_at` and `finished_at`,
`git_commit` and `git_dirty` (nullable when unavailable), `python_version`,
`platform`, `dependency_versions`, master `seed`, derived `seeds` mapping,
`actor_horizon`, and `critic_horizon`. Completed manifests add `row_counts`;
failed manifests add `error`. No credentials are read or saved.

## Diagnostics

Each update saves one episode selected with the independent diagnostic RNG.
NPZ files contain finite numeric arrays, no pickle: `rewards[T]`, `values[T+1]`,
`terminated[T]`; `advantages_1/3/16/full[T]` and corresponding
`effective_horizon_1/3/16/full[T]`; `omitted_tail_h3[T]`, `tail_fraction_h3[T]`,
`sign_disagreement_h3[T]`, and `final_event_credit_h3[T]`. Definitions are in
[mathematical_definitions.md](mathematical_definitions.md). These estimator inputs
permit offline recomputation.

Confirmatory runs additionally store
`diagnostic_trajectories/checkpoint-NNNNNNN.parquet`. Each checkpoint file has 20
fixed-seed deterministic evaluation trajectories and 4,000 per-step rows. Fields
include pre/post observation, bounded and latent action, value prediction,
training and base-dense rewards, physical state, applied torque, upright and
terminal indicators, and delayed-reward payout metadata. Diagnostic collection
uses the evaluation stream and does not consume training steps or training RNG.

Reward transforms return transient dictionaries to the collector. Dense adds
none. Delayed returns zero-based `block_id`, one-based `block_position`,
`emitted_payout`, `payout_count` (zero if none), `pending_reward` (undiscounted
buffer after payout), `discount_corrected_payout`, and `pending_corrected_reward`
(after payout). Sparse returns `upright`, `entered_upright`, `left_upright`,
`upright_streak`, `longest_upright_streak`, nullable `first_upright_timestep`,
and `stable_success`. The default trainer does not persist these transient dictionaries.

## Checkpoint and summary

`checkpoints/final.pt` contains actor/critic state dictionaries, both Adam states,
resolved config/hash, and final env_steps. Day 4 extends this format for recovery,
as described below.

`summary.json` contains identity/hash, `total_env_steps`, `updates`, `episodes`
(all splits), `evaluation_count`, `actor_parameters_changed`,
`critic_parameters_changed`, initial/final actor/critic SHA-256 parameter hashes,
`evaluation_base_dense_auc` (trapezoid area divided by training-step budget), and
`purpose`. Smoke AUC is not used for inference.

## Validation

`scripts/validate_run.py` checks paths, completion, exact Arrow schemas, finite
fields, IDs/counts, step increments, evaluation schedule and aggregates, config
hash, horizons, diagnostics, and checkpoint presence. Corruption tests verify
rejection. Repeated local seed/config runs must match table values (excluding
elapsed time), diagnostic arrays, and final model hashes. Timestamps and platform
metadata may differ. Generated files are ignored by Git.

## Day 4 extensions (Parquet schema remains version 1)

New manifests carry `artifact_protocol: 2`, `phase`, a SHA-256 `source_digest`,
and nullable `image_digest` (required for cloud runs). Cloud manifests also carry
`vertex_run_id`, `artifact_prefix`, `vertex_experiment`, `gcp_project`, `gcp_region`,
and `machine_type`. Resumed manifests add `resumed_from_update` and
`elapsed_before_resume_seconds`. Cloud IDs use hyphens and include phase to meet
Vertex naming/label rules; local Day 3 identities remain unchanged. The disabled
checkpoint interval (0) is omitted from canonical YAML/hash to retain legacy identities.

`validation_report.json` contains the result of the same semantic validator.
`checksums.json` version 1 maps each authoritative relative artifact path to its
SHA-256 and byte size. `_SUCCESS` contains the checksum file's SHA-256 and is
written last. Downloaded bundles must pass both semantic and checksum validation.
Legacy Day 3 manifests without artifact_protocol remain readable.

Checkpoint schema version 1 adds run ID, source/image/runtime fingerprints,
update counter, original actor/critic hashes, accumulated elapsed time, all RNG
states, environment/reward snapshots, and historical table rows/diagnostic bytes.
Checkpoints are taken only after full updates; recovery rebuilds logs through
that boundary, discarding any incomplete work after it. Checkpoint payloads load
with PyTorch's restricted `weights_only=True` loader. See
[day4_cloud.md](day4_cloud.md) for cloud recovery pointers and lease semantics.

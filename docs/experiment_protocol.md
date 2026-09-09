# Experiment protocol: gae-pendulum-v1

Status: excluded pilot preparation, before final experimental training. The
smoke study `gae-pendulum-smoke-v1` checks software validity only. Its returns
must not be used to select a winning horizon or support a performance claim.

## Question and primary hypothesis

Does the advantage-estimation horizon interact with reward delivery timing?
The preregistered directional contrast is

\[
(\mathrm{AUC}_{\mathrm{full,delayed}}-\mathrm{AUC}_{3,\mathrm{delayed}})
> (\mathrm{AUC}_{\mathrm{full,dense}}-\mathrm{AUC}_{3,\mathrm{dense}}).
\]

The primary delayed condition is D=32. D=8 is a secondary timing condition;
sparse repeated upright reward is a separate task-objective comparison. This
distinction avoids selecting whichever delay looks best after observing results.

## Fixed environment

All conditions use the same local Pendulum implementation: g=10, m=1, l=1,
dt=0.05 seconds, maximum speed 8 rad/s, maximum applied torque 2. Initialization
draws theta uniformly from [-pi,pi) and angular velocity independently from
[-1,1]. This broad distribution is an explicit replacement for inconsistent
initializers in older scripts. Original files in the sibling `RL` directory
remain untouched and are never imported.

Velocity is advanced with acceleration `3*g/(2*l)*sin(theta)+3*u/(m*l*l)`,
clipped to the speed limit, then angle is advanced using the new velocity and
wrapped to [-pi,pi). Zero angle is upright. Costs are evaluated at the **new**
state; this differs from Gymnasium Pendulum's pre-transition cost convention.
Actions are clipped before dynamics and torque diagnostics.

Observations are `[cos(theta), sin(theta), angular_velocity, (200-t)/200]`.
Each episode contains exactly 200 transitions. The final transition returns
`terminated=True, truncated=False`, has bootstrap value zero, and flushes any
reward buffer. No success termination is used. GAE is computed separately per
episode and cannot cross a reset. Environment snapshots include physical state,
elapsed steps, and RNG state; reward mechanisms have separate snapshots.

## Reward and estimator conditions

| Condition | Training reward |
|---|---|
| dense | Negative post-step angle squared + 0.1 velocity squared + 0.001 torque squared |
| delayed D=8 or D=32 | Zero except discount-corrected block payouts; partial final block flushed |
| sparse | One at every post-step state inside the versioned upright region; zero otherwise |

The dense row means the negative of the **sum** of the three costs. The exact
delay correction is defined in [mathematical_definitions.md](mathematical_definitions.md).
It preserves the gamma-discounted whole-episode dense return for every trajectory;
it does not preserve the undiscounted sum. Both dense base return and training
return are therefore logged. Sparse reward never adds dense shaping costs.

Gamma is 0.995, lambda is 0.95, and horizons are H=1,3,16,full. Full means all
remaining residuals through the true terminal boundary. Initial actor and critic
target horizons match and are separately recorded. Actor advantage normalization
does not change the critic targets. Distinct horizons for a later actor-only
ablation require a protocol/config validation change; the schema already supports them.

**Interpretation limit:** the block-delayed reward buffer is hidden from the
required four-dimensional observation. A feedforward critic cannot represent an
exact Markov value function for that observation in general. Timing comparisons
therefore include this partial-observability effect. Buffer snapshots support
future continuation experiments; a memory or augmented-observation ablation is
future work. Neither reward conservation nor a passing smoke run resolves this
confound. Sparse reward changes the objective in addition to reward sparsity.

## Training, seeds, and evaluation

The planned final budget is 1,000,000 training transitions per condition/seed,
in rollouts of five complete episodes (1,000 transitions). Final training seeds
are 0 through 9; pilot seeds 100,101,102 are excluded from final inference.
No pilot or million-step runs are part of Days 1–3.

NumPy SeedSequence derives independent model, training-environment, action,
optimization, and diagnostic streams from the training seed. Model initialization
uses child seeds for actor and critic. The evaluation stream derives from fixed
seed 20260905 independently of training seed. Each evaluation reuses the same 20
episode seeds across all checkpoints, conditions, and training seeds. Evaluation
uses the transformed policy mean (no sampled actions), a separate environment,
and separate reward mechanism. Its steps do not consume the training budget.

Evaluate at step 0, every 10,000 training steps, and the final step. Primary
outcome is trapezoidal AUC of the mean **base dense** evaluation return against
training environment steps, divided by the total training budget. No reward-value
normalization is applied. Higher (less negative) is better. Report paired
seed-level contrasts, individual seed curves, and uncertainty across training
seeds; evaluation episodes are not independent training replicates. A paired
bootstrap over the ten seed-level contrasts (10,000 draws, fixed analysis seed
20260906) will produce 95% percentile intervals. Secondary comparisons remain
labeled secondary rather than replacing the D=32 primary contrast.

The single primary interaction contrast is tested without a multiplicity
adjustment. Secondary horizon/reward contrasts form one family controlled with
Holm adjustment; exploratory diagnostics remain descriptive. Dense H=3 is
declared noninferior to dense full GAE when the lower endpoint of the paired 95%
bootstrap interval for `AUC(H3)-AUC(full)` exceeds -100 base-dense-return
points. This margin is fixed before confirmatory outcomes. Infrastructure-
corrupted runs may resume from a verified checkpoint or rerun with the identical
seed and configuration. Algorithmic divergence remains an observed result and
is never relabeled or automatically replaced.

Planned plots are individual-seed learning curves, paired-seed contrasts, reward
distributions, KL/clipping and critic-fit traces, action dispersion/saturation,
and credit-diagnostic summaries. No interim confirmatory result may change these
endpoints, plots, or decision rules.

Secondary metrics are stable-success rate, upright fraction, and final dense
return. Stable success requires at least ten consecutive qualifying post-step
states anywhere in the episode. A one-step upright fly-through is not success.
Training return, losses, KL, clip fraction, entropy, critic fit, torque energy,
omitted-tail magnitude, and direct reward credit are diagnostics.

The two dense smoke configs run two updates, five 200-step episodes per update,
seed 0, two evaluation episodes at step 0/1000/2000, and two optimizer epochs.
Smaller 20-step integration-test environments check code paths only. Smoke and
test overrides do not alter the final protocol. Run identity includes the full
resolved config (including output directory); elapsed time and platform metadata
are excluded from reproducibility comparisons, not from logging.

## Optimization and artifacts

Actor and critic each have two 64-unit tanh layers. The actor has a learned
state-independent log standard deviation, initialized to -0.5, and uses a
Gaussian transformed by tanh then scaled to the torque bound. PPO stores latent
actions and applies the exact stable log-Jacobian correction. Optimization uses
Adam, actor LR 0.0003, critic LR 0.001, ten epochs, 250-sample minibatches,
clipping epsilon 0.2, value coefficient 0.5, entropy coefficient 0.001, gradient
norm limit 1.0, and target KL 0.01 with stopping between epochs. Entropy is a
Monte Carlo estimate for the transformed policy; `action_std` is the Gaussian
standard deviation before transformation. No hyperparameter tuning occurs here.

All runs use CPU and one PyTorch thread. The package versions are fixed by
`uv.lock`; platform-to-platform bitwise equivalence is not claimed. Local runs
require no cloud account. Manifest, resolved config, three typed Parquet tables,
diagnostic arrays, final checkpoint, and summary are validated on completion.
See [data_dictionary.md](data_dictionary.md). Checkpoints support exact
update-boundary recovery. Confirmatory runs save 20 fixed-seed raw diagnostic
trajectories at each 50,000-step checkpoint. Generated runs, credentials, PDFs,
and old scripts are excluded from Git history.

## Protocol changelog

- 2026-09-05: Initial protocol established from the Days 1–3 plan. Chose a shared
  broad initial distribution, explicit final/pilot/evaluation seeds, D=32 as the
  primary delay contrast, and the fixed evaluation/AUC/inference protocol to
  remove unspecified decisions before training. Documented the hidden-buffer
  confound and the difference between post-transition and Gymnasium costs.
- Future changes must append date, exact change, rationale, and whether any
  pilot/final outcomes were available when the change was made.
- 2026-09-06: Added optional cloud execution and update-boundary checkpoint
  recovery. The smoke uses excluded seed 100 for 5,000 steps; pilot/final reward,
  environment, estimator, evaluation and training contracts are unchanged. No
  pilot/final outcomes were available. Day 4 does not freeze final settings.
- 2026-09-07: The excluded stochastic-policy calibration observed entry in
  21.03% of 3,000 episodes at the original 0.262-radian/1.0-rad/s region. Per the
  preregistered one-adjustment rule, the angle bound was tightened once to 10
  degrees (0.1745329252 radians), retaining the 1.0-rad/s velocity bound. The
  adjusted calibration observed nonzero entry in 16.90% of 3,000 episodes and
  10-step success in 6.17%. No trained pilot or confirmatory outcomes existed.
  The single-adjustment limit is exhausted; this threshold is used for the pilot.
- 2026-09-09: All 24 excluded pilot runs passed operational quality control.
  Frozen the confirmatory PPO hyperparameters globally, added 20 raw diagnostic
  evaluation trajectories at each 50,000-step checkpoint, and specified the
  noninferiority, multiplicity, plotting, and failure rules above. These changes
  follow the preregistered plan and operational needs, not pilot treatment
  comparisons. No confirmatory outcomes existed.

## Deferred work

The VPG replication, final hypothesis tests, and publication graphics remain
deferred until the 160-run confirmatory PPO dataset is complete.

# GAE credit assignment

This project studies how the temporal horizon of generalized advantage estimation
affects policy-gradient learning under dense, delayed, and sparse rewards. It
tests whether short-horizon GAE remains competitive when immediate feedback is
informative, while full-horizon GAE becomes more useful when outcomes occur far
after the actions that caused them.

The primary hypothesis is:

\[
\left[\mathrm{AUC}_{\mathrm{full,delayed}}-\mathrm{AUC}_{H=3,\mathrm{delayed}}\right]
>
\left[\mathrm{AUC}_{\mathrm{full,dense}}-\mathrm{AUC}_{H=3,\mathrm{dense}}\right].
\]

Truncated GAE sums at most H discounted TD residuals. Full GAE includes every
remaining residual through the episode boundary. Both use real rewards and
critic predictions. Comparing dense rewards with discounted-return-preserving
delayed rewards helps study how reward timing interacts with that cutoff.

**Current scope: the validated foundation, completed 24-run excluded pilot, and
Day 6 confirmatory PPO launch. No learning performance claims.** Statistical
comparisons wait for the complete dataset. Eventually this repository will accompany a report with the frozen
protocol, seed-level learning curves, credit diagnostics, and reproducible
analysis. The remote remains private during foundation work.

## Quick start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).
Python 3.11.16 is selected by `.python-version`; uv installs it if necessary.
`uv.lock` fixes exact package versions and platform-specific CPU PyTorch wheels.
Run from `RLProject`, the repository root:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m gae_credit.train --config configs/smoke_dense_h3.yaml
uv run python -m gae_credit.train --config configs/smoke_dense_full.yaml
for run in runs/ppo__*; do uv run python scripts/validate_run.py "$run"; done
```

Each smoke config performs two updates, five complete 200-step episodes per
update, and deterministic evaluation at steps 0,1000,2000. Both networks must
change, metrics must be finite, and artifacts must pass validation. Differences
in smoke returns do **not** establish estimator quality.

Inspect a run with `uv run python scripts/inspect_run.py runs/RUN_ID`. Existing
runs are rejected; use `--overwrite` explicitly to replace that exact local run, or
`--output-dir runs/another-location` for a separate identity. Use `--resume` with a
matching checkpoint to recover an interrupted run. `configs/base.yaml` describes the future million-step budget and
is not needed for smoke validation.

## Experiment contract

- Shared Pendulum: 200 steps, dt=0.05, torque in [-2,2], four-value observation
  including time remaining, and zero bootstrap at the true terminal boundary.
- Rewards: post-step dense cost, corrected delayed blocks D=8/32, or repeated
  sparse upright rewards. Stable evaluation success requires ten consecutive steps.
- Estimators: H=1,3,16,full, gamma=0.995, lambda=0.95; actor/critic horizons match initially.
- Separate 64×64 tanh actor/critic; squashed Gaussian actions with stable density
  correction, PPO clipping, gradient clipping, and target-KL epoch stopping.
- Independent model, environment, action, evaluation, diagnostic, and optimizer RNG streams.
- Typed local Parquet data, complete configuration, manifest, diagnostics, and checkpoint.
  No Google Cloud credentials or services required.

The hidden delayed-reward buffer makes the shared four-value observation
partially observable. This limits interpretation of horizon effects and is
documented in the protocol. Sparse reward also changes the task objective.

Read the [experiment protocol](docs/experiment_protocol.md),
[mathematical definitions](docs/mathematical_definitions.md), and
[data dictionary](docs/data_dictionary.md).

The [Days 1–3 validation record](docs/validation.md) records the 127 passing
tests, both local smoke runs, Docker/schema parity, and successful GitHub Actions.

## Day 4 cloud pipeline

See the [cloud execution and recovery guide](docs/day4_cloud.md). Install the
optional SDKs with `uv sync --locked --extra gcp`. The new cloud smoke runs 5,000
steps with excluded seed 100. Submission supports a credential-free `--dry-run`,
uses an immutable image digest, and creates one CPU Vertex Custom Job. Vertex
tracks live metrics; GCS stores the authoritative checksummed artifacts, with
`_SUCCESS` published last. The same validator checks downloaded runs.

The [Day 4 validation record](docs/day4_validation.md) includes the submitted
job link, image digest, completed checks, and commands to verify the cloud result.

Periodic checkpoints preserve model/optimizer states, all RNG streams, reward
buffers, and prior logs. Resume rejects mismatched configuration, code, image,
runtime, or checkpoint schema. The cloud path uses ADC and a dedicated runtime
service account; no keys are stored in this repository.

## Day 5 excluded pilot

See the [pilot workflow](docs/day5_pilot.md) and
[decision record](docs/pilot_decisions.md). Sparse feasibility calibration and
the deterministic 24-run matrix are committed before submission. Pilot seeds
100–102 are excluded from confirmatory inference. Four representative jobs form
the initial gate; the remaining queue cannot be released until those artifacts
pass study-level validation.

All 24 pilot runs passed. The rendered QC report is
[`analysis/pilot_quality_control.html`](analysis/pilot_quality_control.html).
The confirmatory configuration adds 20 per-step diagnostic trajectories at each
50,000-step checkpoint and a checksum-frozen 160-row manifest. Submission is
guarded by `scripts/preflight_day6.py` and staged by
`scripts/submit_confirmatory.py`. `scripts/validate_confirmatory.py` checks the
frozen identities and complete artifact contract without computing treatment
comparisons. The [Day 6 launch boundary](docs/day6_launch.md) records the exact
image, manifest, quota cap, dry run, failed-run reset/resume commands, queue
refill loop, and post-run validation procedure.

## Day 8 analysis

The [Day 8 PPO analysis workspace](analysis/day8/README.md) contains a separately
frozen protocol and numbered scripts for GCS discovery, checksum-validated local
storage, core-table assembly, run-level endpoints, paired seed contrasts,
reward-by-horizon interactions, and credit-mechanism summaries. Generated data
and results stay out of Git. Plotting is deferred.

The [VPG robustness replication](docs/vpg_replication.md) defines the focused
six-cell, 60-run follow-up and its freeze, submission, validation, resume, and
guarded restart commands.

## Artifacts

```text
runs/ppo__dense__h003__seed-000__CONFIG-HASH/
├── manifest.json
├── resolved_config.yaml
├── updates.parquet
├── episodes.parquet
├── evaluations.parquet
├── diagnostics/update-00001.npz
├── checkpoints/final.pt
└── summary.json
```

Validation checks exact schemas, config hashes, horizons, finite values, unique
IDs, counts, evaluation schedules, aggregates, checkpoints, and diagnostics.
Repeated seed/config runs are tested for matching data locally; elapsed time and
machine metadata differ. Cross-platform bitwise equivalence is not claimed.
All generated artifacts are ignored by Git.

## Docker

```bash
docker build -t gae-credit-assignment:smoke .
mkdir -p runs/docker
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/runs/docker:/app/runs" gae-credit-assignment:smoke \
  --config configs/smoke_dense_h3.yaml
for run in runs/docker/*; do uv run python scripts/validate_run.py "$run"; done
```

The image pins Python 3.11.16 and uv 0.12.10, installs from the lockfile, and
defaults to a non-root user. Passing the host UID/GID makes Linux bind-mounted
outputs writable. The container produces the same artifact schemas as the host;
Git metadata may be unavailable because `.git` is excluded from its build context.

## Layout

`src/gae_credit` contains configuration, environment/reward mechanisms, pure
NumPy estimators, networks/collection/PPO, local logging, and the training CLI.
`configs` contains standalone base and smoke YAML files; `docs` defines the
experiment and artifacts; `scripts` inspects/validates runs; `tests` checks units,
integration, reproducibility, and corruption detection. GitHub Actions installs
the lockfile, runs lint/tests, and executes both full smoke configurations.

The earlier scripts in the sibling `RL` directory are retained as references
and never imported or modified. The local repository root is `RLProject`, as
requested, rather than the nested directory in the original plan. License: MIT.

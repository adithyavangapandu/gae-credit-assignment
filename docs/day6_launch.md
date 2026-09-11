# Day 6 confirmatory launch boundary

The confirmatory implementation is frozen at Git tag `experiment-v1` and commit
`2e33a841d06a5bfea41d9f1b8abc4f44c300f027`. Cloud Build
`c2ce9779-5b70-4dab-89d4-e03b35112ac4` passed its image tests and smoke run. The
published image is:

```text
us-central1-docker.pkg.dev/gae-experiment-507805/gae-training/trainer@sha256:be8ee1c8dc3bd54e58fa7e8582f73e10a3b36adc858860d65ea890285aa67fb4
```

The frozen manifest contains 160 unique runs and has SHA-256
`f6239f565eafcfb15020ce9752e85c6ba35e24ee488f7a81d70969a285ed683e`.
The preflight report was `READY` before the canary launch. Seven canaries later
published `_SUCCESS`. The delayed-8, H=3, seed-0 canary failed after its first
50,000-step recovery checkpoint. Vertex logs identify an infrastructure failure
in Vertex Experiments metadata: a transient stale-schema 409 followed by an
existing-context 409. It is not classified as algorithmic divergence.

## Capacity and staging

The `us-central1` quota snapshot records 42 N1/E2 training vCPUs and 200
concurrent custom-training pipelines. Each `n1-standard-4` run requests four
vCPUs, so no more than ten training runs can be active. This quota cannot support
the planned 16-job step or the 32-job target. Keep the operational cap at ten
until an approved CPU quota increase is visible.

The first launch action is an eight-run canary. It covers all four reward
conditions with seed 0 and GAE horizons 3 and full. After all eight runs have
uploaded `_SUCCESS` and passed `scripts/validate_confirmatory.py`, submit the
remaining matrix with `--max-concurrency 10`. Re-run the launcher to fill open
slots as jobs finish. The launcher skips completed or already-submitted runs and
refuses the remaining batch until every canary has succeeded.

The canary command is intentionally recorded without `--submit`, so it performs
a read-only readiness check:

```bash
uv run python scripts/submit_confirmatory.py \
  --service-account gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com \
  --batch canary --max-concurrency 8 --offline
```

Offline mode validates the eight job specifications without contacting GCP and
cannot be combined with `--submit`. Before launch, remove `--offline` to refresh
remote state; when launch is authorized, also add `--submit`. Do not edit the configuration,
manifest, tag, image URI, or preflight report between the dry run and submission.

## Failed-run recovery

A terminal failed run with a published recovery checkpoint can continue without
duplicating environment steps. Supply its exact frozen run ID:

```bash
uv run python scripts/submit_confirmatory.py \
  --service-account gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com \
  --max-concurrency 10 \
  --resume-run confirmatory-ppo-delayed8-h003-seed0-9213ff920697
```

This first invocation is read-only. Add `--submit` after it reports
`ready-to-resume`. Resume requires a terminal failed, cancelled, or expired
Vertex job, the prior submission receipt, and `_LATEST_CHECKPOINT.json`. It
restores model and optimizer state, reward buffers, all RNG streams, and prior
metric rows. The failed canary would continue from 50,000 steps.

To discard a partial attempt and restart at step zero, first inspect the guarded
reset plan:

```bash
uv run python scripts/reset_confirmatory_run.py \
  --run-id confirmatory-ppo-delayed8-h003-seed0-9213ff920697
```

The command refuses completed runs, nonterminal jobs, unknown manifest rows, and
identity mismatches. Its dry run currently reports five partial GCS objects and
one existing Vertex experiment run. To execute the reset, repeat the exact ID as
an explicit confirmation:

```bash
uv run python scripts/reset_confirmatory_run.py \
  --run-id confirmatory-ppo-delayed8-h003-seed0-9213ff920697 \
  --execute \
  --confirm-run-id confirmatory-ppo-delayed8-h003-seed0-9213ff920697
```

This deletes the failed run's Vertex experiment/TensorBoard run and only the GCS
objects below its exact frozen prefix. Then rerun the canary launcher with
`--submit`; it skips the seven completed canaries and creates only the reset run.
Deleting GCS alone is insufficient because the replacement would collide with
the existing Vertex experiment context.

## Remaining 152 runs

Do not release the remaining matrix until all eight canaries have `_SUCCESS` and
the downloaded canary artifacts pass the validator. Then invoke:

```bash
uv run python scripts/submit_confirmatory.py \
  --service-account gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com \
  --batch remaining --max-concurrency 10 --submit
```

The launcher counts queued, pending, and running jobs toward the ten-job cap. It
skips `_SUCCESS` runs and existing submissions, so rerunning the same command is
the queue-refill mechanism. Run it whenever jobs finish until all 152 are
submitted. The regional 42-vCPU quota limits `n1-standard-4` training to ten
active jobs; increase `--max-concurrency` only after the corresponding quota has
actually increased.

For each refill cycle, run `scripts/validate_confirmatory.py --fetch-gcs` first.
It downloads and validates new `_SUCCESS` runs while retaining existing local
downloads. Inspect terminal jobs without `_SUCCESS`. Resume infrastructure
failures with `--resume-run`; use the guarded reset only when deliberately
discarding a partial attempt. Keep numerical divergence as a completed
experimental result when its artifacts satisfy the contract. Do not replace or
exclude a run merely because its learning curve is poor.

## Integrity checks after jobs exist

Download all completed runs and apply the frozen artifact contract without
computing treatment comparisons:

```bash
uv run python scripts/validate_confirmatory.py \
  --fetch-gcs --project gae-experiment-507805
```

Each valid run must reach 1,000,000 environment steps and match the frozen
commit, image digest, configuration hash, environment version, project, and
region. It must contain finite metric tables, checkpoints every 50,000 steps,
and 20 complete diagnostic trajectories at every checkpoint. Use
`--require-complete` only when all 160 runs are expected to have finished.

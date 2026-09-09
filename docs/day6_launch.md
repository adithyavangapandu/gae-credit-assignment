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
The preflight report is `READY`. No confirmatory job had been submitted when this
launch boundary was recorded.

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

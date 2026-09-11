# VPG robustness replication

The VPG replication uses six cells: dense, delayed D=32, and sparse rewards,
each crossed with H=3 and full GAE. Every cell uses seeds 0–9, one million
training transitions, evaluation seed 20260905, and the same Pendulum dynamics,
reward definitions, machine type, bucket, checkpoint interval, and diagnostic
trajectory contract as PPO. This gives exactly 60 runs. D=8 and H=1/H=16 are
omitted because this is a focused algorithm-robustness replication.

VPG takes one full-batch on-policy actor step per 1,000-transition rollout. The
critic uses the same value target and ten fitting epochs as the frozen PPO
configuration. `clip_fraction` is recorded as zero because VPG has no clipped
surrogate. All other metric schemas remain identical.

The VPG jobs use `configs/cloud/gcp_vpg.yaml`. They retain the existing project,
region, bucket, TensorBoard, service account, and `n1-standard-4` machine, while
using the dedicated `vpg_replication_v1` GCS prefix and
`gae-pendulum-vpg-v1` Vertex Experiment. This preserves an initial set of six
otherwise-valid canaries whose first image embedded an incorrect Git SHA; those
artifacts are excluded on provenance grounds and are never overwritten.
The validator mirrors corrected runs under `runs/vpg_replication_v1_downloads`,
so previously downloaded excluded canaries cannot be mistaken for replacement
runs with the same experimental run IDs.

## Freeze and generate the matrix

Run tests, commit the VPG implementation, and build from that clean commit:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
VPG_SHA=$(git rev-parse HEAD)
gcloud builds submit --project gae-experiment-507805 --region us-central1 \
  --config cloudbuild.yaml --substitutions="_GIT_SHA=$VPG_SHA" .
VPG_DIGEST=$(gcloud artifacts docker images describe \
  "us-central1-docker.pkg.dev/gae-experiment-507805/gae-training/trainer:$VPG_SHA" \
  --project gae-experiment-507805 --format='value(image_summary.digest)')
VPG_IMAGE="us-central1-docker.pkg.dev/gae-experiment-507805/gae-training/trainer@$VPG_DIGEST"
git tag -a vpg-replication-v1 "$VPG_SHA" -m "Freeze VPG replication v1"
git push origin vpg-replication-v1
uv run python scripts/generate_vpg_matrix.py \
  --git-sha "$VPG_SHA" --image-uri "$VPG_IMAGE"
```

Commit the manifest and checksum, then run the cloud-aware hard gate:

```bash
uv run --extra gcp python scripts/preflight_vpg.py \
  --image-uri "$VPG_IMAGE" \
  --regional-cpu-quota 42 --concurrent-job-quota 200
tail -1 reports/vpg/preflight_report.md
```

The last line must be `READY`. The gate checks the six cells, ten seeds per cell,
60 unique run IDs/config hashes/GCS paths, one-million-step budget, common
evaluation seed, frozen tag and image, unused paths, GCP access, quota, and tests.

## Submit in stages

First validate all six seed-zero job specifications without creating jobs:

```bash
uv run --extra gcp python scripts/submit_vpg.py \
  --service-account gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com \
  --batch canary --max-concurrency 6 --offline
```

Remove `--offline` and add `--submit` to create the six canaries. After all six
have `_SUCCESS` and pass validation, submit the remaining 54 with the current
ten-job regional cap:

```bash
uv run --extra gcp python scripts/submit_vpg.py \
  --service-account gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com \
  --batch remaining --max-concurrency 10 --submit
```

Rerun the same remaining command whenever jobs finish. It skips completed and
already-submitted rows and fills only open slots.

## Validate and retry

Download completed runs and apply the full artifact contract:

```bash
uv run --extra gcp python scripts/validate_vpg.py \
  --fetch-gcs --project gae-experiment-507805
```

The report distinguishes valid, missing, and invalid runs. Once all jobs should
be complete, add `--require-complete`; a successful result must report six cells,
60 valid runs, zero missing, and zero invalid.

For an infrastructure failure with `_LATEST_CHECKPOINT.json`, dry-run and then
submit a checkpoint resume:

```bash
uv run --extra gcp python scripts/submit_vpg.py \
  --service-account gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com \
  --resume-run RUN_ID --max-concurrency 10
# Repeat with --submit after the dry run succeeds.
```

For an early terminal failure that should restart at step zero, inspect the reset
first, then repeat the exact run ID as confirmation:

```bash
uv run --extra gcp python scripts/reset_vpg_run.py --run-id RUN_ID
uv run --extra gcp python scripts/reset_vpg_run.py \
  --run-id RUN_ID --execute --confirm-run-id RUN_ID
```

After reset, rerun the appropriate canary or remaining submission command. Never
reset a completed run or replace a run because it learned poorly; the reset tool
refuses `_SUCCESS` runs and nonterminal jobs.

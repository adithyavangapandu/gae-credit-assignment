# Day 5 excluded pilot workflow

Pilot seeds 100–102 are operational validation data. They must never enter the
confirmatory hypothesis tests, effect estimates, confidence intervals, or model
selection claims.

The deterministic calibration is produced by:

```bash
uv run python scripts/calibrate_sparse_reward.py --episodes 3000
```

The initial 15-degree region exceeded the preregistered 10% discovery ceiling.
The one allowed adjustment tightens the angle bound to 10 degrees and leaves the
velocity bound at 1.0 rad/s. Both calibration tables and summaries are stored in
`data/manifests`; the adjusted result remains readily discoverable and is frozen
for the pilot.

Build and publish the tested image before generating the matrix. Then generate
the immutable 24-row manifest:

```bash
uv run python scripts/generate_run_matrix.py \
  --image-uri REGION-docker.pkg.dev/PROJECT/REPOSITORY/trainer@sha256:DIGEST
```

The matrix contains four rewards by two horizons by three excluded seeds. It is
sorted deterministically, has unique config hashes and run IDs, and records the
exact image and GCS prefix for every task.

The pilot image was built from commit
`ef50d309ff57e990f5402020f99ea414de047292` and pinned at digest
`sha256:3f1d25d4f189263a74deba31406714fab58e031ee1b8616d9992d42c9e7d95da`.

Preflight or submit the first four jobs with:

```bash
uv run --extra gcp python scripts/submit_pilot.py \
  --service-account gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com

uv run --extra gcp python scripts/submit_pilot.py \
  --service-account gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com \
  --submit
```

The gate is dense/H3/seed100, dense/full/seed100, delayed-32/H3/seed100, and
sparse/full/seed100. The controller skips submitted/completed identities,
refuses duplicate hashes, counts active jobs, caps concurrency at eight, and
will not release the remaining queue until all four have `_SUCCESS`.

Fetch and validate all available pilot bundles with:

```bash
uv run --extra gcp python scripts/validate_study.py \
  --project gae-experiment-507805 --fetch-gcs
```

Add `--require-complete` for the final quality gate. Validation checks the full
matrix, exact identities, progress and evaluation cadence, finite values,
positive action standard deviation, sparse discovery, delayed payout timing and
discounted reward conservation, per-seed initialization, 250 diagnostic updates,
and checkpoint artifacts. It writes `analysis/pilot_qc_data.json`, consumed by
`analysis/pilot_quality_control.qmd`.

After the four-run gate passes, preflight and submit the remaining queue with
`--batch remaining`. Repeat validation until all 24 runs pass. Only then render
the QC report, estimate cost from median pilot runtime with a 20% margin, create
the confirmatory matrix, freeze decisions, and tag `experiment-v1`.

## Initial gate submission

The four gate jobs were submitted on September 7, 2026, each as attempt 1:

| Condition | Vertex Custom Job |
|---|---|
| delayed-32 / H=3 / seed 100 | `4340729665732739072` |
| dense / H=3 / seed 100 | `6608292078113783808` |
| dense / full / seed 100 | `4707773035363434496` |
| sparse / full / seed 100 | `574594477344161792` |

All four were pending during the post-submission duplicate-safety check. The
remaining 20 jobs have not been submitted and remain behind the quality gate.

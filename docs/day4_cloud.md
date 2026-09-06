# Day 4: cloud execution and recovery

Day 4 extends the tested local PPO pipeline; it does not launch pilots or final
training. Use `configs/cloud/smoke_dense_h3.yaml`: PPO, dense rewards, H=3,
excluded seed 100, five complete 200-step episodes per rollout, and 5,000 total
training steps. Two optimizer epochs and two fixed evaluation episodes keep the
smoke inexpensive. Evaluate and checkpoint every 1,000 training steps. The
future pilot template checkpoints every 50,000 steps and is not run on Day 4.

## Deployment configuration

`configs/cloud/gcp.yaml` contains non-secret identifiers for the user's project:
project `gae-experiment-507805`, region `us-central1`, Artifact Registry repository
`gae-training`, bucket `gs://gae-experiment-gae-credit-data`, experiment
`gae-pendulum-v1`, and the existing TensorBoard resource. The bucket's actual name
is intentionally used instead of inventing a new project-ID-prefixed bucket.
The CPU machine is `n1-standard-4`, a documented custom-training machine type.
No accelerator or rendering is enabled.

Deployment overrides: `GAE_GCP_PROJECT`, `GAE_GCP_REGION`, `GAE_ARTIFACT_BUCKET`,
`GAE_VERTEX_EXPERIMENT`, and `GAE_VERTEX_TENSORBOARD`. `GAE_RUN_ID` can assert the
generated identity; arbitrary overrides are rejected to prevent mixing runs.
Credentials come from ADC during development and the attached service account
in Vertex. No service-account key or token belongs in YAML, the image, or Git.

Required setup: billing enabled; APIs for Vertex AI, Artifact Registry, Cloud
Storage, Cloud Build, and Compute; a Docker repository; a regional bucket;
a TensorBoard resource; and `gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com`.
The inspected runtime account has Vertex AI User, Artifact Registry Reader on
the image repository, and Storage Object Admin on the experiment bucket. Object
read, write, and delete are used for validation, checkpoint recovery, and leases.
The submitter needs Vertex job creation and permission to act as that account.
The build account needs Artifact Registry Writer and access to Cloud Build logs/source.

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project gae-experiment-507805
uv sync --locked --extra gcp
uv run --extra gcp ruff check .
uv run --extra gcp pytest
```

The `gcp` extra is optional; ordinary local training imports no GCP clients and
requires no authentication. The Parquet schemas and reward/GAE semantics are
unchanged from Days 1–3. Legacy local runs remain valid.

## Build an immutable image

```bash
docker build --build-arg INSTALL_GCP=true -t gae-credit-assignment:day4 .
docker run --rm gae-credit-assignment:day4 --config configs/cloud/smoke_dense_h3.yaml
gcloud builds submit --region us-central1 --config cloudbuild.yaml \
  --substitutions="_GIT_SHA=$(git rev-parse HEAD)" .
```

Submit a clean committed tree. Cloud Build tests the image, builds Linux/amd64
for Vertex, runs a local container smoke, then publishes
`us-central1-docker.pkg.dev/gae-experiment-507805/gae-training/trainer:<git-sha>`.
The image uses the exact uv lockfile, pinned Python/uv versions, and a non-root
runtime account. Source uploads and Docker builds exclude credentials, virtual
environments, generated runs, and `.git`.

Resolve the published tag to an immutable digest:

```bash
gcloud artifacts docker images describe \
  "us-central1-docker.pkg.dev/gae-experiment-507805/gae-training/trainer:$(git rev-parse HEAD)" \
  --format='value(image_summary.digest)'
```

Pass the full `.../trainer@sha256:...` URI below. A mutable tag is rejected.
The digest and source hash are recorded in each cloud manifest and checkpoint.
The build's Git SHA is baked into the image for provenance.

## Submit exactly one fixed job

```bash
uv run --extra gcp python scripts/submit_vertex_job.py \
  --config configs/cloud/smoke_dense_h3.yaml --phase smoke \
  --image-uri 'us-central1-docker.pkg.dev/gae-experiment-507805/gae-training/trainer@sha256:REPLACE' \
  --service-account gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com \
  --dry-run
```

`--dry-run` prints the resolved job request without constructing SDK clients,
reading ADC, or creating resources. Replace the digest and remove `--dry-run`
to submit. The script prints the Vertex job resource name and artifact prefix.
It submits one CPU replica, a bounded timeout, no hyperparameter tuning, and
no blind automatic retries. The resolved experiment config is passed as JSON
in a non-secret environment variable; the image need not contain a custom YAML.

Run IDs use hyphens to work unchanged as Vertex run names and job labels:
`smoke-ppo-dense-h003-seed100-<config-hash>`. The same ID appears in local paths,
Parquet rows, manifests, checkpoints, Vertex metrics, GCS prefixes, and job labels.
GCS layout is
`study_v1/<phase>/algorithm=ppo/reward=<condition>/horizon=<horizon>/seed=<seed>/run_id=<id>/`.
The phases are smoke, pilot, and confirmatory. Pilot seeds are validated as
excluded from final training seeds; no pilot matrix is submitted here.

## Logging and completion

`RunLogger` delegates to `LocalArtifactLogger` and optional `GCSArtifactLogger`
and `VertexMetricsLogger`. Vertex receives compact live loss, entropy, gradient,
KL, clip, advantage, explained-variance, dense-evaluation and success metrics.
Time-series metrics use training environment steps and the configured TensorBoard.
GCS remains the authoritative record.

A completed bundle includes the manifest, resolved config, all three Parquet
tables, diagnostics, checkpoints, summary, `validation_report.json`,
`checksums.json`, and `_SUCCESS`. Training completes, tables are closed and
validated, file SHA-256/size inventory is written, and uploads are read back
before `_SUCCESS` is written last. The marker contains the checksum inventory's
SHA-256. Failed training, logging, or upload attempts do not publish success.
The validator checks integrity in addition to existing schema/count/finite-value checks.

Create-only `_SUBMIT_LOCK.json` prevents duplicate submissions; `_LEASE.json`
allows only one training writer. GCS operations use generation preconditions
and CRC32C checks. `_IDENTITY.json` rejects reuse with a different run/config/code/image.
Completed prefixes cannot be overwritten. Partial uploads can be replaced only
under the exclusive writer lease; no reader should use them without `_SUCCESS`.

## Checkpoint and resume

Checkpoints are atomic and only taken **after complete PPO updates**, including
scheduled evaluation and diagnostics. They contain actor/critic and Adam states,
update/step counters, Python/global NumPy/global PyTorch state, all dedicated RNG
streams, both environment snapshots, both reward-buffer snapshots, accumulated
table rows and diagnostic files, original model hashes, config/hash/run ID,
source/image digests, runtime fingerprint, and checkpoint schema version.
This lets a fresh worker reconstruct the run without duplicate rows. Recovery
never silently switches configurations or images and never resumes mid-episode.

Local recovery exercise (the first command intentionally exits unsuccessfully):

```bash
uv run python -m gae_credit.train --config configs/cloud/smoke_dense_h3.yaml --stop-after-update 2
uv run python -m gae_credit.train --config configs/cloud/smoke_dense_h3.yaml \
  --resume runs/RUN_ID/checkpoints/update-000002.pt
```

Cloud checkpoints are uploaded under `recovery/` with SHA-256 names, then a
checksummed `_LATEST_CHECKPOINT.json` pointer is updated. Re-submit the same
configuration, phase, and immutable image with `--resume` to download that
checkpoint. The submission script verifies the prior job is FAILED/CANCELLED/
EXPIRED before reclaiming its stale worker lease. A completed or still-running
job is not eligible. A submission whose API response was lost leaves a submit
lock intentionally: inspect Vertex jobs and repair `_SUBMISSION.json` before
removing that lock. Do not blindly retry and create duplicate billed jobs.

Tests compare resumed versus uninterrupted tables (except elapsed time),
diagnostics, and final model hashes. They also exercise fresh-worker recovery,
failed uploads, collision protection, and every identity/version mismatch.
Local and cloud platform results need not be bitwise equal; resuming itself
requires the same code, image, runtime fingerprint, config, and RNG state.

## Download, validate, and estimate runtime/cost

```bash
uv run --extra gcp python scripts/download_cloud_run.py \
  --project gae-experiment-507805 --prefix 'gs://BUCKET/PREFIX/run_id=RUN_ID' \
  --destination runs/downloaded/RUN_ID
uv run python scripts/validate_run.py runs/downloaded/RUN_ID
uv run python scripts/estimate_run_cost.py runs/downloaded/RUN_ID \
  --target-env-steps 1000000 --hourly-price HOURLY_COMPUTE_RATE
```

The downloader requires `_SUCCESS`, verifies every checksum, stages downloads,
and runs the same local validator before exposing the destination. Runtime is
linearly extrapolated from the observed cloud run; the price must be supplied
from current pricing. Startup/build, GCS, TensorBoard, retries, and taxes are
separate. Short smoke estimates are approximate and are not performance results.

## Official references

- [Custom Job SDK](https://docs.cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomJob)
- [Vertex experiment runs](https://cloud.google.com/vertex-ai/docs/experiments/create-manage-exp-run)
- [Custom training compute](https://cloud.google.com/vertex-ai/docs/training/configure-compute)
- [GCS generation preconditions](https://cloud.google.com/storage/docs/request-preconditions)
- [Cloud Build container images](https://cloud.google.com/build/docs/building/build-containers)

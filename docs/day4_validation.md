# Day 4 validation record

Source commit: `15700e8158b32cb60c7c7b6e1fa09e3186189105`.
This is infrastructure and pipeline validation, not a learning-performance result.

## Completed prerequisites

- 149 unit/integration tests passed, including interrupted/resumed dense,
  delayed, and sparse training with identical final model hashes and metrics
  excluding elapsed time. Recovery on a fresh local directory and recovery
  after simulated cloud upload failure also passed.
- Ruff lint and formatting checks passed.
- The 5,000-step seed-100 H=3 smoke passed locally and in the non-root,
  cloud-enabled Docker image, with five updates, 25 training episodes, 12
  evaluation episodes, and six evaluation checkpoints.
- [GitHub Actions](https://github.com/adithyavangapandu/gae-credit-assignment/actions/runs/34048432053)
  passed for the source commit.
- No generated runs, environments, checkpoints, or credentials are tracked.

## Verified GCP setup

Project: `gae-experiment-507805` (number `927884708339`), region `us-central1`.
Required APIs are enabled. Runtime service account:
`gae-training-runner@gae-experiment-507805.iam.gserviceaccount.com`.
It has Vertex AI User, repository-scoped Artifact Registry Reader, and
experiment-bucket-scoped Storage Object Admin.

The existing bucket is `gs://gae-experiment-gae-credit-data`; configuration uses
this actual name. TensorBoard is
`projects/927884708339/locations/us-central1/tensorboards/562116523607457792`.
The local ADC quota project was missing and was set to `gae-experiment-507805`.
No service-account key was downloaded.

Cloud Build currently uses the project's default Compute service account,
which has the Editor role. This did not block validation; a dedicated build
account with narrower roles is a future IAM improvement. The runtime account
uses the separately scoped roles described above.

## Image build and publication

[Cloud Build 3bbda530-4fa8-4bdb-8b60-630fa38dfdf4](https://console.cloud.google.com/cloud-build/builds;region=us-central1/3bbda530-4fa8-4bdb-8b60-630fa38dfdf4?project=927884708339)
succeeded, including tests and a container smoke run before publication.

Published image:
`us-central1-docker.pkg.dev/gae-experiment-507805/gae-training/trainer:15700e8158b32cb60c7c7b6e1fa09e3186189105`

Immutable digest:
`sha256:c2ff56a70b1bc8beb0e0fbb497f5062ea007819d366cb4c637a735ddefb99a04`

## Vertex smoke and artifacts

Run ID: `smoke-ppo-dense-h003-seed100-cc025069deb5`.
Configuration SHA-256:
`cc025069deb52eba451ea3128f3d48f1a0e325b3a7d19858236b21197a834c86`.

Job:
`projects/927884708339/locations/us-central1/customJobs/2455577397492187136`.

Artifact prefix:
`gs://gae-experiment-gae-credit-data/study_v1/smoke/algorithm=ppo/reward=dense/horizon=h003/seed=100/run_id=smoke-ppo-dense-h003-seed100-cc025069deb5/`

The job succeeded on September 6, 2026 (America/New_York), with runtime from
`2026-09-07T01:48:14Z` through `2026-09-07T01:48:44Z`. The authoritative GCS
bundle contains `_SUCCESS`, the checksum inventory, all three Parquet tables,
the resolved configuration, manifest, summary, validation report, diagnostics,
and checkpoints.

Check completion in the [Vertex job page](https://console.cloud.google.com/vertex-ai/locations/us-central1/training/2455577397492187136?project=gae-experiment-507805),
or run:

```bash
gcloud ai custom-jobs describe 2455577397492187136 \
  --project gae-experiment-507805 --region us-central1 \
  --format='yaml(state,error)'
```

The bundle was downloaded and validated with:

```bash
uv run --extra gcp python scripts/download_cloud_run.py \
  --project gae-experiment-507805 \
  --prefix 'gs://gae-experiment-gae-credit-data/study_v1/smoke/algorithm=ppo/reward=dense/horizon=h003/seed=100/run_id=smoke-ppo-dense-h003-seed100-cc025069deb5' \
  --destination runs/downloaded/smoke-ppo-dense-h003-seed100-cc025069deb5
```

The result was `"valid": true`, with 5,000 steps, five updates, 37 total
episodes, and six evaluation checkpoints. Vertex experiment metrics include
finite losses, gradient norms, KL, entropy, explained variance, sparse-success
diagnostics, and the final evaluation return. Runtime/cost was estimated with:

```bash
uv run python scripts/estimate_run_cost.py \
  runs/downloaded/smoke-ppo-dense-h003-seed100-cc025069deb5 \
  --target-env-steps 1000000 --hourly-price 0.21849885
```

The measured training interval in the artifact was 12.56 seconds. Linear
extrapolation to one million environment steps was 2,512 seconds (41.9 minutes)
and approximately $0.152 in compute at the documented hourly rate. This remains
a planning estimate because the pilot optimizer and evaluation schedule differ.

Day 4 is complete. During metrics readback, the SDK unexpectedly created a
second default TensorBoard. Resource `3165197108227604480` was identified and
deleted on September 7, 2026; the configured TensorBoard
`562116523607457792` remains.

## Pricing basis

The public [custom-training pricing table](https://cloud.google.com/products/gemini-enterprise-agent-platform/pricing)
lists `n1-standard-4` in Iowa (`us-central1`) at **$0.21849885/hour** without a
savings-plan discount, checked September 6, 2026. Estimates use that rate and
measured runtime; they are not a final invoice. Image build/storage, GCS,
TensorBoard, retries, provisioning overhead, and taxes are separate. The pilot
template also differs from the smoke in evaluation size, PPO epochs, and
checkpoint interval, so a linear smoke extrapolation is only a planning estimate.

Pilot runs and the final configuration freeze remain Day 5 work.

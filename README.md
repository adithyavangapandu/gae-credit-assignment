# GAE credit assignment experiment

This repository contains the source code, frozen configurations, tests, and
non-visual analysis pipeline for a controlled reinforcement-learning study of
generalized advantage estimation (GAE). It intentionally contains **no datasets,
run artifacts, figures, generated reports, credentials, or website source**.
Those products are generated locally or stored in the configured cloud bucket
and are ignored by Git.

The experiment asks whether the useful GAE horizon depends on reward timing.
It compares dense, delayed, and sparse Pendulum rewards across truncated and
full-horizon estimators. The primary confirmatory contrast is the difference in
the full-minus-H=3 evaluation-AUC effect between delayed-32 and dense reward:

```text
(AUC[full, delayed-32] - AUC[H=3, delayed-32])
  - (AUC[full, dense] - AUC[H=3, dense])
```

The main study is a 160-run PPO factorial design (4 reward conditions × 4 GAE
horizons × 10 seeds). A 24-run excluded pilot and a 60-run VPG robustness
replication are also defined. See [the experiment protocol](docs/experiment_protocol.md)
and [mathematical definitions](docs/mathematical_definitions.md) before changing
the frozen study files.

## Repository contents

```text
configs/                 Frozen study, smoke-test, and cloud configuration
src/gae_credit/          Environment, estimators, PPO/VPG, logging, and storage
scripts/                 Matrix generation, launch, recovery, and validation CLIs
analysis/day8/           Non-visual PPO analysis protocol and numbered pipeline
tests/                   Unit, integration, corruption, and reproducibility tests
docs/                    Protocol, schemas, cloud workflow, and decision records
.github/workflows/       CI lint, tests, and end-to-end smoke runs
```

Generated paths such as `data/`, `runs/`, `reports/`, analysis results, and
image files are ignored. `research_site/` is also ignored in full so website
code cannot accidentally enter this repository.

## 1. Clone and install

Prerequisites:

- Git
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker only if you want container or cloud execution
- A Google Cloud project only if you want Vertex AI execution

Python 3.11.16 is selected by `.python-version`. `uv.lock` pins the complete
environment, including CPU PyTorch on Linux.

```bash
git clone https://github.com/adithyavangapandu/gae-credit-assignment.git
cd gae-credit-assignment
uv sync --locked
```

For cloud commands, install the optional Google Cloud dependencies too:

```bash
uv sync --locked --extra gcp
```

No `.env` file or credential is needed for local work. Cloud authentication uses
Google Application Default Credentials; never place a key in this repository.

## 2. Verify the checkout

Run the same static checks and test suite used by CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Then execute both end-to-end smoke conditions:

```bash
uv run python -m gae_credit.train --config configs/smoke_dense_h3.yaml
uv run python -m gae_credit.train --config configs/smoke_dense_full.yaml
for run in runs/ppo__*; do
  uv run python scripts/validate_run.py "$run"
done
```

The smoke configurations perform two updates and are pipeline checks, not
scientific evidence. Each completed run contains its resolved configuration,
manifest, Parquet metrics, diagnostics, checkpoint, and summary under `runs/`.

Useful local-run operations:

```bash
# Inspect a completed run.
uv run python scripts/inspect_run.py runs/RUN_ID

# Write a separate run identity to another output root.
uv run python -m gae_credit.train \
  --config configs/smoke_dense_h3.yaml \
  --output-dir runs/alternate

# Resume a matching interrupted local checkpoint.
uv run python -m gae_credit.train \
  --config configs/smoke_dense_h3.yaml \
  --resume runs/RUN_ID/checkpoints/latest.pt
```

Existing run identities are rejected by default. `--overwrite` replaces only
the exact derived run directory and should be used deliberately.

## 3. Understand the experiment contract

The frozen implementation uses:

- 200-step Pendulum episodes with a four-value observation that includes time
  remaining and zero bootstrap at the true terminal boundary;
- dense reward, discounted-return-preserving delayed reward (blocks 8 and 32),
  and a repeated sparse upright reward;
- GAE horizons 1, 3, 16, and full with gamma 0.995 and lambda 0.95;
- separate 64×64 tanh actor and critic networks, squashed Gaussian actions,
  gradient clipping, and target-KL stopping;
- independent model, environment, action, evaluation, diagnostic, and optimizer
  random-number streams; and
- typed, checksummed artifacts validated against exact schemas.

The authoritative parameter set is
[`configs/confirmatory/study_v1.yaml`](configs/confirmatory/study_v1.yaml). The
[data dictionary](docs/data_dictionary.md) defines every generated artifact.

## 4. Run the excluded pilot

The pilot uses seeds 100–102 and must never enter confirmatory inference. First
run sparse-reward calibration; its generated Parquet and JSON outputs stay in
the ignored `data/` directory:

```bash
uv run python scripts/calibrate_sparse_reward.py \
  --config configs/pilot/study_v1.yaml
```

Cloud execution requires an immutable container digest. Complete the cloud setup
in [the cloud guide](docs/day4_cloud.md), edit the non-secret identifiers in
`configs/cloud/gcp.yaml`, authenticate with ADC, build the image, and capture its
digest as `IMAGE_URI`:

```bash
gcloud auth application-default login
GIT_SHA="$(git rev-parse HEAD)"
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions="_GIT_SHA=${GIT_SHA}"

# Set this to the digest reported by Artifact Registry, never a mutable tag.
IMAGE_URI='us-central1-docker.pkg.dev/PROJECT/REPOSITORY/trainer@sha256:DIGEST'
```

Generate the pilot matrix, review a dry run, and only then submit:

```bash
uv run python scripts/generate_run_matrix.py --image-uri "$IMAGE_URI"

uv run --extra gcp python scripts/submit_pilot.py \
  --service-account SERVICE_ACCOUNT_EMAIL \
  --batch initial

uv run --extra gcp python scripts/submit_pilot.py \
  --service-account SERVICE_ACCOUNT_EMAIL \
  --batch initial \
  --submit
```

Follow the gate, remaining-batch, download, validation, and QC instructions in
[the pilot workflow](docs/day5_pilot.md). All outputs remain ignored.

## 5. Run the confirmatory PPO study

After the pilot gate passes, generate the 160-row manifest from source. Both the
Parquet file and checksum sidecar are reproducible generated data and are not
committed:

```bash
GIT_SHA="$(git rev-parse HEAD)"
uv run python scripts/generate_confirmatory_matrix.py \
  --image-uri "$IMAGE_URI" \
  --git-sha "$GIT_SHA"
```

Run the preflight check with the quotas for your project:

```bash
uv run --extra gcp python scripts/preflight_day6.py \
  --image-uri "$IMAGE_URI" \
  --regional-cpu-quota REGIONAL_CPU_QUOTA \
  --concurrent-job-quota CONCURRENT_JOB_QUOTA
```

Preview the canary jobs, submit them, validate their downloaded artifacts, and
then repeat with `--batch remaining`:

```bash
uv run --extra gcp python scripts/submit_confirmatory.py \
  --service-account SERVICE_ACCOUNT_EMAIL \
  --batch canary

uv run --extra gcp python scripts/submit_confirmatory.py \
  --service-account SERVICE_ACCOUNT_EMAIL \
  --batch canary \
  --submit

uv run --extra gcp python scripts/validate_confirmatory.py \
  --project PROJECT_ID \
  --fetch-gcs \
  --require-complete
```

Use [the launch and recovery guide](docs/day6_launch.md) for queue refill,
checkpoint resume, and guarded reset procedures. Submission commands default to
dry-run; the explicit `--submit` flag is required to create cloud jobs.

## 6. Run the VPG robustness replication

The VPG study has its own config, cloud namespace, generated 60-row manifest,
preflight, submission, and validator:

```bash
GIT_SHA="$(git rev-parse HEAD)"
uv run python scripts/generate_vpg_matrix.py \
  --image-uri "$IMAGE_URI" \
  --git-sha "$GIT_SHA"

uv run --extra gcp python scripts/preflight_vpg.py --image-uri "$IMAGE_URI"

uv run --extra gcp python scripts/submit_vpg.py \
  --service-account SERVICE_ACCOUNT_EMAIL \
  --batch canary

uv run --extra gcp python scripts/submit_vpg.py \
  --service-account SERVICE_ACCOUNT_EMAIL \
  --batch canary \
  --submit

uv run --extra gcp python scripts/validate_vpg.py \
  --fetch-gcs \
  --require-complete
```

See [the VPG replication guide](docs/vpg_replication.md) before releasing the
remaining jobs or recovering a failed run.

## 7. Reproduce the non-visual analysis

The numbered Day 8 pipeline discovers cloud artifacts, validates checksums and
provenance, builds core tables, and computes the preregistered statistics. It
does not make figures:

```bash
uv run --extra gcp python analysis/day8/scripts/01_discover_gcs.py
uv run --extra gcp python analysis/day8/scripts/02_sync_validate.py
uv run python analysis/day8/scripts/03_build_core_tables.py
uv run python analysis/day8/scripts/04_compute_run_outcomes.py
uv run python analysis/day8/scripts/05_paired_differences.py
uv run python analysis/day8/scripts/06_reward_horizon_interaction.py
uv run python analysis/day8/scripts/07_credit_mechanism.py
```

Stages 4–7 refuse incomplete confirmatory data unless explicitly placed in
synthetic-development mode. Full definitions and output grains are in the
[analysis guide](analysis/day8/README.md) and
[analysis data dictionary](analysis/day8/DATA_DICTIONARY.md).

## Extending the experiment and adding figures later

Keep frozen files immutable once results have been inspected. To add a study,
copy a versioned YAML file under `configs/`, give it a new `study_id` or study
version, and add a matching matrix generator/validator plus tests. Reuse
`gae_credit.config`, the algorithm modules, typed logging schemas, and integrity
validators instead of bypassing them. This keeps new runs self-describing and
prevents accidental collisions with the original experiment.

Future figure code can be added under a dedicated source directory such as
`analysis/figures/`, with tests alongside it. Read only the validated tables in
`analysis/day8/results/`; never make a figure script depend on a website build.
Write rendered `.png`, `.svg`, or `.pdf` products to an ignored output directory.
The current ignore rules prevent those binaries and the separate
`research_site/` tree from being committed while allowing future plotting code
(`.py`, `.R`, or `.qmd`) to be versioned.

Before committing any extension, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/check_repository_hygiene.py
git status --short
```

Review `git status` to confirm that no run artifacts, data, figures, credentials,
or website files are staged.

## Container-only smoke test

```bash
docker build -t gae-credit-assignment:smoke .
mkdir -p runs/docker
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/runs/docker:/app/runs" \
  gae-credit-assignment:smoke \
  --config configs/smoke_dense_h3.yaml
for run in runs/docker/*; do
  uv run python scripts/validate_run.py "$run"
done
```

The container pins Python and uv, installs from `uv.lock`, and runs as a non-root
user. Git metadata and all generated/local-only directories are excluded from
the build context.

## License

MIT. See [LICENSE](LICENSE).

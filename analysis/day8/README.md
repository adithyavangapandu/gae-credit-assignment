# Day 8 PPO analysis workspace

This directory contains the machine-readable protocol and numbered, non-visual
analysis pipeline. Generated raw data, analysis tables, and results stay under
`analysis/day8/data` and `analysis/day8/results`; Git ignores both because GCS is
the canonical artifact store. The scripts never change cloud objects.
See [the analysis data dictionary](DATA_DICTIONARY.md) for every stored product
and its row grain.

Run the stages from the repository root:

```bash
uv run --extra gcp python analysis/day8/scripts/01_discover_gcs.py
uv run --extra gcp python analysis/day8/scripts/02_sync_validate.py
uv run python analysis/day8/scripts/03_build_core_tables.py
uv run python analysis/day8/scripts/04_compute_run_outcomes.py
uv run python analysis/day8/scripts/05_paired_differences.py
uv run python analysis/day8/scripts/06_reward_horizon_interaction.py
uv run python analysis/day8/scripts/07_credit_mechanism.py
```

The sync defaults to four concurrent runs and twelve object downloads per run.
Use `--run-workers` and `--workers` to reduce local/network concurrency. A
specific run can be checked with one or more `--run-id RUN_ID` arguments.

Stages 1–3 may run repeatedly while jobs finish. Discovery stores the frozen
160-row GCS inventory. Synchronization reads each `_SUCCESS` inventory, downloads
the analysis subset in parallel, and validates checksums, schemas, provenance,
and completion. It stores metric tables, metadata, 1,000 update diagnostics, and
20 checkpoint trajectory files per run. Model and recovery checkpoints remain
in canonical GCS because Day 8 does not load model weights. Core-table
construction adds reward, horizon, and paired-seed identifiers to the
evaluations, updates, and episode tables.

Stages 4–7 refuse to analyze fewer than 160 valid runs. The
`--allow-incomplete` flag exists for synthetic development checks and must not be
used to report confirmatory findings. No script displays or plots treatment
results.

Run outcomes use mean base-dense evaluation return. Evaluation AUC is
trapezoidal and divided by 1,000,000 environment steps. Final metrics average the
last five evaluation checkpoints; final critic metrics average the last 50 PPO
updates. The fixed return threshold is -200. A successful evaluation requires a
stable-upright rate of at least 0.5. Unreached times remain null and are excluded
from paired time comparisons rather than imputed. Pre-reward advantage magnitude
is the mean absolute assigned-horizon advantage over diagnostic transitions that
occur before a later nonzero reward; conditions with no such transitions remain
null.

The paired stage reports full-minus-truncated seed differences, individual seed
values, paired standardized effects, exact sign-flip p-values, percentile
bootstrap intervals, and Holm-adjusted secondary p-values. The interaction stage
stores seed-level difference-in-differences, bootstrap summaries, and a
reward/horizon model with seed fixed effects. The preregistered primary contrast
is D32 `(full-H3)` minus dense `(full-H3)` for evaluation AUC.

The mechanism stage streams the per-update diagnostic episodes rather than
building a large transition file. It stores summaries by run, 50,000-step band,
and reward-distance bin. Critic prediction error is Monte Carlo discounted return
minus the logged value. Meaningful credit means absolute raw advantage at least
0.01. `33+` includes transitions with no later nonzero reward. The stored outputs
contain absolute advantage, advantage sign, full-reference error, actor-gradient
norm, later-return correlation, critic RMSE, and meaningful-signal fraction.

The supplied Day 8 text names a later VPG robustness replication but does not
specify its matrix, seeds, training budget, or optimizer. This workspace covers
the fully specified PPO analysis sections. VPG needs its own frozen design before
any VPG jobs or comparisons are created.

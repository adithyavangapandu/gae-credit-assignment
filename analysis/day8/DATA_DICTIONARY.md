# Day 8 analysis data products

All generated products are Parquet unless shown as JSON. Run identity columns
come from the checksum-frozen confirmatory manifest.

| Product | Grain | Purpose |
|---|---|---|
| `data/gcs_inventory.parquet` | one row per planned run | GCS completion/submission/checkpoint discovery plus frozen condition identity |
| `data/gcs_inventory_summary.json` | one file | count of absent, submitted, and complete prefixes |
| `data/validation_registry.parquet` | one row per synchronized run | local checksum/schema/provenance status |
| `data/raw_runs/RUN_ID/` | original analysis artifacts | metric tables, metadata, update diagnostics, and checkpoint trajectories |
| `data/core/run_registry.parquet` | one row per planned run | valid/missing/invalid state joined to the experimental matrix |
| `data/core/evaluations.parquet` | one row per run and evaluation checkpoint | evaluation metrics with reward/horizon/seed columns |
| `data/core/updates.parquet` | one row per run and PPO update | optimizer, critic, estimator, and stability metrics |
| `data/core/episodes.parquet` | one row per train/evaluation episode | returns, upright behavior, success, and control costs |
| `results/run_outcomes.parquet` | one row per run | preregistered primary, secondary, exploratory, critic, and instability outcomes |
| `results/paired_seed_differences.parquet` | one row per metric/comparison/seed | individual full-minus-truncated paired differences |
| `results/paired_summaries.parquet` | one row per metric/comparison | means, medians, bootstrap intervals, effects, direction counts, and p-values |
| `results/interaction_seed_values.parquet` | one row per interaction/seed | paired difference-in-differences |
| `results/interaction_summaries.parquet` | one row per interaction | bootstrap interaction estimates and primary-contrast flag |
| `results/interaction_fixed_effect_coefficients.parquet` | one row per model term | reward, horizon, interaction, and seed fixed-effect coefficients |
| `results/horizon_response_monotonicity.parquet` | one row per reward/seed | H=1,3,16,full AUC values and monotonicity indicator |
| `results/credit_distance_summaries.parquet` | one row per run/step-band/distance-bin | aggregated credit signal and critic mechanism metrics |
| `results/credit_run_outcomes.parquet` | one row per run | pre-reward advantage magnitude and contributing transition count |

Nullable time-to-event values mean the threshold was not reached. They are
censored observations and are never converted to the training budget. The
`interim_override` field in JSON summaries records any explicit development run
that bypassed the 160-run completeness lock.

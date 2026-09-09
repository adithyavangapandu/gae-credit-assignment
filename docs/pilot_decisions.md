# Pilot decisions and experiment-v1 freeze

Pilot results are excluded from final inference. This record may assess
feasibility, numerical health, runtime, and operational failures; it may not
select a reward/horizon winner.

## Frozen before pilot submission

- Training source commit: `ef50d309ff57e990f5402020f99ea414de047292`.
- Tested container image:
  `us-central1-docker.pkg.dev/gae-experiment-507805/gae-training/trainer@sha256:3f1d25d4f189263a74deba31406714fab58e031ee1b8616d9992d42c9e7d95da`.
  Cloud Build `2d9f7576-28a9-4363-b526-2f1f2aa1e996` passed 153 tests and
  the 5,000-step container smoke before publication.
- Pilot design: PPO, four reward conditions, H=3/full, seeds 100–102, and
  250,000 environment steps per run.
- Global optimizer/evaluation settings are in `configs/pilot/study_v1.yaml`.
- The original sparse region produced 21.03% entry across 3,000 untrained
  stochastic-policy episodes. The single allowed adjustment tightened the angle
  bound from 0.262 radians to 10 degrees (0.1745329252 radians), retaining the
  1.0-rad/s velocity bound. The adjusted entry rate was 16.90%, with nonzero
  discovery and 6.17% reaching ten consecutive upright steps. No further sparse
  threshold adjustment is permitted for study version 1.
- Stable success remains ten consecutive qualifying post-step states.
- Pilot seeds are disjoint from confirmatory training seeds 0–9. Evaluation seed
  20260905 and diagnostic streams remain separate.

## Completed pilot quality gate

All 24 runs reached 250,000 environment steps and passed checksums, schema,
numeric-health, reward-delivery, checkpoint, seed, and GCS validation. There were
zero missing or invalid runs. Median artifact runtime was 302.85 seconds and the
longest was 326.43 seconds. Linear projection is 1,211 seconds per one-million-
step run and 53.84 total compute hours for the 160-run PPO sweep before margin.
The rendered QC report is `analysis/pilot_quality_control.html`.

The pilot was used only for feasibility and operations. No reward/horizon result
was used to change a treatment, endpoint, or cell-specific hyperparameter.

## Confirmatory settings

The global pilot optimizer settings, sparse thresholds, dynamics, seed lists,
reward formulas, GAE implementation, evaluation endpoint, metric schemas,
analysis rules, and failure policy are frozen in
`configs/confirmatory/study_v1.yaml` and `docs/experiment_protocol.md`.
Confirmatory runs add 20 raw fixed-seed diagnostic trajectories at each
50,000-step checkpoint; this logging requirement was requested before final
launch and does not affect the training RNG streams. The immutable final image,
manifest checksum, and freeze tag are recorded after the final build.

# Pilot decisions and pending experiment-v1 freeze

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

## Pending the 24-run quality gate

The runtime/cost estimate, failure count, metric endpoints, confirmatory matrix,
final metadata commit, and `experiment-v1` tag are recorded here only after all
pilot runs pass automated validation. Day 6 runs must not launch before that
freeze.

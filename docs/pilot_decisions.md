# Pilot decisions and pending experiment-v1 freeze

Pilot results are excluded from final inference. This record may assess
feasibility, numerical health, runtime, and operational failures; it may not
select a reward/horizon winner.

## Frozen before pilot submission

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

The final source commit, container digest, runtime/cost estimate, failure count,
metric endpoints, confirmatory matrix, and `experiment-v1` tag are recorded here
only after all pilot runs pass automated validation. Day 6 runs must not launch
before that freeze.

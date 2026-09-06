# Days 1–3 validation record

Date: 2026-09-05. Local source revision: `4148fc6493f9fc0c4feb608599b37bd43e1947c7`.
These checks establish pipeline validity, not learning performance.

| Acceptance check | Evidence |
|---|---|
| Repository root | `/Users/adivangapandu/RLProject`; private remote `adithyavangapandu/gae-credit-assignment` created before implementation |
| Typed config and protocol | Resolved YAML/hash round trips; invalid settings rejected; protocol, math and data dictionary present |
| Environment | Seeded reset, action clipping, physical cost calculation, exact termination, observations, snapshots and finite random rollout tested |
| Rewards | Dense, D=1/8/32 delayed, partial-block flushing, discounted conservation, snapshots/reset and sparse streak semantics tested |
| GAE | Hand-calculated H=1/3/full, H>=T, terminal boundaries, Monte Carlo telescoping, event/segment credit and omitted-tail tests |
| Networks/PPO | Corrected log density matches transformed distribution; saturation remains finite; ratio initially one; both actor and critic update |
| Tests | 127 passed locally |
| Lint and format | Ruff check and format check passed |
| Reproducibility | Same seed/config reproduces rollout arrays, metrics excluding elapsed time, diagnostics, summary and model hashes |
| Local H=3 smoke | `runs/ppo__dense__h003__seed-000__6698d718e75e` validated |
| Local full smoke | `runs/ppo__dense__full__seed-000__4e9730699b62` validated |
| Docker | Image built from locked Python 3.11.16 environment; H=3 smoke passed as non-root |
| Artifact parity | Host/container updates, episodes and evaluations have exactly equal Arrow schemas |
| Artifact corruption checks | Invalid hashes/horizons, NaN, duplicate IDs, schema changes and wrong step counts rejected |
| Git exclusions | No generated runs, environments, checkpoints or PDFs tracked; common credential-pattern scan passed |
| Push and GitHub Actions | Implementation pushed to main; [CI run 33997258793](https://github.com/adithyavangapandu/gae-credit-assignment/actions/runs/33997258793) passed every step, including both smoke configs and artifact validation |

Each full smoke has 2,000 training transitions, two updates, ten training
episodes, six evaluation episodes, and three evaluation checkpoints. Local
manifests record the revision above with `git_dirty=false`. Docker output is
under `runs/docker/ppo__dense__h003__seed-000__6698d718e75e`; Git metadata is
unavailable inside the image because the build excludes `.git`.

Local runtime: Python 3.11.16 on macOS arm64, CPU PyTorch, one thread. Container:
Linux arm64, Python 3.11.16, CPU wheels from the same lockfile. Minor numeric
differences across platforms are expected; cross-platform bitwise equality is
not part of the acceptance contract.

Implementation commits follow the original milestone sequence:

1. `b457125` — Add shared finite-horizon Pendulum environment.
2. `d41fd75` — Implement tested reward schedules and finite-horizon GAE.
3. `4148fc6` — Add locally reproducible PPO smoke-training pipeline.

The repository is connected over HTTPS using GitHub CLI authentication. SSH key
registration was not required for the completed push or CI verification.

Google Cloud, pilots, full training, VPG, and performance comparisons remain
deferred as specified by the plan.

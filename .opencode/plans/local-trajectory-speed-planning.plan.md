# Local trajectory and speed planning milestone

## Capability

Add an explicit, deterministic local trajectory and speed-plan interface for
the hybrid stack. The plan will expose a short centerline horizon, curvature,
bounded target speeds, and a conservative fallback trajectory for a later
tracking controller. An isolated experimental controller may consume the plan;
the existing controller defaults and TrackMania action path remain unchanged.

## Baseline and failure signature

- `trackmania_rl/hybrid_state.py` already provides centerline geometry,
  projection, curvature, and lookahead samples.
- `trackmania_rl/hybrid_planner.py` currently exposes predictive risk features,
  but not a trajectory-plan object or a speed profile constrained across the
  horizon.
- `ConservativeCenterlineController` selects one dynamic lookahead target and
  one curvature-derived speed target per step.
- Existing deterministic simulator scenarios complete, but this is only a
  bounded kinematic result. Real A01 evidence remains `0/3` finishes and is
  unrelated to a validated classical controller.

## Hypothesis and smallest experiment

If curvature targets are converted into a horizon plan and passed through
forward acceleration and backward braking constraints, the controller will see
an anticipatory, finite speed profile rather than independent single-step
targets. The smallest test is to compare the current nominal controller with
an isolated plan-consuming controller on the existing synthetic nominal and
off-center scenarios, recording completion, tracking error, speed-target
behavior, and determinism.

## Scope

Files/interfaces in scope:

- `trackmania_rl/hybrid_planner.py`: `LocalTrajectoryPlan`, planner config, and
  `plan_local_trajectory`.
- `trackmania_rl/hybrid_controller.py`: isolated plan-consuming controller and
  configuration; no changes to existing default values or class behavior.
- `tests/test_hybrid_planner.py`: contract, boundary, constraint, and
  determinism tests.
- `tests/test_hybrid_controller_planning.py`: consumer behavior and bounded
  command tests.
- `scripts/evaluate_hybrid_local_planning.py`: baseline/candidate simulator
  report with explicit limitations and promotion status.
- `tests/test_hybrid_local_planning.py`: report integrity tests.
- `eval_results/hybrid_rollouts/local_trajectory_speed_planning.json`: offline
  evidence artifact.
- `HANDOVER_DOC.md` and `SESSION_CONTINUATION.md`: milestone evidence and next
  decision.

## TEST / EVAL definition

Focused tests must prove:

- finite, deterministic, shape-matched horizon arrays;
- nondecreasing horizon distances and end-of-track clipping;
- target speeds are non-negative and bounded by the configured curvature cap;
- forward acceleration and backward braking constraints hold for each adjacent
  horizon sample;
- the fallback trajectory stays centered and is slower or equal to the normal
  plan;
- invalid geometry/configuration inputs fail closed; and
- the experimental controller emits bounded commands without changing the
  default controller behavior.

The offline report must compare the existing nominal controller and the new
isolated candidate on the two established synthetic scenarios. It must record
completion, progress, tracking error, recovery, speed-profile constraints,
determinism, and explicit non-TrackMania limitations.

## Acceptance and regression gates

- All focused tests pass.
- Existing hybrid tests pass without changing their expected default behavior.
- Both synthetic scenarios finish for the candidate.
- No candidate metric is promoted as live TrackMania evidence.
- No default controller setting, protected checkpoint, training branch, or live
  action path is modified.
- Python compilation, focused Ruff, full test suite when practical, artifact
  integrity, and `git diff --check` pass.

## Safety boundaries and authorization

- Work only inside `C:\Code_Stuff\TRACK`.
- Do not install packages, start training, launch TrackMania, load checkpoints,
  commit, push, or use external services.
- Do not alter protected checkpoint directories or the active RL configuration.
- This milestone is offline software-in-the-loop only. A visible TrackMania
  run requires a separate named experiment and explicit user authorization.
- Preserve all existing working-tree changes and do not overwrite prior
  evidence artifacts.

## Expected artifacts and negative result

The primary artifact is
`eval_results/hybrid_rollouts/local_trajectory_speed_planning.json`.
If the candidate fails completion, constraint, or regression gates, retain the
implementation as an isolated diagnostic only, record the failure signature,
and do not integrate it into live control. The next decision will be whether
to revise the planner contract or proceed to a separate tracking/MPC design;
residual RL remains blocked until the nominal planning layer has a credible
offline gate.

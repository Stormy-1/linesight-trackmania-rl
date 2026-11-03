# Recorded-transition dynamics calibration milestone

## Capability

Measure how well the current bounded kinematic assumptions predict one-step
motion in saved privileged telemetry. The experiment will use disjoint rollout
files for calibration and held-out evaluation, report timestamp/position/speed/
heading transition residuals, and produce a diagnostic empirical-dynamics
summary without changing MPC or live defaults.

## Baseline and failure signature

- The current simulator uses `dt_s=0.1`, speed-dependent linear drag,
  acceleration/braking bounds, and a steering-to-yaw-rate bound.
- The local MPC result is positive only on a synthetic centerline and uses
  those assumptions without a recorded-transition calibration.
- Existing rollout CSVs contain timestamp, pose, heading, and speed but no
  action inputs, so throttle/brake response and steering authority cannot be
  identified causally from these files.
- Existing heading/projection calibration is complete; this milestone measures
  transition residuals and timestamp/speed consistency rather than repeating
  that heading-convention study.

## Hypothesis and smallest experiment

The one-step predictor `position_next ≈ position + heading * speed * dt` will
have a stable residual scale across disjoint captured runs, while the no-input
speed model will expose how much acceleration is unmodeled. The smallest
experiment is a deterministic transition summary with fit traces from the
earlier Hockolicious/A01 captures and held-out traces from later A01/A07
captures. It will not fit action-response parameters that the schema cannot
identify.

## Scope

Files/interfaces in scope:

- `trackmania_rl/hybrid_dynamics.py`: validated transition metrics and split
  calibration helpers.
- `scripts/calibrate_hybrid_recorded_dynamics.py`: named fit/held-out report
  using only existing raw `Challenge_*.csv` telemetry.
- `tests/test_hybrid_dynamics.py`: parsing, finite metrics, split integrity,
  determinism, and edge-case tests.
- `eval_results/hybrid_rollouts/recorded_transition_dynamics.json`: diagnostic
  artifact with source paths, metrics, limitations, and promotion status.
- `HANDOVER_DOC.md` and `SESSION_CONTINUATION.md`: evidence and next decision.

## TEST / EVAL definition

The calibration must report, separately for fit and held-out splits:

- sample/transition counts and timestamp interval quantiles;
- displacement-rate versus recorded-speed alignment;
- one-step position residual quantiles for the current kinematic assumption;
- speed-change and drag-only residual quantiles; and
- observed horizontal yaw-rate quantiles.

Tests must prove malformed timestamps and non-finite rows fail closed, source
files cannot appear in both splits, output is deterministic, and all reported
numeric fields are finite. The artifact must state that action-response
parameters are not identifiable from telemetry without actions.

## Acceptance and regression gates

- Fit and held-out splits contain at least one valid transition each and are
  disjoint by source path.
- All metrics are finite and deterministic.
- The report explicitly separates measured transition residuals from any
  proposed simulator values.
- No MPC parameter or controller default is modified based on this artifact.
- Focused tests, compilation, focused Ruff, full tests when practical, artifact
  integrity, and `git diff --check` pass.

## Safety boundaries and authorization

- Work only inside `C:\Code_Stuff\TRACK`.
- Do not install packages, start training, launch TrackMania, load checkpoints,
  commit, push, or use external services.
- Read only existing saved telemetry; do not overwrite prior captures.
- Do not infer causal throttle/brake/steering dynamics from files with no action
  columns. A later action-response study needs a separately authorized capture
  design.

## Expected artifacts and negative result

The primary artifact is
`eval_results/hybrid_rollouts/recorded_transition_dynamics.json`.
If held-out residuals are unstable or the fit/held-out comparison is not
credible, retain the report as a negative calibration result and keep the
synthetic MPC candidate diagnostic-only. The next decision will be whether to
design an explicit, authorized action-transition capture or refine the
controller without claiming calibrated dynamics.

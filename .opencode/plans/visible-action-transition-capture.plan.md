# Visible action-transition capture plan

## Capability

Add an opt-in telemetry format that records privileged pose/speed/heading
together with the discrete action index and its accelerate/brake/left/right
inputs at the same action decision. Use it for a named visible TrackMania
action-transition capture after manual desktop preparation. The capture is for
offline dynamics identification only; it will not start training or promote a
controller.

## Baseline and failure signature

- Existing hybrid CSVs contain pose, heading, speed, and timestamp but no
  causal action columns.
- Recorded-transition calibration therefore cannot identify steering,
  throttle, or brake response and is a negative promotion gate.
- The collector already stores the discrete action index internally and has an
  opt-in privileged telemetry writer, but the writer currently emits only the
  eight-column Phase 1 schema.

## Hypothesis and smallest experiment

If the opt-in writer emits one validated action row aligned with each pose
sample at action-decision time, a short visible capture can support causal
one-step action-response residuals. The smallest experiment is writer/parser
validation plus one named repeated visible capture on A01 with training
disabled and the foundational checkpoint loaded read-only.

## Scope

- `trackmania_rl/hybrid_telemetry.py`: backward-compatible extended writer and
  action-column validation.
- `trackmania_rl/tmi_interaction/game_instance_manager.py`: opt-in action
  capture directory and action-aligned write timing; normal telemetry behavior
  remains unchanged when the new environment variable is absent.
- `HYBRID_ROLLOUT_SCHEMA.md`: extended schema and capture procedure.
- `tests/test_hybrid_telemetry.py`: legacy/extended writer contracts.
- `tests/test_hybrid_action_telemetry.py`: extended parsing and alignment
  validation.
- `trackmania_rl/hybrid_action_telemetry.py`: safe parser and transition-row
  contract for later offline identification.
- `trackmania_rl/hybrid_action_dynamics.py`: action-conditioned observational
  transition summaries for the measured runs; synchronization traces are
  excluded from the causal-style summary.
- `scripts/analyze_hybrid_action_transition_capture.py`: provenance-checked
  analysis of the three benchmark-linked measured CSVs.
- `tests/test_hybrid_action_dynamics.py`: action grouping, transition
  alignment, and finite-metric tests.
- `eval_results/hybrid_rollouts/action_transition_a01_20261103_retry3.json`:
  measured action-transition artifact.
- `eval_results/hybrid_rollouts/` named capture artifacts only after the manual
  run; never overwrite existing captures.

## TEST / EVAL definition

Offline gates before a visible run:

- legacy writer output remains exactly `TELEMETRY_COLUMNS`;
- extended rows contain finite timestamps, pose, speed, action index, and
  boolean input fields;
- action fields are required together and reject invalid/conflicting values;
- the parser rejects non-monotonic timestamps, missing columns, and malformed
  action values; and
- no TrackMania or training process is started by tests.

Capture evaluation after manual authorization will report action/telemetry row
counts, timestamp alignment, action distribution, one-step speed/yaw/lateral
response residuals, and separate per-run artifacts. It will not silently merge
runs or overwrite existing evidence.

## Acceptance and regression gates

- Focused writer/parser tests pass and existing tests remain green.
- The new environment variable is the only path that enables action columns.
- Existing telemetry capture remains backward-compatible when the variable is
  absent.
- The visible capture is a new named repeated experiment with training off,
  no checkpoint writes, and explicit `promotion_allowed=false`.
- Live results are kept separate from simulator/offline reports.

## Safety boundaries and authorization

- User authorization for this named visible capture was received on
  2026-11-03; manual visible-desktop preparation is still required before the
  run.
- Do not launch TrackMania or training from tests or automatically.
- Use the visible normal Windows desktop, close TMLoader before the evaluator,
  load the foundational checkpoint read-only, and preserve each run separately.
- No package installation, external service, commit, push, or protected
  checkpoint modification.

## Expected artifacts and negative result

The implementation artifacts are the writer/parser/schema tests. The later
capture artifact will be named under `eval_results/hybrid_rollouts/` and will
include provenance, limitations, and promotion status. If action alignment or
held-out response residuals fail, retain the capture as negative evidence and
do not promote MPC or residual RL.

# Randomized/action-diverse action-transition capture

## Capability

Collect a new named visible A01 action-transition dataset in which a seeded,
state-independent intervention schedule replaces the nominal action on a
predeclared subset of decisions. Preserve the intervention marker beside the
privileged pose/speed/heading and selected action so an offline response model
can distinguish randomized assignments from same-policy observations.

This is an offline system-identification experiment only. It must not start
training, modify controller defaults, modify the foundational checkpoint, or
promote an MPC/residual-RL candidate.

## Baseline and dominant failure signature

- The completed action-aligned A01 capture has three greedy foundational-
  checkpoint runs with exact action/telemetry alignment but `0/3` finishes.
- A runs-1--2 versus run-3 descriptive holdout preserved response signs for
  `8/8` sufficiently represented actions, but response magnitudes shifted.
- The existing action rows are same-map, same-policy, on-policy observations;
  `action_was_greedy` does not identify an intentional intervention.
- Causal fitting and promotion are therefore false. The completed capture must
  not be repeated for this decision.

## Hypothesis and smallest experiment

If the action assignment is randomized independently of the observed state at
the decision point, and its provenance is recorded, an action-conditioned
one-step response model will have finite parameters and will predict held-out
randomized transitions better than an action-blind baseline. The smallest
useful design is:

1. Extend the opt-in action telemetry contract with an explicit
   `action_is_randomized` boolean (or an equivalent validated intervention
   source field). Keep the legacy eight-column and current action-extended
   formats backward-compatible when the field is not requested.
2. Add a separate capture policy wrapper that keeps the foundational
   checkpoint read-only and uses a local seeded PRNG. At each action decision,
   draw an intervention flag before looking at the selected action. With a
   predeclared intervention probability of `0.25`, choose uniformly from safe
   action indices `0..8` on intervention steps; otherwise use the nominal
   greedy action. The `0..8` set covers forward/coast/brake crossed with
   left/neutral/right and excludes the simultaneous accelerate+brake indices
   `9..11` from the first identification experiment.
3. Capture four measured A01 runs for the fit split and two measured A01 runs
   for the held-out split, each with a distinct recorded seed. Warm-up and
   synchronization rows are excluded. Runs remain separate files and no
   transition may cross a file boundary.
4. Require at least `50` randomized transitions for each of the nine safe
   action indices in each split before fitting or comparing a response model.
   The quota is a gate, not a reason to silently merge runs or manufacture
   samples.

The randomization schedule must be independent of pose, speed, map section,
heading, and model output. The analysis may condition on those measured
variables after assignment, but must not use future telemetry to choose an
action.

## Exact files and interfaces in scope

Preparation/implementation, only after this plan is explicitly approved:

- `trackmania_rl/hybrid_telemetry.py`: optional validated intervention field;
  preserve existing writer contracts and reject partial action metadata.
- `trackmania_rl/hybrid_action_telemetry.py`: parse the intervention field and
  retain backward-compatible parsing for prior named captures.
- `trackmania_rl/tmi_interaction/game_instance_manager.py`: carry the optional
  intervention marker from the policy result to the action-aligned row without
  changing normal training behavior.
- `scripts/capture_hybrid_randomized_action_transitions.py`: visible-run entry
  point using the foundational checkpoint read-only, training disabled, local
  seeded randomization, explicit run naming, and non-overwriting output paths.
- `trackmania_rl/hybrid_action_dynamics.py` or a focused companion module:
  compute randomized-subset transition records and action counts without
  crossing run boundaries.
- `scripts/evaluate_hybrid_randomized_action_response.py`: provenance checks,
  quota checks, fit/holdout split checks, action-conditioned response fitting,
  held-out prediction metrics, and the promotion decision.
- `tests/test_hybrid_telemetry.py` and
  `tests/test_hybrid_action_telemetry.py`: schema and validation contracts.
- `tests/test_hybrid_randomized_action_capture.py`: seeded policy determinism,
  intervention independence from the nominal action, action-set bounds, and
  no-boundary-crossing checks.
- `tests/test_hybrid_randomized_action_response.py`: finite metrics, quota
  gates, disjoint provenance, baseline comparison, and deterministic report
  generation.
- `HYBRID_ROLLOUT_SCHEMA.md`: opt-in randomized-capture schema and procedure.
- `eval_results/hybrid_rollouts/<new_named_capture>/`: separate CSVs, seed/
  configuration provenance, and analysis output after authorization and the
  visible run.
- `HANDOVER_DOC.md` and `SESSION_CONTINUATION.md`: record the decision and
  evidence without changing historical capture artifacts.

The exact policy return contract must be inspected before implementation. If a
fifth return value is used for intervention provenance, all rollout callers
and tests must be updated together; do not overload `action_was_greedy`.

## TEST / EVAL definition

Offline preparation gates, before any visible run:

- legacy telemetry output is byte/schema compatible with its existing
  contract;
- prior action captures still parse without the new optional field;
- new randomized rows require a valid boolean marker and reject malformed or
  partial intervention metadata;
- the seeded wrapper produces the same sequence for the same seed and a
  different sequence for a changed seed;
- intervention decisions are generated without reading telemetry or Q-values;
- selected actions are in `0..8` only when randomized, and the nominal action
  is preserved on non-intervention decisions;
- no test launches TrackMania, loads a checkpoint, or starts training.

Post-capture evaluation gates:

- every measured CSV has exact action/telemetry row alignment and strictly
  increasing timestamps;
- all six measured files and their seeds are preserved, with fit and held-out
  paths disjoint and no synchronization file included;
- each split meets all nine randomized-action quotas;
- randomized transition metrics are finite and include speed, yaw, and lateral
  response summaries;
- the fitted action-response model beats a predeclared action-blind baseline
  on held-out randomized transitions for the primary response metric without
  relying on future rows;
- the report explicitly records `causal_fitting_allowed` separately from
  `promotion_allowed`, with promotion remaining false.

## Acceptance, regression, and negative decision

The capture is usable for a separately reviewed offline identification study
only if provenance, randomization, alignment, quota, finiteness, and held-out
baseline gates all pass. Even then, no fitted value may be applied to MPC or
live control by this plan. A later controller study would require its own plan,
simulator gates, and real-physics validation.

If any gate fails, retain the files as named negative evidence, set causal
fitting and promotion false, and do not repeat the same-policy A01 capture.
The next decision must name a changed source, map, intervention design, or
missing verification gate.

## Safety boundaries and authorization

- Work only in `C:\Code_Stuff\TRACK`; no package installation, external
  service, upload, commit, or push.
- Do not modify protected checkpoints or controller/structured-state defaults.
- Do not launch training or resume the frozen `gen_v3_r1` evidence branch.
- A visible TrackMania run requires a visible normal Windows desktop, manual
  preparation, and explicit user authorization for this named experiment.
- Close TMLoader before the evaluator; load the foundational checkpoint
  read-only; keep training disabled; preserve every run separately.
- Stop and preserve diagnostics on startup, telemetry, or infrastructure
  failure. Do not silently retry into an existing output path.

## Expected artifacts and next decision

Expected artifacts are the approved plan, focused offline tests, a new named
capture directory with per-run CSVs and provenance, and a separate analysis
JSON. The next decision is:

- pass: authorize a separately planned offline model-identification evaluation;
- fail or underpowered: keep causal fitting and promotion false and design a
  changed held-out source or intervention schedule.

No visible run is authorized by writing this plan. Explicit authorization is
still required before the named capture is launched.

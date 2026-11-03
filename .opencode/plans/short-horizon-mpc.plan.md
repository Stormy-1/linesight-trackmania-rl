# Short-horizon MPC action-evaluator milestone

## Capability

Add an isolated model-predictive tracking layer that evaluates a bounded family
of constant action sequences over a short horizon and returns the first action
of the lowest-cost prediction. It will consume the local trajectory plan and
structured state, use the existing deterministic kinematic model as an
explicit approximation, and remain separate from the default controller and
TrackMania action adapter.

## Baseline and failure signature

- The nominal controller completes both existing synthetic scenarios.
- The explicit local-plan controller also completes both, but the latest
  artifact reports higher mean lateral error and slightly longer elapsed time
  than the nominal reference.
- The local plan exposes speed constraints, but the current consumer still
  applies one bounded steering decision without comparing predicted outcomes.
- Real A01 evidence remains `0/3` finishes for the foundational RL policy; no
  live classical-controller evidence exists.

## Hypothesis and smallest experiment

If a bounded action evaluator scores predicted lateral error, heading error,
speed error, and action effort over a short horizon, its receding-horizon first
action will reduce tracking error or preserve completion against the nominal
and local-plan references in the deterministic simulator. The smallest
experiment is a constant-action sequence family with fixed horizon and no
learned dynamics; this isolates action selection from model identification.

## Scope

Files/interfaces in scope:

- `trackmania_rl/hybrid_mpc.py`: validated MPC configuration, bounded
  prediction, cost evaluation, and `ShortHorizonMPCController`.
- `tests/test_hybrid_mpc.py`: validation, deterministic selection, bounded
  commands, and prediction safety tests.
- `scripts/evaluate_hybrid_mpc.py`: nominal/local-plan/MPC comparison report.
- `tests/test_hybrid_mpc_evaluation.py`: report integrity and promotion gates.
- `eval_results/hybrid_rollouts/local_trajectory_mpc.json`: offline artifact.
- `HANDOVER_DOC.md` and `SESSION_CONTINUATION.md`: evidence and next decision.

## TEST / EVAL definition

Focused tests must prove:

- invalid horizon, timestep, candidate, and weight inputs fail closed;
- predicted poses remain finite and speed/action values remain bounded;
- candidate selection is deterministic and returns a finite cost;
- the selected command has bounded steering, throttle, and brake values;
- local-plan speed targets are used during prediction; and
- the default controller configuration and behavior remain unchanged.

The report must compare the existing nominal reference, the isolated local-plan
controller, and the MPC candidate on the established nominal and off-center
synthetic scenarios. It must record completion, progress, tracking error,
recovery, action cost, candidate count, determinism, and explicit simulator
limitations.

## Acceptance and regression gates

- Focused MPC tests and existing hybrid tests pass.
- The candidate completes both synthetic scenarios.
- The report is deterministic and has no non-finite prediction or action values.
- Any tracking improvement is reported as simulator-only; no live promotion is
  inferred from it.
- No default controller setting, checkpoint, training branch, or live action
  path is modified.
- Compilation, focused Ruff, full tests when practical, artifact integrity, and
  `git diff --check` pass.

## Safety boundaries and authorization

- Work only inside `C:\Code_Stuff\TRACK`.
- Do not install packages, start training, launch TrackMania, load checkpoints,
  commit, push, or use external services.
- Do not modify protected checkpoints or active RL configuration.
- This is software-in-the-loop only. Visible validation requires a new named
  experiment and explicit user authorization.
- Preserve prior artifacts and unrelated working-tree changes.

## Expected artifacts and negative result

The primary artifact is
`eval_results/hybrid_rollouts/local_trajectory_mpc.json`.
If the MPC candidate fails completion or does not improve the defined tracking
metric, retain it as an isolated diagnostic and do not tune indefinitely on the
same synthetic split. The next decision will be whether to calibrate an
empirical dynamics model on held-out recorded transitions or redesign the
trajectory/tracking interface before residual RL.

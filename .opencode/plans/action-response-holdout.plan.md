# Action-response held-out consistency milestone

## Capability

Evaluate whether the named action-aligned A01 capture supports a minimal
held-out action-response model. Runs 1–2 are the fit split and run 3 is the
held-out split; transitions never cross file boundaries. This is a consistency
check, not a causal dynamics fit or controller promotion.

## Baseline and failure signature

- The final capture has three measured greedy runs with exact action/telemetry
  row alignment but `0/3` finishes.
- Action-conditioned summaries show forward/brake speed-response separation,
  but those are on-policy observations and may vary with speed, map section,
  and unobserved game state.
- The existing recorded-transition calibration showed that older telemetry
  sources cannot be merged safely with newer sources for dynamics fitting.

## Hypothesis and smallest experiment

If the action rows contain a reusable local response signal, the sign and rough
scale of speed-change response for actions present in both fit and held-out
splits will replicate. Compute per-action transition summaries on runs 1–2 and
run 3 independently, requiring at least 10 transitions per action in each
split for comparison. No parameters are fitted or applied.

## Scope

- `scripts/evaluate_hybrid_action_response_holdout.py`: split/provenance checks,
  per-action fit/held-out summaries, sign agreement, and decision status.
- `tests/test_hybrid_action_response_holdout.py`: no-boundary-crossing,
  disjoint-source, finite-output, and sign-agreement logic.
- `eval_results/hybrid_rollouts/action_response_a01_20261103_holdout.json`:
  diagnostic artifact.
- `HANDOVER_DOC.md` and `SESSION_CONTINUATION.md`: evidence and next decision.

## TEST / EVAL definition

Acceptance is descriptive only:

- fit and held-out paths are disjoint and each has at least one transition;
- per-action summaries use only within-run adjacent rows;
- common actions meet the minimum-count threshold before comparison;
- all metrics are finite and deterministic; and
- the report explicitly states whether response signs replicate and whether
  causal fitting is allowed.

## Gates and safety

- No TrackMania launch, training, checkpoint operation, or controller change.
- Do not call an observational sign match a causal model.
- Keep the existing three CSVs and action-capture artifact unchanged.
- Do not repeat the visible capture for this analysis.

## Expected result

If signs or scales do not replicate, retain the capture as descriptive failure
evidence and require a new randomized/action-diverse capture design before
dynamics fitting. If they do replicate, permit only a separately planned
offline model-identification study; do not promote values to MPC or live code.

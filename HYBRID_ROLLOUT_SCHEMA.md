# Hybrid rollout telemetry schema

This is the offline interchange format for Phase 1 structured-state
validation. It is deliberately independent of TrackMania and the RL learner.

## CSV columns

Each row is one telemetry sample. The header must contain exactly these required
fields (additional columns may be ignored by the parser):

```text
timestamp_s,
position_x_m,position_y_m,position_z_m,
heading_x,heading_y,heading_z,
speed_mps
```

Conventions:

- `timestamp_s` is finite and strictly increasing.
- Position and heading use the same 3D coordinate convention as the saved
  centerline `.npy` files.
- Heading is a non-zero direction vector; only its horizontal components are
  used for heading error.
- `speed_mps` is finite. It may be signed if the source telemetry preserves
  reverse motion, but the current conservative controller treats it as a
  measured speed and clamps only the lookahead speed contribution at zero.

## Offline analysis

Run:

```powershell
& "C:\Users\parik\miniconda3\envs\linesight\python.exe" `
  -m scripts.analyze_hybrid_rollout `
  --centerline maps/A01-Race_0.5m_cl2.npy `
  --rollout path/to/rollout.csv `
  --output eval_results/a01_hybrid_rollout.json
```

The output contains per-sample map-matched state and nominal-controller
commands, plus aggregate progress, horizontal tracking, elevation, speed-error,
and recovery metrics. It also reports separate strict-3D and
elevation-tolerant finish diagnostics, first divergence/recovery events, and
fixed progress-bin section metrics. The analyzer never launches the game,
loads a checkpoint, or trains.

## Opt-in action-extended capture

For a separately authorized visible action-transition experiment, set:

```powershell
$env:LINESIGHT_HYBRID_ACTION_CAPTURE_DIR = "eval_results/hybrid_rollouts"
```

This opt-in variable writes a new CSV with the base columns plus:

```text
action_idx,
action_accelerate,action_brake,action_left,action_right,
action_was_greedy
```

The action row is written at the same action-decision point as the captured
pose. Parse it with `trackmania_rl.hybrid_action_telemetry`; legacy eight-column
files remain valid for the existing analyzer. Each run creates a new file and
must be preserved separately. This format is for offline action-response
identification only and does not enable training or controller promotion.

## Randomized/action-diverse capture

The separately planned randomized capture sets:

```powershell
$env:LINESIGHT_HYBRID_RANDOMIZED_ACTION_CAPTURE_DIR = "eval_results/hybrid_rollouts/<new_named_capture>"
```

It adds one required provenance field to the action schema:

```text
action_is_randomized
```

`true` means the action was assigned by the seeded, state-independent
intervention schedule; `false` means the nominal policy action was retained.
The first design uses intervention probability `0.25` and chooses uniformly
from safe action indices `0..8` when intervening. The simultaneous
accelerate+brake actions `9..11` are excluded from this first identification
capture. The marker must never be inferred from `action_was_greedy`.

The capture entry point requires an explicit `--authorize-named-capture` flag,
uses the foundational checkpoint read-only with training disabled, and refuses
to reuse an existing output directory. It performs one uncaptured warm-up and
then six measured A01 runs: four fit runs and two held-out runs with distinct
seeds. Analyze only complete per-run files with
`scripts/evaluate_hybrid_randomized_action_response.py`; the evaluator keeps
fit and held-out files disjoint, requires per-action randomized-transition
quotas, and always reports `promotion_allowed=false`.

The existing aggregate benchmark CSVs do not satisfy this schema: they contain
episode summaries rather than pose/time-series telemetry and therefore cannot
be used for structured-state trajectory validation.

## Collector capture hook

The live collector now has an opt-in capture hook. It is disabled unless the
environment variable below is set before starting a deliberately planned
visible run:

```powershell
$env:LINESIGHT_HYBRID_TELEMETRY_DIR = "eval_results/hybrid_rollouts"
```

Each rollout writes a new CSV in that directory and records its path in the
rollout statistics. The hook captures privileged simulation position, the car's
world-forward heading axis, and speed at each action-state sample. It does not
change the policy or enable training by itself.

After a capture, analyze a file with the command above. The visible run still
requires the normal project procedure: close TMLoader, use a visible desktop
session, and deliberately choose a new non-training capture experiment. Do not
set this variable for the frozen `gen_v3_r1` evidence run.

## Shadow observer replay

The predictive observer can be evaluated after capture without changing any
action:

```powershell
& "C:\Users\parik\miniconda3\envs\linesight\python.exe" `
  scripts.evaluate_hybrid_shadow_observer.py `
  --output eval_results/hybrid_rollouts/hybrid_shadow_observer.json
```

The observer reports feature-warning intervals with activation/release
hysteresis and contains no steering, throttle, or brake output. Keep the first
real capture in shadow mode; use its event timing and false-positive cost as
evidence before considering any intervention policy.

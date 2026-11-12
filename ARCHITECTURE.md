# Architecture

This document explains how the car drives: what each component does, why it is built that way, and what
evidence backs it. The results are in [RESULTS.md](RESULTS.md).

The stack is a classical **sense → plan → act** loop with a learned **correct** stage on top. It runs once every
50 ms of game time, with the game running at 10× real time. The game gives exact pose through TMInterface,
so localisation is free. The hard parts are the map, the speed plan, and control with digital keys.

```text
sense    game state (pose, velocity, wheel contact) → map matching → virtual lidar
plan     official-replay line → speed profile (learned pro speed ∧ learned grip cap, braking pass, jump physics)
act      pure pursuit → safety layers → sigma-delta key adapter → game
correct  residual TD3 policy: Δsteering, Δtarget speed, added before the safety layers
```

---

## 1. Racing line and map matching

**Racing line.** Every campaign map ships an official replay of the author's run. `scripts/build_official_centerlines.py`
converts each one (100 ms ghost samples) into a line resampled every 0.5 m. Lines exist for 64 of the 65 Nations maps
(E04 has no ghost).

*Why:* the RL era's recorded lines were sometimes hazards themselves. The old A05 line hopped a fence that a
ground-following controller can't hop. On the official line the controller finished A05 (21.58 s) for the
first time in the project (`eval_results/hybrid_live/A05_official_v3b/`).

**Map matching** (`trackmania_rl/hybrid_live.py`) projects the car onto the line every step. The search is
temporal: it looks only within ±200 segments (about ±100 m) of the last match, so the car can't jump onto a
crossing or parallel part of the track. The projection gives progress, lateral error, heading error and
elevation error.

## 2. Block map (environment model)

`trackmania_rl/hybrid_block_map.py` reads the block layout of any campaign map from its `.Challenge.Gbx` file, using
`pygbx`. It covers all 242 block types across the 65 maps. A cell is 32 × 8 × 32 m, and a car at height y drives on
level `floor(y/8) − 1`. Wall-border blocks have a fixed solid edge by rotation (0 → z = 0, 1 → x = 32, 2 → z = 32,
3 → x = 0). It is the map half of mapping and localisation. It was used to diagnose the A05 fence crash and is the
base of the virtual lidar (§8).

## 3. Track speed profile

`trackmania_rl/hybrid_track_profile.py` computes a target speed for every point on the line:

1. **Corner limit.** It uses horizontal curvature smoothed over 15 m. Raw 0.5 m curvature is dominated by vertical
   noise and once pinned the car at 12 m/s. In v4 the limit was a hand rule `70 / (1 + 28·|k|)`. From v5 it is the
   **learned pro speed** (§6), capped from v6 by the **learned grip limit** (§7).
2. **Braking pass.** A backward pass limits every point to what the car can still brake off before the next
   corner, at 15 m/s².
3. **Jump physics.** Over a jump the reference line traces the reference car's flight. A parabola fit to its height
   gives the vertical curvature y″, and the horizontal speed of that flight is `v = sqrt(g / |y″|)`. That speed
   becomes a floor on the approach, and the controller uses full throttle when it is below the floor. The floor is
   capped at the corner limit, so a jump approach never overrides a curve.

**Gravity was measured, not assumed.** Steady free fall in 53 pro ghosts gives **g = 24.87 m/s²**
(2,328 samples, IQR 24.55–25.25; `eval_results/gravity_measurement.json`). The earlier value, 15.4, was a guess.
With the measured value, the detector's jump speed matches the pro's actual speed (median ratio 1.01, against 0.80 before).

## 4. Pure pursuit and the key adapter

`PurePursuitController` (`trackmania_rl/hybrid_controller.py`) steers toward a point on the line ahead of the car.
The lookahead is 10 m plus 0.45 s of travel. The aim is taken from the car's actual position, so a large drift gets
a proportionally large correction. Throttle and brake track the profile's target speed.

**Speed-dependent steering cap.** Steering authority is 1.0 up to 35 m/s and falls linearly to 0.5 at 60 m/s.
Sustained full lock at about 50 m/s started a ground slide on A01.

**Sigma-delta key adapter.** TrackMania takes digital keys only, through 12 combinations of
accelerate / brake / left / right. The adapter accumulates the continuous steering and throttle commands and presses a key
whenever the accumulator crosses ±0.5. Over a few steps, the fraction of time a key is held matches the commanded
magnitude. That is the closest a keyboard gets to analog input.

## 5. Safety layers

These sit in `HybridLivePolicy` and take over the car when the normal controller can't cope:

- **Recovery mode.** It engages beyond 10 m lateral error or 55° heading error. It uses a short lookahead and keeps
  a minimum speed. The first version stopped the car dead with zero throttle.
- **Unstick / reverse.** Below 2 m/s for 0.6 s it holds brake for 1 s, which reverses at standstill. Steering is
  mirrored so the nose swings back toward the line.
- **Slip guard.** When the angle between the nose and the velocity exceeds 0.3 rad (exit at 0.15 rad, above 10 m/s),
  it steers `clip(−0.8 · angle, ±0.5)` instead of following the line.
  *Lesson:* the first version used the opposite sign, based on a one-step key-response statistic, and spun the car
  on A01 and A07. That statistic measures the tyre's transient response, not how a slide develops. The sign was
  measured live and is fixed in a unit test.

**Impact classifier** (`scripts/classify_hybrid_impacts.py`). The game reports no collisions, so contacts are inferred:
- an impact is a speed drop of more than 4 m/s in one 50 ms step;
- a landing is an impact preceded by falling;
- a bump is an impact with no deflection near the line;
- otherwise it's a wall hit;
- a scrape is sustained deceleration under throttle that climbing doesn't explain.

## 6. Learned speed model (supervised deep learning)

*Question:* how fast can each part of a track be driven? A hand rule on curvature can't see crests, jumps or what lies
further ahead.

- **Data.** 139,063 points from 63 official replays (`scripts/build_speed_dataset.py`). The target is the pro's speed
  (ghost Δposition / 0.1 s) mapped onto the line.
- **Features** (`trackmania_rl/speed_features.py`, 18 per point, geometry only):
  - curvature here, maximum ahead in five windows out to 220 m, and maximum behind in two windows;
  - three slopes, vertical curvature, and the sharpest crest within 50 m;
  - four jump features (in a jump, its speed, distance to the next jump, the next jump's speed);
  - distance from start.
- **Model** (`trackmania_rl/speed_model.py`): MLP 18-64-64-1 with standardised inputs and target, Huber loss.
- **Validation.** 5-fold cross-validation with **folds by map**, never by point, on training maps only
  (`eval_results/speed_model/validation_v1.json`):

| Predictor | CV r | CV MAE |
|---|---|---|
| MLP | 0.81 | 11.7 m/s |
| linear, same features | 0.57 | 20.3 m/s |
| hand rule `70/(1+28·abs(k))` | 0.33 | 34.1 m/s |

On its own the model didn't improve live results (v5, RESULTS.md). It predicts pro speed well, but pros
drift through corners at speeds this car and controller can't hold. The bottleneck had moved to the vehicle, which
motivated §7.

## 7. Learned grip limit (system identification)

*Question:* what lateral acceleration can *this* car with *this* controller hold before it runs wide?

- **Data.** Our own live logs on 54 non-test maps: 32,210 on-ground cornering steps (`scripts/fit_grip_model.py`).
  - The demand is `a = v²·|k|`.
  - The label "runs wide" means the outward lateral error grows by more than 1.5 m and ends above 2.5 m within 1 s
    (base rate 5.7%).
- **Model.** Logistic regression with three numbers (`eval_results/grip_model/grip_model_v1.json`):
  `logit P(wide) = −12.0 + 0.132·v²|k| + 0.108·v`.
- **Speed limit.** For a chosen risk p, the corner speed limit is the positive root of
  `0.132·|k|·v² + 0.108·v − 12.0 − logit(p) = 0`. At p = 0.2, the allowed lateral acceleration falls from 56 m/s² at
  30 m/s to 23 m/s² at 70 m/s. The speed term captures the controller's steering cap shrinking at high speed.
- **Validation.** Leave-one-map-out AUC 0.754, against 0.708 for speed alone. Test maps were scored once afterwards,
  with nothing chosen on them: AUC 0.84.
- **Use.** The corner limit = min(learned pro speed, grip limit). The risk level, 0.2, was chosen on dev maps only.

## 8. Virtual lidar (learned perception)

The block map says which block sits in each 32 m cell, but not where inside the cell the solid parts are.
`trackmania_rl/range_sensor.py` learns those shapes:

- For each hazard block type and relative height, every instance is rotated into a canonical frame. Each 1 m bin
  that a pro ghost's centre passed through is marked, using 53 training replays.
- Bins no pro ever entered are treated as solid. Because the data are car centres, the solid region is the
  **configuration-space obstacle**: a range of about 0 means contact.
- Shapes are learned per block type, never per map, so they transfer to unseen maps.

The sensor casts 13 rays from −90° to +90°, out to 60 m.

**Validation** (leave-one-map-out over our live runs, 43 wall hits; `eval_results/range_sensor/validation_v1.json`):
- It reads contact (≤ 2 m) at 56% of wall hits, against 10% of ordinary steps.
- As a standalone warning it is weak (AUC 0.62, against 0.76 for lateral error).

So it is **not** used as a safety rule. It is an **observation for the RL policy**, telling it where the walls are
rather than just how far it is off the line.

## 9. Residual reinforcement learning

*Idea* (residual policy learning; Silver et al. 2018, Johannink et al. 2019): the classical stack keeps driving, and a
small learned policy adds a bounded correction. RL only has to learn what the stack gets wrong, not how to drive.

| Part | Design | Code |
|---|---|---|
| Observation (39-D) | track state + base command (10), lookahead geometry: curvature, slope, crests, profile targets, jumps (15), 13 lidar rays, wheel ground contact. Every value is divided by a fixed physical scale, and there is no map identity. | `residual_observation.py` |
| Action | a ∈ [−1, 1]² → Δsteering = 0.25·a₀ (then clipped to the steering cap), Δtarget speed = 8·a₁ m/s | `residual_action.py` |
| Shield | the residual is off while a safety layer owns the car (recovery, slip guard, reverse) | `residual_action.py` |
| Reward | 0.1 per metre of progress − 0.4 per impact − 0.05 per recovery or reverse step − 0.05·\|a\|²; −10 if stuck | `residual_reward.py` |
| Learner | TD3: twin critics, delayed actor, target smoothing; MLP 2×256; γ 0.99 | `residual_td3.py` |

Design choices that made it work:

- **Zero start.** The actor's output layer starts at zero, so an untrained policy drives exactly like the base stack.
  This was checked live: 760 of 760 steps were identical on A07.
- **Critic burn-in.** Before any exploration, the critics were fitted to the base stack's own runs on the 6 training
  maps (`eval_results/residual_rl/critic_burnin_v2.json`). Q correlated with the true return at r 0.89–1.00, so the
  actor's first updates follow a sensible critic.
- **Anchor strength.** Training uses a TD3+BC anchor: the actor loss is −λ·Q + |π(s)|². In td3_v1 (λ = 2.5/mean|Q|) the
  anchor outweighed the critic's action gradient and the actor stayed at |a| ≈ 0.02. With 10× weaker anchoring
  (alpha 25), the actor moved to |a| 0.07–0.15 and stayed inside the data. With no anchor it saturated at the action
  bounds, where there was no data.
- **Held exploration noise.** White noise redrawn every 50 ms mostly averages out through the key adapter. Holding each
  draw for 10 steps (0.5 s, σ 0.2) lets a correction last long enough to change the outcome.

**Training** (`scripts/train_residual.py`) uses only the 6 rl_train maps. There were two runs of 60 live episodes,
about 1 hour each: td3_v1, then td3_v2 initialised from td3_v1's data, ending with a 155k-transition buffer. The deterministic td3_v2 actor learned a consistent pattern:
- extra heading damping on top of pure pursuit;
- slightly lower targets just before jumps, slightly higher elsewhere.

The final policy is `eval_results/residual_rl/td3_v2/final.pt`.

## 10. Rejected ideas, and why

- **Per-map speed caps ("pace notes").** A 42 m/s cap over A01's crest removed its wall hit, but a hand-placed number
  for one map is overfitting. The option still exists in the runner, but the test-set report refuses any run that uses it.
- **Jump-floor shield for the residual** (the residual may only raise the target before a jump). It fixed one map and
  broke two others: a jump's "required speed" is the pro's speed, not a true minimum. It is kept as an option, off by default.
- **An LLM in the control loop.** An LLM takes 0.3–3 s per call against a 50 ms step, isn't deterministic, and adds no
  driving skill.
- **A transformer policy.** It would be oversized for 39 inputs and 155k transitions. An MLP is enough.

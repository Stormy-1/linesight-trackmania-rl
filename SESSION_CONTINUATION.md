# Project Linesight — Session continuation brief

> Updated: 2026-11-13 (STEP 4 DONE — headline (§84): base + residual RL on the held-out test set 4/10, mean completion 58.1%, recovery 431 (v7 base: 2/10, 46.2%, 747). Best ever on finishes and completion. Next: README/architecture/results, then ask the user about committing, then the website. Commits still PAUSED.)
> Read `CLAUDE.md` first (rules, commands, measured facts). This file holds the
> current state and the exact next step. History: `HANDOVER_DOC.md` §60.

## Goal (user decision, 2026-11-11 late — supersedes "zero wall hits on ~10 maps")
**A classical robotics stack drives; learned models replace its hand-set numbers; residual RL learns
corrections on top — judged on held-out maps it was never tuned on.** Resume project (robotics + AI/ML/DL/RL):
every component must be explainable. Contacts (scrapes, light wall touches, hard landings) are fine in a game;
they're a quality metric, not pass/fail. **No per-map tuning** (no pace notes / speed caps) — that's overfitting.
Then the showcase website (phase 2). Deadline was ~2026-11-15; RL (step 4) may run past it.

## Where we are
The hybrid controller drives TrackMania live. Before this, no agent in the project
had ever finished a map it wasn't trained on (the RL foundational checkpoint: 0 of
about 40 runs).

**All 6 maps now finish.** The current per-map results are in the "Latest" table below
(official-replay lines, v3b code).

Versions: v1 = pure pursuit, track profile, recovery creep and unstick. v2 = v1 plus the
jump-speed lock and the nose heading while rolling backward. v3 = v2 plus the steering cap
and the slip guard (wrong sign). v3b = slip-guard sign fixed. `_official_` = driven on the
official-replay centerline.

Logs: `eval_results/hybrid_live/<run name>/` (per-step `run01.csv` + `summary.json`).
The current runs are `<MAP>_official_v3b` for A01 A03 A05 A07 A11, and `hock_pp_v3b`.

## Latest (2026-11-10): all 5 campaign maps finish on official-replay lines — HANDOVER §64
| Map | Official line, live (v3b code) | Impacts | Recovery steps |
|---|---|---|---|
| A01 | 43.75 s | 2 landings, 1 wall (529 m) | 0 |
| A03 | 26.25 s | 0 | 0 |
| A05 | **21.58 s** (first finish ever) | 0 | 0 |
| A07 | 41.38 s | 1 landing, 1 wall (1261 m) | 0 |
| A11 | 25.88 s | 0 | 0 |
| hock (old line, no official replay) | 103.28 s | 5 walls (lat ±10–13 m) | 110 |

How to drive: `--track A05 --centerline A05-Race_official_0.5m_cl.npy` (the same pattern for
A01 A03 A07 A11). The files are `maps/<MAP>-Race_official_0.5m_cl.npy`, built from
`...\TrackMania Nations Forever\GameData\Tracks\Campaigns\Nations\**\<MAP>.Replay.gbx` with
`gbx_to_raw_pos_list` + `extract_cp_distance_interval`. The latter writes `maps/map.npy`, so rename it.

Code state: `PurePursuitControllerConfig.steering_limit` (1.0 up to 35 m/s → 0.5 at ≥60 m/s), the
slip guard in `HybridLivePolicy` (|fva|>0.3: steering = clip(−0.8·fva, ±0.5)), the block map
`trackmania_rl/hybrid_block_map.py`, and the scripts `classify_hybrid_impacts.py` and
`check_block_map_impacts.py`. The runner takes `--centerline`. Tests: 262 pass.

Environment map (HANDOVER §62): the block layout of all 65 Nations campaign maps is readable;
the official replays load for 64. Wall-border edges by rotation: 0→z=0, 1→x=32, 2→z=32, 3→x=0.

## Campaign result (2026-11-11) — HANDOVER §65: **27 of 63 campaign maps finish**
White 12/15, Green 9/15, Blue 4/15, Red 2/15, Black 0/3. Table: `python scripts/run_campaign_batch.py --report`.
- Tools: `build_official_centerlines.py --install` (64 lines; all 65 maps installed in the game's
  `Official Maps`), runner `--map <stem>` / `--max-race-s`, and `run_campaign_batch.py` (resumable;
  stops on TMI_Disrupt).
- **Launches are unattended:** `trackmania_rl/tmi_interaction/boot_keys.py` presses `~`, Enter,
  Enter (TMI console, profile select, map intro), only into the foreground TrackMania window.
  Opt out with `LINESIGHT_NO_BOOT_KEYS=1`.
- How the failures happen: 22 fell below the line (usually 8+ m/s under target beforehand),
  10 stuck on the line, 2 at the end of the line (A10, B11), 1 off-line, 1 stuck near the end (E02).

## SCOPE (user decision, 2026-11-11): ~10 showcase maps, then a showcase website  — SUPERSEDED by the Goal above (kept for history)
The full campaign is dropped (resume project: depth over breadth). Cite the 27/63 unseen-map result instead.

**Showcase set (proposed, awaiting the user's OK)**, official lines unless noted. Latest live result:

| Map | Type | Result | Landings / walls | Why it's in |
|---|---|---|---|---|
| A01-Race | Race | 43.75 s | 2 / **1** | Story map: the first live crashes |
| A03-Race | Race | 26.25 s | 0 / 0 | Story map |
| A05-Race | Race | 21.58 s | 0 / 0 | Story: the fence, solved by the block map |
| A07-Race | Race | 41.38 s | 1 / **1** | Story: the jump spin, solved by the official line |
| A11-Race | Race | 25.88 s | 0 / 0 | Story map |
| A02-Race | Race | 21.82 s | 0 / 0 | Never seen before, first try |
| A08-Endurance | Endurance | 122.06 s | 0 / 0 | 6.6 km |
| A12-Speed | Speed | 37.92 s | 0 / 0 | Speed type |
| B09-Acrobatic | Acrobatic | 19.18 s | 0 / 0 | Acrobatic type, Green |
| D01-endurance | Endurance | 326.97 s | 0 / 0 | Red difficulty, 18.6 km |
| Hockolicious (old line) | Race | 103.28 s | 0 / **5** | RL story (the agent's 54 s run) |

**Done = every showcase map finishes with zero wall hits** (hard landings are allowed). Remaining:
A01 (1 wall hit, 529 m), A07 (1 wall hit, 1261 m), Hockolicious (5 wall hits at ±10–13 m lateral,
110 recovery steps).

## USER DIRECTION (2026-11-11, late): STOP per-map tuning; rethink the goal first
**Decided:** the user chose the FULL PLAN (steps 1–4 under "Next step"), with few live maps.

The user's points, close to verbatim:
- Don't make runs fragile. Light barrier touches, scrapes and hard landings are fine; it's a game
  with game physics. "Zero contact" was the wrong bar.
- Per-map speed caps ("pace notes") are **overfitting to the map**; they'd fail on unseen maps. Don't use them.
- The main goal: an AI that uses **robotics** for control/driving/mapping and **RL for corrections**.
- Discuss the possibilities and end goal first. Then fix things properly, with iterations, and suggest
  improvements the user can showcase *and explain* (AI/ML, DL, RL, deep RL, robotics, design decisions).
Status: the next-step plan was proposed to the user and is awaiting their choice. The pace-note runs
(A01 cap42, B09 cap42, A02 cap44) are experiments only and are NOT part of the method. B03-Race v4:
37.15 s, B10-Speed v4: 73.29 s (both unclassified yet). B09 is not cleanly drivable by the current tracker.
Fact for the plan: official replays give 100 ms ghost samples plus the pro's control entries for 64 maps
(`map_loader.gbx_to_raw_pos_list`), so speeds and inputs are available as supervised and imitation data.

## v4 (2026-11-11): wall-hit fixes — HANDOVER §66
Showcase set **approved by the user as proposed** (all 11 maps).
- **Jump approach capped at the corner limit** (`hybrid_track_profile.py`). A jump's 120 m approach used to
  lock the target at the jump speed and override curves. A07 1261 m: a 70 m/s approach ran a 50–80 m
  radius curve at 62 m/s. Hockolicious 1591/2441/2812 m: false "jumps" (downhill curves) at 56–63 m/s
  through 35–50 m/s corners. Curve points in an approach stay unlocked, so braking still applies.
- **Full throttle under a jump floor** (`PurePursuitController`; `TrackSpeedProfile.is_jump_locked`).
  Proportional throttle left the car about 3 m/s under the floor.
- **Pace notes** (`TrackProfileConfig.speed_caps`, runner `--speed-cap START:END:MPS`, recorded in
  summary.json). A01 340–520 m is a hill crest over continuous slope road (block map: SlopeEnd →
  SlopeStraight). At the reference's 58.9 m/s the car flies, lands with the nose yawed 0.15 rad, bounces
  and slides into the wall (529/545 m). At 42 m/s there are no impacts at all. The block map can't tell
  this crest from A05's obstacle jump automatically (both show only a few gap cells at takeoff), hence a note.
- **Impact classifier gets a "bump" class**: no deflection (|fva| and heading change < 0.1 rad) and
  |lat| < 3 m. Hockolicious 3456 m is the car crossing a `StadiumPlatformToRoad` lip on the line.
  Old v3b wall hits all stay walls.
- Single-map results: A01 42.35 s, 0 impacts (`A01_official_v4b_cap42`). A07 39.97 s, 0 (`A07_official_v4a`).
  hock 98.98 s, 1 bump, 0 walls (`hock_pp_v4a`).
- **Showcase pass:** `scripts/run_showcase.ps1 -Tag v4` drives all 11 (runs `<stem>_showcase_v4`, `hock_showcase_v4`).
  Tests: 272 pass.

## v4 showcase pass (2026-11-11) — `scripts/run_showcase.ps1 -Tag v4`
Classifier now also reports **scrapes**: sustained deceleration under throttle that climbing doesn't explain,
at |lat| ≥ 3 m, not right after a fall. The 4 m/s-in-50 ms proxy missed these. Grinding along a wall
or platform edge never loses 4 m/s in one step.

| Map | v4 result | Impacts (landing/bump/wall) | Scrapes |
|---|---|---|---|
| A01-Race (pace note 340:520:42) | 42.35 s | 0 | 0 |
| A03-Race | 26.06 s | 0 | 0 |
| A05-Race | 20.26 s | 0 | 0 |
| A07-Race | 39.97 s | 0 | 0 |
| A11-Race | 25.38 s | 0 | 0 |
| A08-Endurance | 122.06 s | 0 | 0 |
| A12-Speed | 37.91 s | 0 | 0 |
| D01-endurance | 326.97 s | 0 | 0 |
| Hockolicious | 98.98 s | 1 bump (3456 m lip), 0 walls | 0 |
| A02-Race | rerun pending (`A02-Race_showcase_v4b`); v3b scraped 234–252 m (lat −5, −11 m/s) | | |
| **B09-Acrobatic** | **96% stuck**; wall at 402 m | 2/0/1 | 347–383, 501–514, 671–705 |

**B09 diagnosis:** at 331–382 m the reference jumps diagonally from platform (11,12) to (12,13) and lands 5 m
inside a tilted `PlatformSlope2Straight`. Our car instead rides that platform's raised edge at z = 416
(the cell boundary; y pinned at the 56 m top edge) and grinds 52 → 27 m/s. **v3b did the same** (−28 m/s)
and only just survived, so B09 was never truly clean. v4's full-throttle floor arrives 2 m/s faster and
tips it into the wall. Being tried: pace note `--speed-cap 300:400:42` (`B09-Acrobatic_v4_cap42`).
If that doesn't work, swap B09 for another Green/acrobatic map (the user said swaps are fine).
Boot-keys fix: the background thread now calls `pythoncom.CoInitialize()` (`_set_window_focus_from_thread`).
Before that, focus always failed in that thread.

## Next step (do this first) — THE FULL PLAN (user-approved 2026-11-11). Steps 1–3 done → see "NEXT after Step 3"
Keep live driving small (each map costs ~5 min). The 64 official replays are **offline** data and cost nothing to use.

**Step 1 — evaluation protocol: DONE (2026-11-12, HANDOVER §67).**
- Split (fixed, never re-draw): `maps/map_split.json` from `scripts/make_map_split.py` (seed 20261112).
  - dev: A01 A03 A05 A07 A11 + A02, B09 (both had pace-note experiments) + Hockolicious.
  - **test (10):** A09 A15 | B04 B07 B13 | C02 C10 C13 | D04 D07 (4 finished in campaign v3b, 6 failed). No Black.
  - rl_train (6): A06 A08 B11 B12 C05 D05. offline_train: the 53 non-test drivable replays (dev + rl_train included).
- Drive: `python scripts/run_test_set.py --tag <tag>` (resumable; runs `<map>_test_<tag>`; identical args for every map).
  Report: `python scripts/report_test_set.py --tag <tag> [--json out.json]`. The report rejects speed-capped runs.
  `scripts/run_showcase.ps1` (with the A01 cap) is deleted. Tests: `test_make_map_split.py`, `test_report_test_set.py`.
- Metrics: finish rate (headline), mean completion %, time ratio vs official replay (finished maps, median),
  contacts landings/bumps/walls/scrapes (quality, not pass/fail), recovery steps.

**Baseline v4 on the test set (`--tag v4`, `eval_results/test_set_v4_baseline.json`): finished 3/10 (30%),
mean completion 57.9%, median time ratio 1.60, contacts 12 landings / 0 bumps / 2 walls / 1 scrape, 529 recovery steps.**
Finishes: A09 41.51 s (1.60), A15 41.39 s (1.63), B07 44.97 s (1.48). Note: C02 finished in v3b (62.66 s) but v4
stalls at 2693/2849 m (94.6%) — a v4 regression or run-to-run luck; look at it only through general fixes, never by tuning C02.

**Step 2 DONE (2026-11-12, HANDOVER §68).** Offline part:
- `trackmania_rl/speed_features.py`: 17 geometry-only features per centerline point (curvature here/ahead in 5 windows/
  behind in 2, slope ×3, vertical curvature + crest ahead, jump in/speed/distance-to-next/next-speed, distance from start).
- `scripts/build_speed_dataset.py` → `eval_results/speed_model/dataset.npz` (139k rows, 63 maps; target = pro speed
  from ghost Δposition/0.1 s mapped by arclength; every 4th 0.5 m point).
- `trackmania_rl/speed_model.py` (MLP 64-64, standardised in/out, Huber) + `scripts/train_speed_model.py`:
  5-fold **by-map** CV on offline_train: **MLP r 0.81 / MAE 11.7 m/s; linear r 0.57 / 20.3; hand rule r 0.33 / 34.1.**
  Test maps scored once (not used for choices): MLP r 0.84 / 10.3 vs hand r 0.21 / 31.1. Model: `speed_model_v1.pt`.
- Finding: pros carry 65–110 m/s median on dev maps (A01 109, A03 94, A09 81); we carry 40–60. The 70 m/s profile cap
  (`TrackProfileConfig.max_speed_mps`) is the main limit, and is a hand-set number (step 3 territory: vehicle limits).
- Integration: `build_track_speed_profile(..., base_speed_mps)` replaces the corner rule; braking pass + jump physics
  stay. Runner: `--speed-model PATH --speed-scale S --top-speed MPS` (global, recorded in summary.json).
  `run_test_set.py --set dev|test --extra "<global args>"` (rejects per-map args). Tests: `test_speed_model.py`.
- Live dev tuning (dev maps only, global knobs only). Dev: v4 6/7 (median ratio 1.35, 2 walls, 134 rec); v5a ×0.85 6/7
  (1.40); v5b ×0.85 top 90 5/7 (worse: controller can't use >70 m/s); **v5c ×1.0 top 70: 7/7** (1.48, 3 walls, 185 rec;
  A01 and B09 now finish; slower on A03/A07/A11 where pro-speed corners push the car wide into recovery).
- **Test set v5 = v5c (`eval_results/test_set_v5_speed_model.json`): 3/10, mean completion 52.3%, median ratio 1.63,
  14 landings / 11 walls / 4 scrapes, 734 recovery steps** (v4: 3/10, 57.9%, 1.60, 12/2/1, 529). C02 now finishes
  (63.63 s), A09 now fails (42.7%, wall). **No net gain:** the model predicts pro speed well, but pros drift through
  corners at speeds our car + pure pursuit can't hold. The bottleneck moved to the vehicle/controller → Step 3.
- A01 crest (349–506 m) is a "jump" lock at 58.9 m/s on a convex downhill road; the car drifts sideways airborne.
  The model agrees with the pro there (75–84 m/s); it's in-air tracking, not speed.

**NEXT: Step 3 (learned vehicle model / grip limit).** Fit from live logs (`eval_results/hybrid_live/*/run01.csv`, now
incl. the v5 runs which probe higher corner speeds): max lateral accel vs speed (friction circle), steering → yaw-rate
response, slip onset. Use it as a physics cap on the learned speed: target = min(learned pro speed, sqrt(a_lat_max / |k|)),
plus braking from the fitted decel. Tune nothing per map; dev maps for any global choice; test set once as `--tag v6`.

**Step 3 DONE (2026-11-12, HANDOVER §69): learned grip limit.**
- `trackmania_rl/grip_model.py` + `scripts/fit_grip_model.py` → `eval_results/grip_model/grip_model_v1.json`.
  Logistic regression on our own live logs (54 non-test maps, 32k on-ground cornering steps):
  `logit P(run wide within 1 s) = −12.0 + 0.132·v²|k| + 0.108·v`. Leave-one-map-out AUC 0.754 (demand-only 0.737,
  speed-only 0.708). At risk 0.2: a_lat_max 56/48/40/32/23 m/s² at 30/40/50/60/70 m/s (car + steering cap).
  Braking decel couldn't be fitted (noisy), so 15 m/s² stays.
- Profile: `build_track_speed_profile(..., grip_limit=...)`: corner limit = min(learned pro speed, grip limit).
  Runner: `--grip-model eval_results/grip_model/grip_model_v1.json --grip-risk 0.2` (global, logged).
- Dev: v6a risk 0.2 **7/7, ratio 1.33, 1 wall, 1 recovery step** (v5c 7/7, 1.48, 3 walls, 185). v6b risk 0.3 failed A01 → stopped.
- **Test set v6 (`eval_results/test_set_v6_grip_model.json`): 2/10, mean completion 54.0%, median ratio 1.50,
  19 landings / 0 bumps / 2 walls / 0 scrapes, 677 recovery steps** (v4 3/10, 1.60, 2 walls; v5 3/10, 1.63, 11 walls).
  A15 40.11 s and B07 42.93 s are the best times yet. Lost: A09 (66%, airborne over a drop at 1250 m with steering
  pinned at the 0.5 cap) and C02 (94.8%, a hard landing at 2635 m that all versions hit; v5 got lucky after it).
- Full live command (v6 settings): `--speed-model eval_results/speed_model/speed_model_v1.pt --top-speed 70
  --grip-model eval_results/grip_model/grip_model_v1.json --grip-risk 0.2`.

**Gravity fix DONE (2026-11-12, HANDOVER §70).** Ghost free fall gives **g = 24.9 m/s²** (2,328 samples, 53 maps; was 15.4).
With 24.9 the jump detector's speed matches the pro's (ratio 1.01 vs 0.80). Now `TrackProfileConfig.gravity_mps2 = 24.9` and the
impact classifier uses 24.9; `speed_features` pins 15.4 so `speed_model_v1` keeps its training features. Dev v7a 6/7 (ratio 1.29;
B09 platform-edge grind). **Test v7 (`eval_results/test_set_v7_gravity.json`): 2/10, completion 46.2%, ratio 1.44, 1 wall.**
A15 38.20 s and C02 57.60 s are best-ever; B07 lost (2 m off the line after a faster landing, head-on hit at 286 m).
Finished sets churn v4→v7 (each change flips 1–2 maps through landing events) → the bottleneck is tracking precision around
jumps and landings, plus fixed early stalls (B13, C13, D04 < 10%; B04 54%, C10 45%, D07 55% at the same spot every version).
Test-set looks so far: v4, v5, v6, v7. Don't run it again before a substantively new component.

**USER DIRECTION (2026-11-12):** commit PAUSED; takeoff/landing controller fix PAUSED. **No overfitting:** no micro-adjustments,
only general components (sensor, physics rule, learned policy). Verdicts on the user's ideas (HANDOVER §71): (1) virtual lidar → built;
(2) rover-style jump-readiness check (brake, reverse, retry run-up) → only if logs show under-speed takeoffs still fail;
(3) LLM / multi-agent in the control loop → no (latency, non-determinism); present the stack as the sense–evaluate–decide–act–
correct loop instead. A small CNN/GRU encoder for the RL policy is fine; a transformer is oversized.

**Virtual range sensor DONE offline (HANDOVER §71):** `trackmania_rl/range_sensor.py`, `scripts/build_range_sensor.py`,
`eval_results/range_sensor/footprints_v1.npz` + `validation_v1.json`. Obstacle shapes per hazard block type and height are learned
from 53 pro ghosts (unvisited 1 m bins = configuration-space obstacle); rays are cast over them. Leave-one-map-out on 43 wall hits:
it reads contact (≤ 2 m) at 56% of hits vs 10% of ordinary steps; warning AUC 0.62 (lateral error 0.76), but lateral + range 0.755 >
lateral 0.717. → Use it as an **RL observation**, not as a standalone safety rule. **Wired into live logging (HANDOVER §72):** runner `--range-sensor eval_results/range_sensor/footprints_v1.npz` logs `sensor_yaw_rad` + 13 rays
`range_-90`…`range_+90` per step; doesn't change decisions. A07 check: identical to v7a (all 760 steps), live = offline rays for 99.5%.

**NEXT (user's call):** Step 4, residual RL on steering + target speed. Observation: track-relative state, lookahead geometry,
range-sensor rays, airborne flag; small MLP/1D-CNN (GRU if memory is needed). Train on rl_train maps; headline = base vs base+RL on the
test set (test-set looks so far: v4–v7). The sensor wiring is done. Next sub-step: the RL environment/observation builder around
`HybridLivePolicy` (base command + residual), reusing the Linesight collector/learner infrastructure.

**Step 4 setup DONE (2026-11-12, HANDOVER §73): residual RL plumbing, no training yet.**
- `trackmania_rl/residual_observation.py` (39-D obs, fixed physical scales: track state + base command, lookahead geometry,
  13 lidar rays, wheel ground contact), `residual_action.py` (a ∈ [−1,1]² → Δsteer 0.25, Δtarget 8 m/s; clipped to the
  steering cap; shielded = off in recovery/slip/reverse; `ZeroResidualPolicy`), `residual_reward.py` (0.1/m progress − 0.4 impact
  − 0.05 recovery/reverse step − 0.05|a|², −10 stuck; `build_transitions`).
- `PurePursuitController.command(..., target_speed_offset_mps=0.0)`; `HybridLivePolicy(residual_policy=...)` records `step_records`;
  `game_instance_manager` passes `wheel_contact` to `observe_simulation_state`. Runner `--residual zero` writes `runXX_rl.npz`.
- Tests `tests/test_residual_rl.py` (18). Suite **333 pass**.
- Live smoke `A07-Race_dev_v7a_residual0`: 37.98 s, **760/760 steps identical to v7a**; airborne flag fires exactly at the 3 jumps.

**NEXT (Step 4 training), in order:**
1. DONE (HANDOVER §74): `trackmania_rl/residual_td3.py` (TD3, zero-init actor, TD3+BC-style anchor to the base command,
   shield-aware targets via `next_residual_active`, `ReplayBuffer.add_npz`, `ActorResidualPolicy` with exploration noise,
   save/load); runner `--residual PATH.pt`; `tests/test_residual_td3.py` (10). Suite 343 pass.
   Note: `A07-Race_dev_v7a_residual0/run01_rl.npz` predates `next_residual_active` and won't load; re-record if needed.
2. DONE (HANDOVER §75): `run_test_set.py --set rl_train --tag v7zero` → `<map>_rl_train_v7zero`. **Base on rl_train: 2/6 finish**
   (A08 117.30 s, B12 69.84 s); A06 stuck 98.5%, B11 stuck at its line end (100%), C05 9.1%, D05 9.2%. 6,845 transitions load.
   Optional general fix before training: encode the slip angle as sin/cos (it wraps at ±π during spins).
3. DONE (HANDOVER §76): `scripts/burn_in_residual_critic.py --updates 100000 --out eval_results/residual_rl/critic_burnin_v2`
   (target noise 0 = exact evaluation of the base stack). Q vs true return r 0.89–1.00 on the 6 maps; still 2–7 under (slow
   propagation, harmless offset). v1 (TD3 noise 0.2) was 5–17 under. Start training from `critic_burnin_v2.pt`.
4. DONE (HANDOVER §77): `trackmania_rl/residual_training.py` + `scripts/train_residual.py` (dry-checked, not run live yet).
   **Next action:** first live training run, e.g.
   `python scripts/train_residual.py --name td3_v1 --episodes 60` (~1 h; resumable with `--resume`; log `eval_results/residual_rl/td3_v1/train_log.jsonl`).
   Watch: episode return per map vs the §75 base returns, |residual| growing slowly after the 5k-step actor warm-up, no new stalls.
4b. DONE (HANDOVER §78): `td3_v1` live, 60 episodes (~1 h). Outcome spread = exploration noise, no learning trend; the deterministic
   actor stayed at |a| ≈ 0.01–0.02 because the anchor λ = 2.5/|Q| ≈ 0.23 outweighs the critic's action gradient (≈ 0.15–0.19).
   **Next action:** offline, retrain the actor on td3_v1's buffer (`eval_results/residual_rl/td3_v1/episodes/*.npz` + seed runs) with
   the anchor off (pure TD3; the reward's 0.05·|a|² stays) and check |a|, Q(π) − Q(0); then `td3_v2` live from td3_v1's critic + buffer.
   Needs: a `--anchor-alpha`/no-anchor option in TD3Config + train script, and init-from-run (critic + episodes) in train_residual.py.
4c. DONE (HANDOVER §79): offline retrain on td3_v1's buffer (`scripts/retrain_residual_offline.py`). No anchor → saturated, map-inconsistent
   corrections (offline extrapolation); **alpha 25 → |a| 0.07–0.15, in-data** (`eval_results/residual_rl/td3_v1_offline_anchor_25.pt`).
   Critic's best gain even unconstrained ≤ 0.1 reward → the exploration signal is weak (white noise redrawn every 50 ms).
   **Next action (td3_v2):** add held exploration noise (one draw per ~10 steps, σ ~0.2) to `ActorResidualPolicy` + a train-script option;
   anchor alpha 25 option in train_residual.py; init from `td3_v1_offline_anchor_25.pt` + td3_v1 episodes as buffer; 60 episodes; then
   deterministic eval on dev maps vs v7a.
4d. DONE (HANDOVER §80): `td3_v2` (60 episodes, ~1 h): A06 2/12 finish (base never), C05 5/9 (base 9%), A08 115.8–121.2 s; deterministic
   actor |a| 0.07–0.15 with a consistent heading-damping pattern. Checkpoint `eval_results/residual_rl/td3_v2/latest.pt`.
   **Next action:** deterministic eval, one run per map, identical args:
   `python scripts/run_test_set.py --set rl_train --tag td3v2 --extra "<v7a args> --range-sensor eval_results/range_sensor/footprints_v1.npz --residual eval_results/residual_rl/td3_v2/latest.pt"`
   vs `<map>_rl_train_v7zero` (§75); then the same with `--set dev` vs v7a.
4e. DONE (HANDOVER §81): deterministic `<map>_rl_train_td3v2`: finishes 2/6 → 3/6 (C05 169.35 s fixed; A08 −0.83 s; B12 −1.84 s),
   but A06 98.5% → 15.8% and B11 99.98% → 54.7% (residual lowers the speed into marginal jumps / acts while airborne).
   **Open decision (ask the user):** (1) shield extension: Δtarget ≥ 0 on jump-locked approaches + no residual while airborne,
   re-eval td3_v2 (no retraining); or (2) more training (td3_v3 `--resume`-style from td3_v2). Then dev maps vs v7a; test set once.
4f. DONE (HANDOVER §82): jump-floor + airborne shield → A06 98.5/15.8/**92.7%** but B12 and C05 lose their finishes; overall 1/6.
   A jump's required speed is the pro's speed, not a minimum, so the rule isn't justified generally. **Shields default OFF** (options kept).
   Lesson: single deterministic runs flip at knife-edge jumps; 6 maps can't separate small policy differences.
   **Next action:** dev-map eval of td3_v2 (unshielded, `td3_v2/final.pt`) vs v7a: `run_test_set.py --set dev --tag td3v2 --extra "<v7a args>
   --range-sensor ... --residual eval_results/residual_rl/td3_v2/final.pt"` (dev v7a runs: `<map>_dev_v7a`). Optionally a robustness
   metric (finish rate over a few slightly perturbed runs per map) before the single test-set look.
4g. DONE (HANDOVER §83): dev eval `<map>_dev_td3v2`: 6/7 (same as v7a), times −0.46/−0.11/−0.77/−0.38 s on A02/A03/A05/A11,
   +0.22/+0.23 s on A01/A07; walls 1 → 0; recovery 97 → 196 (A01, B09). Small, suggestive gain; no finish lost on unseen maps.
   **Open decision (ask the user):** run the held-out test set ONCE now (`run_test_set.py --tag v8_rl --extra "<v7a args> --range-sensor
   ... --residual eval_results/residual_rl/td3_v2/final.pt"`, vs v7 2/10), or first train more (td3_v3) / add a robustness metric.
4h. DONE — **HEADLINE (HANDOVER §84):** test set `v8_rl` (td3_v2/final.pt): **4/10 finish** (A09 39.56 s and B07 41.82 s, both best ever,
   + A15 38.39 s, C02 62.37 s), mean completion 58.1%, recovery 431 vs v7 2/10, 46.2%, 747. No v7 finish lost. Report
   `eval_results/test_set_v8_rl_td3v2.json`. Test-set looks: v4–v7, v8_rl — don't look again without a substantively new component.
   **Next (phase 1 wrap-up):** README / architecture / results write-up (resume-ready); ask the user about committing everything
   (all Step 3+ work is uncommitted); then phase 2, the website. Optional later: td3_v3 (more training), robustness metric.
5. Evaluate on dev maps (choices only there), then the test set ONCE as `--tag v8_rl` vs v7. Headline = base vs base+RL.
Rules: no per-map knobs, no micro-adjustments, train only on rl_train, ask before commit/push.

**Step 2 — learned speed model from pro replays (supervised DL / imitation, ~1–1.5 days).**
- Replays give 100 ms ghost samples + the pro's control entries (`trackmania_rl/map_loader.py`
  `gbx_to_raw_pos_list`; `ghost.records`, `ghost.control_entries`, `ghost.sample_period`). Speed = Δposition.
- Features per centerline point: curvature lookahead profile, slope/vertical curvature (crests), jump
  detection, block-map features (hazards/gaps near). Target: pro speed (scaled down by a safety factor).
- Small MLP/1D-CNN; validate on held-out maps (by map, never by point). Replaces `curvature_gain` +
  hand jump rules in `hybrid_track_profile.py`. Would have handled A01's crest (pros slow/float there).

**Step 3 — learned vehicle model (system identification, ~1 day).**
- From existing per-step logs (`eval_results/hybrid_live/*/run01.csv`): grip limit (max lateral accel vs speed),
  steering → yaw-rate response, slip onset. Replace magic constants (steering cap 1.0→0.5, slip thresholds,
  corner limit) with fitted ones. Fixes A02's understeer (234–252 m) generally (friction-circle speed limit).

**Step 4 — residual RL (stretch; likely past 11-15).**
- Base controller stays in charge; policy outputs small Δsteering and Δtarget-speed. Observation: track-relative
  state + lookahead geometry + block-map features (no pixels). Reward: progress − crash/stuck/recovery penalties.
- Reuse the Linesight training infrastructure (collectors/learner, TMI). Train on the RL train maps; headline =
  base vs base+RL on the test set. Story: pure RL memorised one map (54 s Hock, 0/40 unseen) → robotics
  generalised (27/63) → hybrid gets both.

**Then:** README/architecture/results (resume-ready), ask the user before the first commit/push, then phase 2 website.
Known open issues: A01 crest (lands yawed → wall 545 m, step 2 target), A02 understeer scrape (step 3),
B09 platform seams (tracking precision; good RL-residual showcase case).

## PHASE 2 (later, after phase 1 is done): showcase website
A scrolling story: RL era (early crashes → recovery → the 54 s Hockolicious run) → why the switch to
robotics → the stack (map matching, speed profile, pure pursuit, recovery, slip guard, block map /
official lines) → results on the showcase set → the 27/63 unseen-map generalisation result.
- **The user has phone video recordings** of the monitor showing crashes, finishes, spins and full races.
  Ask where they are when phase 2 starts.
- Data: per-step logs in `eval_results/hybrid_live/` (trajectories can be animated); history in
  `HANDOVER_DOC.md` §60–65.

## Session switching
The user wants a session switch whenever one gets long (it saves tokens), with no lost context.
Before switching: update this file (state + next step) and `HANDOVER_DOC.md` for any results.
The new session starts from `CLAUDE.md` + this file.

## Dropped: the full campaign (cite 27/63 as the generalisation result)
27 of 63 campaign maps already finish (HANDOVER §65). The tooling is ready: `build_official_centerlines.py`,
`run_campaign_batch.py`, and unattended boot keys. Known failure modes: under-speed falls (22),
stuck on the line (10), end of the line (A10, B11), plus free-space carving and an edge-avoiding planner.

## Verification state
- Full pytest: 358 passed (2026-11-12, incl. residual-RL 18 + TD3 11 + training 8 tests).
- Nothing is committed. All work is uncommitted in `TRACK_Copy`. New 2026-11-12: `scripts/make_map_split.py`, `run_test_set.py`, `report_test_set.py`, `maps/map_split.json` (run_showcase.ps1 deleted).

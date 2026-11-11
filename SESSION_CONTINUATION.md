# Project Linesight — Session continuation brief

> Updated: 2026-11-12 (Step 1 done: map split, test-set runner, metrics, v4 baseline on the test set).
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

## Next step (do this first) — THE FULL PLAN (user-approved 2026-11-11). Step 1 done → do Step 2
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

**NEXT: Step 2 (learned speed model from pro replays).** Train on `offline_train` only; validate by held-out *map*
(e.g. leave dev maps out); then drive the test set once (`--tag v5`) and compare with the v4 baseline table.

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
- Full pytest: 285 passed (2026-11-12, incl. the split and report tests).
- Nothing is committed. All work is uncommitted in `TRACK_Copy`. New 2026-11-12: `scripts/make_map_split.py`, `run_test_set.py`, `report_test_set.py`, `maps/map_split.json` (run_showcase.ps1 deleted).

# Project Linesight (TrackMania hybrid self-driving) — read first

**Goal (user decision, 2026-11-11 late):** a classical robotics stack drives (map matching, track
speed profile, pure pursuit, recovery/unstick, block-map environment model). Learned models replace
its hand-set numbers: a speed model learned from pro replays and a vehicle model fitted from logs.
**Residual RL** adds corrections on top. Everything is judged on a **held-out test set of maps** never tuned on.
It's a resume project (robotics + AI/ML/DL/RL), so every component must be explainable. Then a showcase website.
- **No per-map tuning** (pace notes / `--speed-cap`): that's overfitting. Fixes must be general rules or learned.
- Contacts (scrapes, light wall touches, hard landings) are acceptable in a game. They're a quality metric, not pass/fail.
- Keep live driving small (~10 test maps, a few dev maps). The 64 official replays are free offline data.
- The world-record chase on Hockolicious was dropped long ago; don't frame work around it.

**Current state and next step:** `SESSION_CONTINUATION.md` (always up to date).
Full history: `HANDOVER_DOC.md` (section 60 onward is the live-driving era).

## Working rules (from the user)
- Work only in `C:\Code_Stuff\TRACK_Copy`. `C:\Code_Stuff\TRACK` is the protected original; don't touch it.
- `WORKSPACE_SAFETY_POLICY.md` and `.opencode/` rules were for an earlier agent. **Ignore them.**
- The user has approved driving TrackMania live from the Terminal panel (`run_in_terminal`). TrackMania needs the visible desktop.
- Prefer live results over more offline toy-simulator studies.
- Update `SESSION_CONTINUATION.md` after every meaningful milestone, so sessions can be switched at any time without losing context.
- Suggest a session switch when a session gets long (it saves tokens). Update `SESSION_CONTINUATION.md` (and `HANDOVER_DOC.md` for results) first, so nothing is lost.
- Finish the project before the showcase website (the user's order).
- Ask before committing or pushing.

## Commands
- Python: `C:\Users\parik\miniconda3\envs\linesight\python.exe` (3.11). Tests: `-m pytest -q -p no:cacheprovider` (285 pass, ~45s).
- Live run, one map:
  `& "C:\Users\parik\miniconda3\envs\linesight\python.exe" scripts\run_hybrid_controller_live.py --track A05 --runs 1 --speed 10 --stuck-cutoff-s 6 --name A05_pp_v3`
  - Tracks: A01 A03 A05 A07 A11 hock (old recorded lines).
  - Any campaign map on its official-replay line: `--map A02-Race` (add `--max-race-s` for long maps).
    Older form: `--track A05 --centerline A05-Race_official_0.5m_cl.npy`.
  - Game launches are unattended: `trackmania_rl/tmi_interaction/boot_keys.py` presses `~`, Enter, Enter.
  - Runs are deterministic, so 1 run per change is enough.
  - The output folder must be new.
  - Chain several maps with `;` in one terminal.
- Held-out test set (`maps/map_split.json`, never re-draw): `scripts/run_test_set.py --tag <tag>`, then `scripts/report_test_set.py --tag <tag>`. No per-map args.
- Analyze a run: per-step CSV in `eval_results/hybrid_live/<name>/run01.csv`, plus `summary.json`.

## Key code
- `trackmania_rl/hybrid_live.py`: `HybridLivePolicy` (game-loop policy, unstick/reverse, impact proxy, CSV log) and `DiscreteActionAdapter` (sigma-delta onto 12 digital actions).
- `trackmania_rl/hybrid_track_profile.py`: smoothed horizontal curvature, braking pass, ballistic jump detection with reference-speed lock.
- `trackmania_rl/hybrid_controller.py`: `PurePursuitController` (the live default; the older controllers are kept as baselines).
- `trackmania_rl/tmi_interaction/game_instance_manager.py`: `observe_simulation_state` hook.

## Measured facts (don't re-derive)
- Positive steering = RIGHT key; it increases the world yaw `atan2(z, x)`.
- Car forward axis is `orientation[2]`.
- Gravity is about 15.4 m/s². Brake held at standstill = reverse.
- There is no collision telemetry: `impact_suspect` = speed drop > 4 m/s in 50 ms. That also catches hard landings.
- Centerlines are recorded reference runs. Over a jump they trace the reference's flight arc, so the car can legitimately run several metres below them.
- Recorded centerlines: A01, A03, A05, A07, A11, Hockolicious and map5.
- Official-replay centerlines: 64 Nations campaign maps, `maps/<stem>_official_0.5m_cl.npy`, from
  `scripts/build_official_centerlines.py`. They're better lines; the old A05 line hopped a fence.
- Block map of any campaign map: `trackmania_rl/hybrid_block_map.py`. A cell is 32×8×32 m, and the car
  drives on level `floor(y/8)-1`.
- Slip guard sign (measured live): steering = −k·forward_velocity_angle. The +k version spun the car.

## Dev tooling installed (project-scoped)
- **codebase-memory-mcp** in `.mcp.json`: a code graph. Re-index with `codebase-memory-mcp cli index_repository --repo-path C:/Code_Stuff/TRACK_Copy --mode moderate`. If a stopped run left stale processes, kill them first.
- **ECC subset** in `.claude/`: agents python-reviewer, pytorch-build-resolver, mle-reviewer, tdd-guide; skills python-testing, pytorch-patterns, mle-workflow, eval-harness, verification-loop; rules in `.claude/rules/ecc/`.
- **ponytail** skills: `/ponytail`, `-audit`, `-debt`, `-review`.
- Rejected: freellmapi (sends code to third parties) and Trail-Code (duplicates the code graph).

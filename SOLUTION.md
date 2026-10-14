# 🛠️ Project Linesight — Comprehensive Audit, Problem Analysis & Solutions

> **Status:** PROPOSED PLAN — AWAITING USER APPROVAL  
> **Rule:** NO CODE OR REPO FILES WILL BE MODIFIED UNTIL THIS SOLUTION DOCUMENT IS APPROVED.

---

## 📑 Table of Contents
1. [Executive Summary & Paradigm Shift](#1-executive-summary--paradigm-shift)
2. [Category A: Claims, Scientific Framing & Presentation](#2-category-a-claims-scientific-framing--presentation)
   - [A.1 World Record Claim vs 240ms Deficit](#a1-world-record-claim-vs-240ms-deficit)
   - [A.2 "No Hardcoded Racing Lines" vs Centerline Inputs & Progress Reward](#a2-no-hardcoded-racing-lines-vs-centerline-inputs--progress-reward)
   - [A.3 "Entropy Collapse" vs Loss of Plasticity & Buffer Domination in DQN](#a3-entropy-collapse-vs-loss-of-plasticity--buffer-domination-in-dqn)
   - [A.4 γ=1.0: Undiscounted Episodic vs Infinite Horizon](#a4-γ10-undiscounted-episodic-vs-infinite-horizon)
   - [A.5 Causal Attribution of the 8.5M Learning Rate Bump](#a5-causal-attribution-of-the-85m-learning-rate-bump)
   - [A.6 Saturating Metrics & The 50ms Action Granularity Hurdle](#a6-saturating-metrics--the-50ms-action-granularity-hurdle)
   - [A.7 Infrastructure Journey: 0–10.6M Local Laptop Grind vs Cloud Migration](#a7-infrastructure-journey-0106m-local-laptop-grind-vs-cloud-migration)
3. [Category B: Missing Artifacts, Verifiability & Baselines](#3-category-b-missing-artifacts-verifiability--baselines)
   - [B.1 Broken Clone: Missing `maps/ESL-Hockolicious_0.5m_cl2.npy`](#b1-broken-clone-missing-mapsesl-hockolicious_05m_cl2npy)
   - [B.2 Empirical Baseline Comparison (Upstream vs Linesight Run)](#b2-empirical-baseline-comparison-upstream-vs-linesight-run)
   - [B.3 Verifiable Replay Proof for the 54.000s Personal Best](#b3-verifiable-replay-proof-for-the-54000s-personal-best)
   - [B.4 Human World Record Reference Citation](#b4-human-world-record-reference-citation)
   - [B.5 Visual Learning Curve & Driving Artifacts](#b5-visual-learning-curve--driving-artifacts)
4. [Category C: Core Engineering Hardening & Bug Fixes](#4-category-c-core-engineering-hardening--bug-fixes)
   - [C.1 Silent Failure Risk: `strict=False` in Weight Loading](#c1-silent-failure-risk-strictfalse-in-weight-loading)
   - [C.2 Self-Healing Auto-Recovery Block Flaws in `game_instance_manager.py`](#c2-self-healing-auto-recovery-block-flaws-in-game_instance_managerpy)
   - [C.3 Hardware Abstraction Centralization (`trackmania_rl/device.py`)](#c3-hardware-abstraction-centralization-trackmania_rldevicepy)
   - [C.4 Configuration Drift & Runtime Reproducibility Snapshotting](#c4-configuration-drift--runtime-reproducibility-snapshotting)
   - [C.5 Speedslide Reward Shaping Decay](#c5-speedslide-reward-shaping-decay)
   - [C.6 Throughput Accounting Reconciliation (80x Nominal vs Measured FPS)](#c6-throughput-accounting-reconciliation-80x-nominal-vs-measured-fps)
5. [Category D: Repository Hygiene, Security & Git Lineage](#5-category-d-repository-hygiene-security--git-lineage)
   - [D.1 "TMNF Credentials" Clarification & Config Templating](#d1-tmnf-credentials-clarification--config-templating)
   - [D.2 Upstream License Integration (MIT)](#d2-upstream-license-integration-mit)
   - [D.3 Upstream Lineage & Visible Diff Specification](#d3-upstream-lineage--visible-diff-specification)
   - [D.4 Line Endings & Mode Noise (`.gitattributes`)](#d4-line-endings--mode-noise-gitattributes)
6. [Step-by-Step Implementation Roadmap](#6-step-by-step-implementation-roadmap)

---

## 1. Executive Summary & Paradigm Shift

The review raises valid, high-impact critiques across all facets of the project:
1. **The claims in `README.md` outran the reality in the logs.** Calling the agent a "World Record Agent" when it is 240ms behind the record, claiming "no racing lines" when feeding 400m of centerline coordinates, and calling a DQN exploration plateau "entropy collapse" damages scientific credibility.
2. **The engineering delta is real, but must be represented with precision.** Getting 10.65M frames of TrackMania simulation to run unattended on a $0.52/hr CPU-only Azure student VM, patching device compatibility, and self-healing socket disconnects is authentic systems engineering. Overselling one-line config edits as complex algorithmic breakthroughs detracts from those real achievements.
3. **The strongest portfolio framing** is:
   > *"Porting and scaling a distributional RL driving system to run autonomously on low-cost headless cloud infrastructure, performing a 10.6M-frame optimization campaign on a competitive tech track, analyzing the structural barriers of the final 240ms (50ms action quantization and exploration plateaus), and architecting a multi-map generalization curriculum."*

Below is the concrete, issue-by-issue plan to fix every problem identified.

---

## 2. Category A: Claims, Scientific Framing & Presentation

### A.1 World Record Claim vs 240ms Deficit
- **The Problem:** The README title says *"TrackMania RL World Record Agent"* and the tagline claims *"Teaching an AI to drive faster than any human"*, yet the benchmark table clearly displays a 240ms deficit (54.000s vs 53.760s).
- **The Solution:**
  1. Update Title in `README.md`:  
     `# 🏎️ Linesight — Autonomous RL Racing Agent in TrackMania Forever`
  2. Update Tagline:  
     `> Closing the gap to the human World Record — 240 milliseconds away on ESL-Hockolicious.`
  3. Frame the status transparently:
     - PB: **54.000s**
     - Purist WR: **53.760s** (by *tastyy*, June 2022)
     - Delta: **+0.240s (+240ms)**.
  4. Explicitly state: The agent has reached high-level competitive human pace and is within ~0.44% of the human benchmark, but the World Record has not yet been beaten.

### A.2 "No Hardcoded Racing Lines" vs Centerline Inputs & Progress Reward
- **The Problem:** The README asserts: *"no hardcoded racing lines, no replay imitation, no human demonstrations"*. However:
  - `config.py` sets `n_zone_centers_in_inputs = 40` with `one_every_n_zone_centers_in_inputs = 20`.
  - At 0.5m resolution, this feeds **40 upcoming 3D coordinates spaced 10 meters apart (a 400-meter forward preview of the track path)** into the neural network's dense layers.
  - The reward function is explicitly computed as progress along this 3D reference centerline (`reward_per_m_advanced_along_centerline = 5 / 500`).
- **The Solution:**
  Replace the overclaim with a technically accurate description:
  > *"The policy learns control through Deep Reinforcement Learning from a multi-modal observation space: downscaled visual frames (160×120 grayscale) and vehicle telemetry (speed, RPM, steer angle, contact materials), augmented by a 400-meter local spatial preview of upcoming track centerline coordinates. The reward signal rewards forward progress along the centerline while penalizing elapsed time. The agent is trained purely via reward maximization—it uses no behavioral cloning, no human replay copying, and no pre-scripted action sequences."*

### A.3 "Entropy Collapse" vs Loss of Plasticity & Buffer Domination in DQN
- **The Problem:** `GENERALIZATION_STRATEGY.md` and `README.md` attribute the stagnation and generalization risk to *"entropy collapse"*.
  - IQN is a value-based distributional Q-learning algorithm acting via $\epsilon$-greedy and Boltzmann exploration. It has no parameterized policy distribution $\pi(a|s)$ whose entropy $\mathcal{H}(\pi)$ can collapse.
  - The actual failure modes for deep Q-networks trained for 10M+ steps on a single environment are:
    1. **Loss of Plasticity / Feature Collapse:** Deep networks under continuous gradient descent lose the ability to learn new features as feature covariance matrices collapse in rank and ReLU units permanently deactivate.
    2. **Severe Overfitting to Map Geometry:** The Q-value distribution fits the exact geometric features and timing of `ESL-Hockolicious`.
    3. **Replay Buffer Domination:** The 200,000-transition replay buffer becomes 100% saturated with transitions from one single track.
- **The Solution:**
  1. Revise `GENERALIZATION_STRATEGY.md` and `README.md`: replace all instances of "entropy collapse" with **"Loss of Plasticity, Feature Overfitting, and Replay Buffer Domination"**.
  2. Clearly state that preventing catastrophic forgetting requires **interleaved multi-map replay buffer composition** and **curriculum sampling**, acknowledging that the multi-map cycling infrastructure exists in upstream Linesight's `map_cycle` engine.

### A.4 γ=1.0: Undiscounted Episodic vs Infinite Horizon
- **The Problem:** The README calls $\gamma=1.0$ *"Infinite horizon"*.
- **The Reality:** In RL theory, an infinite-horizon MDP with $\gamma=1.0$ can have unbounded returns and divergent value functions. Because TrackMania rollouts terminate at the finish line or upon cutoff timeout, the task is strictly **episodic**. With $\gamma=1.0$, it is an **undiscounted episodic MDP**.
- **The Solution:**
  Update hyperparameter table in `README.md`:
  `Discount (γ) | 1.0 | Undiscounted episodic return formulation (stepped from 0.999 to 1.0 at 2.5M frames)`

### A.5 Causal Attribution of the 8.5M Learning Rate Bump
- **The Problem:** `ATTRIBUTION.md` lists Contribution #5 as: *"Added a warmup bump to escape a local optimum"*.
- **The Empirical Evidence:** TensorBoard inspection shows:
  - At **7,072,886 frames**, the PB had already reached **54.010s**.
  - Between 7.07M frames and 10.64M frames (3.57 million frames!), the PB did not improve at all.
  - At **10,643,603 frames**, the PB dropped by 10ms to **54.000s**.
  - The 8.5M LR kick (ramping from $5\times 10^{-5}$ to $1\times 10^{-4}$ and decaying back) did **not** trigger a causal step-change in performance.
- **The Solution:**
  Reframe Contribution #5 in `ATTRIBUTION.md` and `README.md`:
  > *"Exploratory Learning Rate Perturbation at 8.5M Frames: Evaluated a temporary learning rate escalation (from 5e-5 to 1e-4) to test whether learning plasticity could be re-engaged on a plateaued policy. Empirical logs show the policy remained on its 54.01s plateau through this period, demonstrating the resilience—and stubbornness—of late-stage distributional Q-learning networks."*

### A.6 Saturating Metrics & The 50ms Action Granularity Hurdle
- **The Problem:** Claiming "99.6% of human perfection" uses a saturating metric that obscures the fundamental structural difficulty of the final 240ms.
- **The Solution:**
  Add a dedicated technical analysis section in `README.md`:
  - **Action Quantization:** The game engine executes 5 physics steps per action ($5 \times 10\text{ ms} = 50\text{ ms}$ discrete action interval).
  - **The 240ms Gap:** $240\text{ ms} \div 50\text{ ms} = \mathbf{4.8\text{ discrete actions}}$.
  - The remaining gap is not just an exploration problem; it represents roughly 5 discrete decisions over a 1,080-action trajectory. In deterministic TrackMania physics with $\epsilon=0.03$, stumbling onto coordinated multi-step micro-adjustments via uncorrelated exploration is exponentially improbable.
  - This mathematically justifies why shifting focus to **multi-map generalization** is the superior scientific objective.

### A.7 Infrastructure Journey: 0–10.6M Local Laptop Grind vs Cloud Migration
- **The Problem:** The current README presents the project as if training was merely launched on Azure, omitting the core engineering grind: **frames 0 through 10,643,603+ (~148 hours of simulated driving physics) were trained entirely on the user's personal laptop.**
- **The Local Hardware Setup & Constraints:**
  - **Machine:** Consumer Windows Laptop (16 GB RAM).
  - **Resource Saturation:** Running 2 TrackMania collectors at 80× speed + Python neural network forward passes consumed **~13.1 GB of 16 GB RAM** (leaving less than 3 GB for the OS and background apps).
  - **Thermal & Wear Impact:** Near-100% CPU/GPU saturation caused high thermal throttling, constant maximum fan noise, and battery/power wear.
  - **Workstation Lockup:** The user could not use their primary laptop for university classes, coding, or daily tasks while training was active.
- **Why Local Training Became Unsustainable for the Long Term (Future Bottlenecks):**
  1. **Replay Buffer RAM Scaling:** At 5M frames, the replay buffer scaled to 100K transitions. At 7M+ frames, it scaled to 200K transitions. At 200K transitions, RAM consumption encroached on 15–16 GB. Any memory spike created a catastrophic risk of Out-Of-Memory crashes or swapping to SSD (causing extreme drive thrashing and sim pauses).
  2. **Interrupted Uptime & Reliability:** Laptops cannot maintain 24/7 uninterrupted uptime: forced Windows updates, accidental sleep/hibernation when closing the lid between classes, and power glitches risked corrupting the in-memory replay buffer and model weights.
  3. **Multi-Map Scaling Bottleneck:** Transitioning to the planned multi-map generalization curriculum requires interleaving 3 to 5 tracks simultaneously with 4+ parallel collectors and a 400K+ transition buffer. A 16 GB laptop cannot host this workload without crashing.
- **The Strategic Migration Milestone:**
  - Once the model proved it could reliably reach competitive pace locally (**54.000s** at ~10.64M frames), training was migrated to an Azure Cloud VM (`Standard_E4as_v4`, 4 vCPUs AMD EPYC, **32 GiB RAM**).
  - The cloud VM eliminated laptop wear, provided **double the RAM (32 GB)** to safely support huge replay buffers, and enabled 24/7 autonomous training with self-healing recovery.
- **The Solution for README:**
  Add a dedicated **"💻 Hardware & Engineering Journey"** section in `README.md` detailing:
  - Phase 1 (0 to 10.6M frames): The Local Laptop Grind (16 GB RAM, 2 collectors, thermal management, local bottlenecks).
  - Phase 2 (10.6M+ frames): The Cloud Migration to Azure (32 GB RAM, 24/7 compute, headless DirectX 9 workaround, self-healing process restarts).

---

## 3. Category B: Missing Artifacts, Verifiability & Baselines

### B.1 Broken Clone: Missing `maps/ESL-Hockolicious_0.5m_cl2.npy`
- **The Root Cause:** In `.gitignore`, lines 15–16 read:
  ```gitignore
  /maps/*
  !/maps/map5_0.5m_cl.npy
  ```
  This ignored all maps except `map5_0.5m_cl.npy`. While `ESL-Hockolicious_0.5m_cl2.npy` existed on local disk (193,616 bytes), it was excluded from git. Anyone cloning the repository gets a broken clone where `scripts/train.py` crashes on launch.
- **The Solution:**
  1. Update `.gitignore`:
     ```gitignore
     /maps/*
     !/maps/map5_0.5m_cl.npy
     !/maps/ESL-Hockolicious*
     ```
  2. Track both required map assets in git:
     - `maps/ESL-Hockolicious_0.5m_cl2.npy` (193 KB)
     - `maps/ESL-Hockolicious.Challenge.Gbx` (challenge file)

### B.2 Empirical Baseline Comparison (Upstream vs Linesight Run)
- **The Discovery:** The repository contains TensorBoard logs for an upstream baseline run (`tensorboard/default_hocko_run`) spanning 0 to 6.0M frames on `ESL-Hockolicious`:
  - `default_hocko_run` (upstream baseline): Best lap achieved was **54.110s** at 5.99M frames.
  - `hocko_v1` (our run): Reached **54.100s** at 6.06M frames, **54.010s** at 7.07M frames, and **54.000s** at 10.64M frames.
- **The Solution:**
  Document this baseline explicitly. Compare the two runs side-by-side with hard data:
  | Run | Speedslide Shaping | Frames | Peak Lap | Rolling Mean |
  |---|---|---|---|---|
  | Upstream Default Baseline (`default_hocko_run`) | Disabled (`0.0`) | 6.00M | 54.110s | ~55.80s |
  | Extended Campaign (`hocko_v1`) | Enabled (`0.003`) | 10.65M | 54.000s | ~55.45s |

### B.3 Verifiable Replay Proof for the 54.000s Personal Best
- **The Proof:** The exact inputs for the 54.000s run are preserved at:
  `save/hocko_v1/best_runs/hock_54_000/hock_54_000.inputs` (138 input timing lines).
- **The Solution:**
  1. Commit `hock_54_000.inputs` and copy it to a public verification folder `replays/hock_54_000.inputs`.
  2. Create a clean guide `docs/VERIFY_REPLAY.md` detailing how any user can:
     - Load `ESL-Hockolicious.Challenge.Gbx` in TrackMania Nations Forever with TMInterface.
     - Execute `execute inputs/hock_54_000.inputs`.
     - Witness the exact deterministic 54.000s lap in-game and export the official `.Replay.Gbx`.

### B.4 Human World Record Reference Citation
- **The Reference:** The 53.760s record was set by **tastyy** on June 26, 2022 on `ESL-Hockolicious`, succeeding Lundstrm's 53.80s record.
- **The Solution:**
  Add formal citation and links in `README.md`:
  - Driver: **tastyy**
  - Date: **June 26, 2022**
  - Track: **ESL-Hockolicious** (ESL Speed & Tech tournament map)
  - Proof/Video: Reference the recorded run on YouTube.

### B.5 Visual Learning Curve & Driving Artifacts
- **The Solution:**
  1. Write a script `scripts/tools/plot_learning_curves.py` to parse `tensorboard/default_hocko_run`, `tensorboard/hocko_v1`, and `tensorboard/hocko_v1_2`.
  2. Generate a high-resolution, publication-grade plot: `docs/assets/pb_vs_frames.png` displaying:
     - Best lap time (PB) vs frames (0 to 10.65M).
     - Horizontal dashed benchmark at 53.760s (Human WR - Tastyy).
     - Annotated milestone markers: Baseline 6M cutoff, 7.07M 54.01s plateau, 8.5M LR perturbation, and 10.64M 54.00s PB.
  3. Embed this plot prominently in `README.md`.

---

## 4. Category C: Core Engineering Hardening & Bug Fixes

### C.1 Silent Failure Risk: `strict=False` in Weight Loading
- **The Risk:** `collector_process.py` (line 48) and `learner_process.py` (lines 138–139) currently call `load_state_dict(..., strict=False)` inside a broad `try/except` block. If an optimizer crash, disk corruption, or mismatched model weights occur, PyTorch silently skips missing layers and proceeds with randomly initialized weights without failing.
- **The Solution:**
  Replace permissive `strict=False` with strict validation that permits *only* the known non-persistent normalization buffers:
  ```python
  load_result = model.load_state_dict(state_dict, strict=False)
  allowed_missing = {"float_inputs_mean", "float_inputs_std"}
  real_missing = [k for k in load_result.missing_keys if k not in allowed_missing]
  if real_missing or load_result.unexpected_keys:
      raise RuntimeError(
          f"CRITICAL: Model checkpoint corruption! Missing keys: {real_missing}, "
          f"Unexpected keys: {load_result.unexpected_keys}"
      )
  ```

### C.2 Self-Healing Auto-Recovery Block Flaws in `game_instance_manager.py`
- **The Bugs Found in Review:**
  1. `os.system("taskkill /PID ...")` returns an exit code and never raises an exception. The `try/except` around it is dead code, and child launcher processes (`TMLoader`) are not killed.
  2. `self.latest_map_path_requested` was **not reset**. Because line 760 checks `map_path != self.latest_map_path_requested`, a relaunched game on the same map would never trigger `self.request_map()`, leaving the game stuck in the main menu!
  3. `ensure_not_minimized` was removed, which could cause a kill-and-relaunch loop if the window was merely minimized.
  4. There was no recovery counter logged to monitoring.
- **The Solution:**
  Refactor lines 766–788 of `game_instance_manager.py`:
  1. Use `subprocess.run(["taskkill", "/PID", str(self.tm_process_id), "/T", "/F"], capture_output=True)` to kill the full process tree.
  2. Explicitly reset `self.latest_map_path_requested = None` so the relaunched instance reloads the track immediately.
  3. Maintain window state checking before force-kill.
  4. Add `self.recovery_count += 1` and record it into rollout statistics.
  5. Consider reducing `tmi_protection_timeout_s` from 500s down to 120s to reduce dead time on stalled sockets.

### C.3 Hardware Abstraction Centralization (`trackmania_rl/device.py`)
- **The Problem:** `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` was copy-pasted across `iqn.py`, `analysis_metrics.py`, `buffer_utilities.py`, and local `device` variables in collector and learner processes.
- **The Solution:**
  Create a single canonical module:
  `trackmania_rl/device.py`:
  ```python
  import torch

  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  IS_CUDA = DEVICE.type == "cuda"
  ```
  Import `DEVICE` and `IS_CUDA` across all modules.

### C.4 Configuration Drift & Runtime Reproducibility Snapshotting
- **The Problem:** `config_copy.py` is the file that actually executes and can be modified hot on-the-fly, but it is gitignored and untracked. The committed `config.py` is merely an initial template.
- **The Solution:**
  In `train.py` / `learner_process.py`, add automated configuration logging:
  - Whenever a new PB is saved or every 1,000,000 frames, save a copy of `config_copy.py` into:
    `save/{run_name}/config_snapshots/config_{frames}_frames.py`.

### C.5 Speedslide Reward Shaping Decay
- **The Problem:** The speedslide shaping reward (`0.003`) is applied as a constant bonus throughout training. In late-stage fine-tuning, dense shaping rewards can compete with the primary objective (finishing the lap in minimum time) by rewarding slight over-sliding rather than carrying straight-line exit speed.
- **The Solution:**
  Document this shaping dynamic in `GENERALIZATION_STRATEGY.md`, and for post-WR or generalization runs, introduce an annealing schedule that tapers shaping bonuses to 0 as $\epsilon \to 0.03$.

### C.6 Throughput Accounting Reconciliation (80x Nominal vs Measured FPS)
- **The Problem:** Stating "80x speed" and "~15 hours of sim time" creates confusion between wall-clock time and in-game simulation time.
- **The Mathematical Reality:**
  - 1 frame (action) = 5 engine steps $\times$ 10 ms = **50 ms (0.05 seconds of in-game driving)**.
  - $10,650,000\text{ frames} \times 0.05\text{ s} = 532,500\text{ seconds} = \mathbf{147.9\text{ hours of simulated physics}}$.
  - Wall-clock time to collect 10.65M frames was **~15 hours** on 2 collectors on the local workstation with RTX 4060 GPU.
  - Aggregate measured throughput: $10,650,000 \div (15.24 \times 3600) \approx \mathbf{194 - 197\text{ frames per second}}$ on local GPU (~98 FPS per collector instance).
  - Effective real-time speedup: $197\text{ frames/s} \times 0.05\text{ s} \approx \mathbf{9.85\times\text{ real-time aggregate speedup}}$.
  - *Note on Headless Azure CPU VM:* Running on 4 vCPUs without a GPU (software rendering via WARP and CPU inference) yielded only **~15 aggregate FPS**, demonstrating that hardware GPU acceleration is mandatory for high-throughput TrackMania simulation.
- **The Solution:**
  Explicitly define these metrics in `README.md` and `SESSION_NOTES_AND_MIGRATION_STATE.md`:
  - **Nominal Physics Multiplier:** 80× (`running_speed = 80`)
  - **In-Game Physics Time:** ~148 hours
  - **Local Workstation Wall-Clock Training Time:** ~15.2 hours (at ~197 FPS)
  - **Headless Cloud CPU Throughput:** ~15 FPS (software-rendering bound)

---

## 5. Category D: Repository Hygiene, Security & Git Lineage

### D.1 "TMNF Credentials" Clarification & Config Templating
- **The Problem:** The README directory tree describes `user_config.py` as holding *"TMNF credentials"*, leading reviewers to suspect passwords or auth tokens are committed.
- **The Reality:** `user_config.py` only contains local filesystem paths and `username = "stormy"`. No passwords exist.
- **The Solution:**
  1. Fix README description:
     `├── user_config.py      # Local paths & TMNF account username (no credentials)`
  2. Create `config_files/user_config.py.template` with clean default placeholder paths.
  3. Add a check ensuring no sensitive environment variables or tokens are exposed.

### D.2 Upstream License Integration (MIT)
- **The Problem:** The repo has no `LICENSE` file; README says "See upstream repository".
- **The Upstream License:** `setup.py` in `Linesight-RL/linesight` explicitly declares:
  `license='MIT'`.
- **The Solution:**
  Add a full `LICENSE` file containing the standard MIT License, crediting:
  `Copyright (c) 2024 Linesight-RL Contributors` and `Copyright (c) 2026 Project Linesight Contributors`.

### D.3 Upstream Lineage & Visible Diff Specification
- **The Problem:** Because the local git repository was initialized without an upstream merge base, `git diff upstream/main...HEAD` fails with `fatal: no merge base`.
- **The Solution:**
  1. Maintain and commit `relevant_changes.diff` showing the clean, exact diff between `upstream/main` and `HEAD` across all functional files (`config_files/`, `trackmania_rl/`, `requirements_pip.txt`).
  2. Document the exact lineage in `ATTRIBUTION.md`, providing the upstream commit hash and command to reproduce the diff.

### D.4 Line Endings & Mode Noise (`.gitattributes`)
- **The Problem:** `docs/make.bat` has CRLF churn and `scripts/launch_game_agade.sh` had executable permission noise.
- **The Solution:**
  Create `.gitattributes`:
  ```gitattributes
  * text=auto
  *.bat text eol=crlf
  *.sh text eol=lf
  *.py text eol=lf
  *.md text eol=lf
  *.npy binary
  *.Gbx binary
  *.torch binary
  *.joblib binary
  ```

---

## 6. Step-by-Step Implementation Roadmap

Once approved, implementation will proceed in four clean phases:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           IMPLEMENTATION PHASES                                   │
├───────────────────────┬───────────────────────────┬───────────────────────────────┤
│ Phase 1: Hygiene &    │ Phase 2: Code Hardening   │ Phase 3: Empirical Validation │
│ Git Asset Recovery    │ & Bug Fixes               │ & Documentation Rewrite       │
│  - Update .gitignore  │  - Centralize device.py   │  - Extract baseline TB logs   │
│  - Track map files    │  - Strict weight loading  │  - Generate learning curves   │
│  - Add LICENSE (MIT)  │  - Patch game manager     │  - Rewrite README.md          │
│  - Add .gitattributes │    recovery & reset logic │  - Update ATTRIBUTION.md      │
│  - user_config template│  - Config snapshotting   │  - Update GENERALIZATION.md   │
└───────────────────────┴───────────────────────────┴───────────────────────────────┘
```

### Detailed Checklist:
- [ ] **Phase 1: Hygiene, Assets & Licensing**
  - Update `.gitignore` to allow `maps/ESL-Hockolicious*`.
  - Ensure `ESL-Hockolicious_0.5m_cl2.npy` and `ESL-Hockolicious.Challenge.Gbx` are tracked.
  - Create `LICENSE` (MIT).
  - Create `.gitattributes` to eliminate CRLF noise.
  - Create `config_files/user_config.py.template`.
- [ ] **Phase 2: Code Hardening**
  - Create `trackmania_rl/device.py` and centralize hardware detection.
  - Replace `strict=False` in `collector_process.py` and `learner_process.py` with validated key checking.
  - Fix `trackmania_rl/tmi_interaction/game_instance_manager.py`:
    - Tree process kill via `subprocess.run`.
    - Reset `self.latest_map_path_requested = None`.
    - Restore window minimization checks.
    - Add `recovery_count` instrumentation.
  - Add config snapshotting in `train.py`.
- [ ] **Phase 3: Empirical Validation & Learning Curve Plot**
  - Run `scripts/tools/plot_learning_curves.py` to generate `docs/assets/pb_vs_frames.png`.
  - Copy `save/hocko_v1/best_runs/hock_54_000/hock_54_000.inputs` to `replays/hock_54_000.inputs`.
  - Write `docs/VERIFY_REPLAY.md`.
- [ ] **Phase 4: Presentation & Documentation Overhaul**
  - Rewrite `README.md` with grounded framing, correct hyperparameters, throughput math, and the learning curve plot.
  - Rewrite `ATTRIBUTION.md` with honest contribution scopes (demoting the 8.5M causal claim).
  - Update `GENERALIZATION_STRATEGY.md` with correct RL terminology (loss of plasticity / replay domination).

---

> ✋ **Confirmation Required:**  
> Review the solutions above. Once you confirm approval, I will begin implementing them phase by phase. Until then, zero code or configuration files will be modified.

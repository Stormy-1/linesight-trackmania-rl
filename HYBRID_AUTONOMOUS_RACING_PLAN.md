# Project Linesight — Hybrid Autonomous Racing Plan

> **Status:** Planning and architecture document. This file describes the
> intended future direction; it does not activate or modify the current
> training curriculum.
>
> **Authoritative current-state source:** `HANDOVER_DOC.md` remains the source
> of truth for active branches, checkpoints, run state, and execution rules.
>
> **Current research direction:** evolve Linesight from a mostly end-to-end
> deep-RL driving agent into a hybrid autonomous-racing system that combines
> robotics-style state estimation, map-aware planning, classical control,
> hierarchical/residual RL, recovery behavior, uncertainty estimation, and
> safety supervision.

## 1. Why change the architecture?

The existing Linesight agent demonstrated that deep RL can learn strong
TrackMania driving behavior. The foundational Hockolicious model reached a
best local time of `54.000s` against a `53.760s` human reference. However,
the generalization experiments also exposed the limits of asking one policy to
learn every responsibility at once:

- infer its state from pixels and telemetry;
- understand where it is on a track;
- predict upcoming geometry;
- choose a racing line and target speed;
- control steering, throttle, braking, slides, jumps, and landings;
- recover after mistakes; and
- learn all of this while protecting behavior already learned on earlier maps.

This is a difficult partially observed control problem. A single end-to-end
policy can solve it, but it is difficult to debug, difficult to make safe, and
especially vulnerable to catastrophic forgetting during sequential map
training.

The proposed change is not to discard RL. It is to give RL a better role in a
larger system:

> Use explicit geometry, estimation, planning, and safety mechanisms for the
> parts that benefit from structure; use learning for aggressive optimization,
> difficult maneuvers, adaptation, and recovery.

This mirrors the layered design of practical autonomous vehicles and mobile
robots while preserving the research value of deep reinforcement learning.

## 2. Project vision

The target system is a map-aware autonomous racing stack:

```text
Track/map representation
          ↓
State estimation and map matching
          ↓
Local trajectory and speed planning
          ↓
Nominal controller / MPC
          ↓
Hierarchical policy or residual RL
          ↓
Uncertainty estimation
          ↓
Safety supervisor and fallback controller
          ↓
TrackMania action execution
```

The long-term objective is not merely a world-record lap on one map. It is a
driver that can:

- complete unfamiliar maps reliably;
- maintain useful behavior after mistakes;
- improve speed without sacrificing completion rate;
- explain which subsystem made a decision;
- recognize when its learned policy is uncertain; and
- retain competence on maps already conquered.

## 3. What remains unchanged for now

The active `gen_v3_r1` Stage 1 experiment remains a controlled baseline. It
must be allowed to produce evidence before the hybrid system is introduced.

Current Stage 1 scope:

```text
Maps:                         Hockolicious + A01
Collectors:                   1
Simulation speed:             80×
TMInterface protection:       60 seconds
Max learner batches/rollout:  8
```

The following checkpoints and evidence branches remain protected:

- `save/hocko_v1_foundational_checkpoint/`
- `save/hocko_v1/`
- `save/gen_v1/`
- `save/gen_v2/`
- `save/gen_v3/`
- `save/gen_v3_r1/` as the current corrective experiment

The hybrid work should use a new branch or explicitly named experiment. It
must not silently change the active Stage 1 configuration or overwrite the
foundational checkpoint.

## 4. Layered system design

### 4.1 Track and map representation

The system should represent the map in a form useful to both planning and
learning. A first representation can use the already available centerlines
and virtual checkpoints. Later versions can add derived geometric features.

Candidate map features include:

- centerline position and tangent;
- local curvature and curvature change;
- track width estimate;
- upcoming checkpoint distance;
- jump, landing, or elevation-change indicators;
- known high-risk sections;
- target speed profile;
- recovery zones; and
- map-section identifiers for skill conditioning.

**Why:** explicit geometry gives the controller a stable frame of reference.
It reduces the burden on the neural network and makes failures easier to
diagnose. The map representation can also be shared across maps even when the
visual appearance changes.

### 4.2 State estimation

The estimator should transform raw telemetry, images, and map information into
a structured state. A useful state may include:

```text
progress along the track
lateral distance from centerline
heading error relative to centerline tangent
current speed and acceleration estimate
target speed and speed error
upcoming curvature and curvature trend
recent steering/throttle/brake history
wheel-ground contact proxies
sliding and grounded-sliding proxies
airborne state
progress rate
confidence or uncertainty values
```

**Why:** the current policy receives useful inputs, but an explicit state makes
the important control variables visible. It also permits classical controllers
and RL policies to operate on the same state interface.

The first estimator should use privileged game telemetry because it is the
fastest way to validate the architecture. A later vision-only estimator can
be added as a separate research problem rather than mixing localization
uncertainty into the first controller experiment.

### 4.3 Map matching and localization

Map matching estimates where the vehicle is relative to the known track. It
can associate the current state with:

- the nearest centerline segment;
- a lookahead window of future segments;
- the current map section;
- the next virtual checkpoint; and
- a local coordinate frame for planning.

**Why:** a controller cannot track a trajectory reliably without knowing which
part of the trajectory is currently relevant. Map matching also enables
section-specific skills and metrics.

This is robotics-style localization, but it is not necessarily full SLAM.

### 4.4 Local trajectory and speed planning

The planner should produce a short-horizon target rather than only a single
action. Its output may include:

```text
lookahead centerline points
desired lateral offset / racing line
target speed over the horizon
braking and acceleration targets
preferred maneuver mode
safe fallback trajectory
```

The first planner can be conservative and centerline-based. Later versions can
learn a faster racing line from successful trajectories.

**Why:** many failures are caused by entering a turn or landing at the wrong
speed, not merely by choosing the wrong instantaneous steering value. Planning
over a short horizon makes speed and curvature visible to the controller.

### 4.5 Nominal controller and trajectory tracking

A deterministic controller should provide a reliable baseline action. Candidate
controllers include:

- Pure Pursuit;
- Stanley steering;
- lookahead heading-error control;
- curvature-aware steering;
- speed-profile tracking; and
- a simple kinematic or empirical TrackMania dynamics model.

The first objective is completion, not record pace. A slow but repeatable
controller is valuable because it provides a fallback, an imitation-learning
source, and a reference against which RL improvements can be measured.

### 4.6 Model Predictive Control

MPC can optimize a short sequence of actions using a local dynamics model. A
first objective function could penalize:

```text
lateral tracking error
heading error
speed error
steering changes
predicted sliding risk
loss of recovery margin
```

It can also enforce limits on steering, throttle, braking, and predicted
lateral deviation.

TrackMania dynamics are not guaranteed to match a simple bicycle model, so MPC
should begin as an approximate local controller. The model can be calibrated
from recorded rollouts. MPC does not need to replace RL; it can be the nominal
controller, a local action evaluator, or a safety fallback.

### 4.7 Residual reinforcement learning

The safest initial hybrid learning formulation is residual RL:

```text
nominal controller action + learned correction = final action
```

The nominal controller handles basic track following. The RL policy learns to:

- reduce lap time;
- choose better racing lines;
- handle map-specific geometry;
- manage slides and landings;
- exploit game-specific dynamics; and
- recover from states where the nominal controller is insufficient.

**Why:** RL no longer has to rediscover basic localization and steering from
scratch. Exploration can begin near a functioning policy, reducing dangerous
random behavior and improving sample efficiency.

The residual must be bounded at first. The supervisor should be able to reduce
or disable it when uncertainty or risk is high.

### 4.8 Recovery control

Normal racing and recovery should be treated as related but distinct skills.
The system should detect recovery states such as:

- large lateral error;
- large heading error;
- sustained sliding;
- unexpected airborne motion;
- unstable landing;
- low progress for too long;
- high curvature approached at excessive speed; or
- disagreement between the learned policy and nominal controller.

Possible recovery modes include:

```text
RECOVER_LEFT
RECOVER_RIGHT
BRAKE_AND_ALIGN
REDUCE_SLIDE
STABILIZE_LANDING
REJOIN_CENTERLINE
RESTART_SAFE_LINE
```

**Why:** recovery is central to robust map conquest. A policy that is slightly
slower but can recover repeatedly is more useful than a faster policy that
fails irrecoverably after one mistake.

### 4.9 Uncertainty estimation

The system should estimate when its action is unreliable instead of assuming
that every high-value prediction is safe. Candidate signals include:

- IQN quantile spread;
- disagreement between an ensemble of policies;
- disagreement between the nominal controller and RL residual;
- prediction error from a learned dynamics model;
- distance from states represented in the training data; and
- repeated contact/sliding or progress anomalies.

**Why:** uncertainty enables the supervisor to lower the RL residual, choose a
conservative trajectory, or switch to recovery. This is safer than treating a
confident-looking neural-network output as ground truth.

### 4.10 Safety supervision and fallback behavior

The safety supervisor is a separate decision layer. It should be allowed to
reject or modify a learned action when constraints are violated.

Examples:

```text
if predicted lateral error is too large:
    limit steering aggressiveness

if speed is too high for upcoming curvature:
    prefer braking or conservative throttle

if uncertainty is high:
    reduce residual authority or use the nominal controller

if recovery margin is nearly zero:
    use emergency stabilization behavior
```

The supervisor should expose a reason code for every intervention. That makes
it possible to distinguish a policy failure from a planner, estimator, or
safety-layer intervention.

## 5. SLAM: where it fits and where it does not

SLAM means simultaneous localization and mapping. It is designed for a robot
that must estimate its pose while building an unknown map from noisy sensors.

Full SLAM is probably not the first tool needed here because:

- TrackMania maps are known before the run;
- the environment is mostly static;
- centerlines and virtual checkpoints already provide map structure;
- useful game telemetry is available; and
- the first goal is reliable control, not discovering an unknown world.

The robotics concepts that are immediately relevant are:

- state estimation;
- map matching;
- local coordinate frames;
- trajectory tracking;
- local planning;
- dynamics identification;
- MPC;
- recovery control; and
- uncertainty-aware safety decisions.

A SLAM-like visual localization subsystem can become a later experiment. For
example, the project could deliberately hide privileged position telemetry and
ask a vision model to estimate track-relative pose from frames. That should be
a separate milestone so localization uncertainty does not obscure the first
hybrid-controller results.

## 6. RL roles to investigate

The project should compare several RL formulations rather than assume that the
current IQN action policy is the final architecture.

### 6.1 Existing IQN policy as a baseline

The current IQN system remains valuable as a reference implementation. It can
serve as:

- the end-to-end baseline;
- a residual policy;
- a recovery policy;
- a high-level maneuver selector; or
- an uncertainty source through its return distribution.

Keeping this baseline makes every new architecture measurable against a known
system.

### 6.2 Residual RL

The policy outputs a bounded correction to a nominal controller. This should be
the first hybrid-RL experiment because it preserves a functioning fallback and
reduces the dangerous exploration space.

### 6.3 Hierarchical RL

A high-level policy selects a maneuver or driving mode, while a low-level
controller executes it. Candidate high-level modes are:

```text
FOLLOW_TRACK
BRAKE_FOR_CORNER
ACCELERATE
PREPARE_JUMP
STABILIZE_LANDING
MANAGE_SLIDE
RECOVER
```

This separates the question “what should the car be trying to do?” from “what
exact input should be sent this frame?”

### 6.4 Skill or option policies

Reusable skills can be trained or extracted for:

- normal cornering;
- acceleration;
- braking;
- speed slides;
- air control;
- landing stabilization; and
- recovery.

A high-level policy can select among these skills. This also offers a natural
way to transfer behavior to new maps without retraining every action from
scratch.

### 6.5 Imitation learning and offline data

Successful human, foundational-agent, classical-controller, and RL trajectories
can initialize a stable policy through behavior cloning or offline RL.

This is useful because random exploration is especially expensive in a racing
environment: a failed rollout may provide little useful progress information.

## 7. Phased implementation roadmap

The phases are deliberately ordered so each phase produces a useful artifact
and a measurable baseline before the next source of complexity is added.

### Phase 0 — Complete and archive the current RL baseline

**Purpose:** obtain a clean reference before architectural changes.

Tasks:

- finish the controlled `gen_v3_r1` Stage 1 experiment;
- preserve its stage report, episode metrics, logs, and checkpoints;
- record Hockolicious and A01 completion, timeout, and infrastructure rates;
- record learner stability, update counts, and rollout throughput; and
- document what the current IQN baseline can and cannot do.

**Why first:** without a baseline, improvements from a hybrid system cannot be
separated from ordinary run-to-run variation.

**Exit criteria:** a terminal Stage 1 decision or a documented infrastructure
stop, with all available evidence preserved.

### Phase 1 — Build structured state and offline diagnostics

**Purpose:** turn raw rollouts into robotics/control quantities.

Tasks:

- define a versioned structured-state schema;
- compute lateral and heading errors relative to centerlines;
- derive curvature and lookahead features;
- record progress rate and target-speed error;
- retain wheel-contact and sliding values as explicitly labeled proxies;
- identify map sections and high-risk events; and
- visualize state trajectories offline without changing the live policy.

**Why:** this creates the common interface for planners, controllers, and RL.
It also lets us understand failure modes before adding more model complexity.

**Exit criteria:** deterministic feature generation from saved rollouts, with
unit tests for map matching, curvature, progress, and proxy telemetry.

### Phase 2 — Implement a conservative completion controller

**Purpose:** create a non-RL driver that can follow the known track.

Tasks:

- implement a first centerline/lookahead controller;
- add a conservative speed profile;
- support steering and throttle/brake action conversion;
- log tracking error and controller interventions; and
- evaluate completion and repeatability on Hockolicious and A01.

**Why:** a slow controller provides a fallback, a data source, and a sanity
check for the state and map representation.

**Exit criteria:** repeatable completion at a documented success rate, even if
the lap time is slower than the RL baseline.

### Phase 3 — Add trajectory and speed planning

**Purpose:** move beyond reactive single-action steering.

Tasks:

- generate a local lookahead trajectory;
- derive curvature-aware target speeds;
- add braking/acceleration planning;
- represent jump and landing sections; and
- expose a safe fallback trajectory.

**Why:** many failures arise from poor anticipation. Short-horizon planning
should improve corner entry, speed management, and recovery margin.

**Exit criteria:** measurable reduction in excessive-speed and large tracking
error events without reducing completion rate.

### Phase 4 — Add residual RL

**Purpose:** let learning improve the conservative controller without removing
its safety structure.

Tasks:

- define a bounded residual action space;
- start with low residual authority;
- train on structured state plus optional visual features;
- compare nominal-only versus nominal-plus-residual behavior; and
- gate residual authority using uncertainty and safety checks.

**Why:** this is the most direct hybrid system and the least disruptive path
from the current IQN agent.

**Exit criteria:** faster or more capable behavior at equal or better
completion and recovery rates, with no unexplained safety regressions.

### Phase 5 — Train a dedicated recovery policy

**Purpose:** improve robustness after mistakes rather than only optimizing
ideal trajectories.

Tasks:

- generate perturbed starting states from successful rollouts;
- create controlled lateral, heading, speed, and sliding perturbations;
- label recovery opportunities and outcomes;
- train a recovery skill or policy; and
- evaluate recovery success separately from clean-lap success.

**Why:** robust map conquest requires the ability to recover from non-ideal
states. This is a separate distribution from normal racing.

**Exit criteria:** recovery success improves on held-out perturbations without
damaging clean driving.

### Phase 6 — Add empirical dynamics and MPC

**Purpose:** use model-based control where a short-horizon prediction is useful.

Tasks:

- collect action/state transition data;
- fit and validate a local dynamics approximation;
- implement a bounded-horizon MPC or action-sequence evaluator;
- compare MPC against the nominal controller; and
- use MPC as a fallback before giving it full authority.

**Why:** MPC can anticipate the consequences of steering and speed choices,
but it should be introduced only after the state representation and baseline
controller are trustworthy.

**Exit criteria:** MPC provides measurable tracking or safety benefits on a
defined test set and fails safely when its model is inaccurate.

### Phase 7 — Introduce hierarchical RL and skills

**Purpose:** separate maneuver selection from low-level action execution.

Tasks:

- define a small high-level maneuver vocabulary;
- train or hand-design low-level skills;
- train a high-level selector with temporal abstraction;
- expose skill transitions and failure reasons; and
- compare hierarchical behavior to flat residual RL.

**Why:** temporal abstraction can make long maps easier to solve and makes the
policy easier to interpret.

**Exit criteria:** high-level decisions are stable, skills are reusable across
map sections, and performance is not worse than the flat baseline.

### Phase 8 — Continual multi-map learning

**Purpose:** add maps without destroying previously acquired competence.

Tasks:

- maintain rehearsal data for conquered maps;
- evaluate all conquered maps at every later stage;
- use protected validation episodes that never enter replay;
- gate promotion on old-map retention as well as new-map improvement;
- test shared versus map-conditioned policies; and
- use rollback when a previously accepted map regresses.

**Why:** sequential curriculum without retention is vulnerable to catastrophic
forgetting. Every conquered map must remain part of the safety contract.

**Exit criteria:** new maps improve while previously accepted maps remain above
their documented reliability thresholds.

## 8. Experiment and evaluation protocol

Every architectural change should be evaluated as a controlled experiment.
The project should not mix a speed change, a new map curriculum, a new policy
architecture, and a new safety layer in one unexplained run.

### 8.1 Baselines to preserve

At minimum, each experiment should compare against:

1. the current IQN end-to-end policy;
2. the nominal classical controller;
3. the hybrid controller without RL residual; and
4. the hybrid controller with RL residual.

For hierarchical experiments, also compare against the flat residual policy.

### 8.2 Behavioral metrics

The primary metrics are behavioral rather than loss-only:

```text
completion rate
finish rate
timeout rate
infrastructure-failure rate
mean completion percentage
median and tail finish time
recovery success rate
large tracking-error frequency
sliding and airborne proxy rates
intervention rate of the safety supervisor
```

Exact barrier collisions must remain explicitly unavailable unless a real
validated telemetry source is added. Wheel contact, sliding, material, and
airborne observations must remain labeled as proxies.

### 8.3 Learning and systems metrics

Also record:

```text
frames and transitions per wall-clock minute
learner wait time
updates per rollout
loss and gradient statistics
IQN uncertainty / quantile spread
policy-controller disagreement
replay composition by map and mode
checkpoint provenance and hashes
```

This separates a modeling failure from a desktop, TrackMania, or TMInterface
failure.

### 8.4 Test strategy

Pure functions should be tested without TrackMania:

- centerline projection;
- map matching;
- curvature and lookahead extraction;
- target-speed generation;
- recovery-state classification;
- uncertainty thresholds;
- safety action vetoes;
- replay routing; and
- checkpoint save/restore behavior.

Integration tests should use recorded rollouts or a deterministic simulator
stub before any visible TrackMania run. Visible runs should be reserved for
validating the actual game interface and end-to-end behavior.

## 9. Continual learning and anti-forgetting policy

The staged curriculum must protect every map that has been accepted, not only
Hockolicious.

When adding a new map, the default future curriculum should be closer to:

```text
Stage 1: Hockolicious + A01
Stage 2: Hockolicious + A01 + A03
Stage 3: Hockolicious + A01 + A03 + A05
Stage 4: Hockolicious + A01 + A03 + A05 + A07
Stage 5: Hockolicious + A01 + A03 + A05 + A07 + A11
```

If the full replay stream becomes too expensive, use a deliberately designed
rehearsal mixture instead of silently dropping old maps. The mixture should
record its per-map proportions and be tested for bias.

Each later stage should:

- keep protected validation episodes separate from replay;
- evaluate the new map and all previously accepted maps;
- require new-map improvement;
- require old-map retention;
- stop on infrastructure-failure clusters; and
- restore the stage baseline on behavioral rollback.

This converts catastrophic forgetting from an assumption into a measurable
acceptance criterion.

## 10. Safety, branches, and checkpoint discipline

The hybrid work should follow the same safety rules as the current corrective
branch:

- never overwrite the foundational checkpoint;
- never resume or overwrite historical evidence branches;
- use a new branch or explicitly named save directory for each architecture;
- record configuration, source hashes, and checkpoint origin;
- keep training and evaluation data separate;
- do not treat a lower loss as proof of better driving;
- stop on repeated TMInterface recovery loops; and
- never claim exact collision telemetry when only proxies are available.

Suggested future experiment names include:

```text
hybrid_state_v1
hybrid_controller_v1
hybrid_residual_v1
hybrid_recovery_v1
hybrid_mpc_v1
hybrid_hierarchical_v1
```

The names are suggestions only. A branch should be created deliberately after
the corresponding design and preflight checks are complete.

## 11. Main technical risks

### Risk: classical controller is reliable but too slow

**Mitigation:** use it as a nominal policy and allow bounded residual RL to
optimize speed after completion is stable.

### Risk: approximate dynamics make MPC harmful

**Mitigation:** begin with short horizons, validate on recorded transitions,
limit MPC authority, and keep a fallback controller.

### Risk: residual RL learns to fight the controller

**Mitigation:** expose controller state to the policy, bound the residual, log
controller/policy disagreement, and gate residual authority by uncertainty.

### Risk: map matching is wrong near jumps or ambiguous geometry

**Mitigation:** use checkpoint/progress continuity, confidence scores, and a
fallback section identifier rather than trusting a single nearest-point match.

### Risk: safety supervision prevents useful exploration

**Mitigation:** measure intervention rate and compare safe exploration against
completion and learning progress. The supervisor should be conservative at
first but tunable from evidence.

### Risk: continual learning still forgets old maps

**Mitigation:** rehearsal, all-map evaluation, protected validation episodes,
map-conditioned policies where appropriate, and rollback gates.

### Risk: architecture becomes too complex to debug

**Mitigation:** add one layer at a time. Every phase must have a standalone
baseline, a clear interface, and an exit criterion.

## 12. Research questions

The hybrid direction creates several concrete research questions:

1. How much does structured state improve sample efficiency compared with raw
   frame-plus-telemetry IQN input?
2. Can a conservative controller achieve reliable completion on unfamiliar
   maps without RL?
3. Does residual RL improve speed while preserving controller reliability?
4. Which uncertainty signal best predicts an unrecoverable action?
5. Does a dedicated recovery policy improve map conquest more than additional
   normal-driving training?
6. Can a low-fidelity dynamics model support useful MPC despite TrackMania's
   game-specific physics?
7. Do hierarchical skills transfer better across maps than a flat action
   policy?
8. How much rehearsal is required to prevent forgetting as maps are added?
9. Is vision-based localization worth the complexity once privileged telemetry
   is removed?
10. Which subsystem contributes most to completion rate: planning, speed
    control, recovery, or safety supervision?

## 13. Immediate next steps after the current Stage 1 run

The next work should be analytical and offline first:

1. archive the completed `gen_v3_r1` evidence;
2. summarize Hockolicious/A01 behavior and infrastructure failures;
3. inspect saved rollout telemetry for tracking-error and recovery features;
4. define the structured-state schema;
5. implement and unit-test centerline projection/map matching;
6. build an offline visualization of progress, curvature, speed, and sliding
   proxies;
7. implement a slow deterministic centerline controller; and
8. compare its completion behavior against the IQN baseline.

Do not begin MPC, hierarchical RL, SLAM-like visual localization, or a new
multi-map training run until the state representation and conservative
controller have been validated.

## 14. Intended final project description

Project Linesight should ultimately be described as:

> A hybrid autonomous racing system for TrackMania that combines map-aware
> state estimation, trajectory and speed planning, classical/MPC control,
> hierarchical and residual reinforcement learning, learned recovery skills,
> uncertainty estimation, and safety supervision to achieve reliable
> cross-map completion.

This direction preserves the value of the existing deep-RL work while opening
up a broader robotics and autonomous-systems research program.

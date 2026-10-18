# TrackMania Generalization Strategy & Checkpoint Memo

## 📌 Context
This document serves as a memo for future agents and developers regarding the decision to branch our reinforcement learning training at roughly the **"95% mastery"** mark on the track `ESL-Hockolicious`.

**Status: The 95% Model is SAVED.** 
The exact weights and stats of this ~54.45s 95% mastery state have been successfully backed up to: 
`save/hocko_v1_foundational_checkpoint/`

## 🤔 What does "95% Mastery" mean?
In the context of Reinforcement Learning (RL) and TrackMania, "95% mastery" is not a strict mathematical accuracy metric (like in supervised image classification). Instead, it represents a specific intersection of **Performance** and **Behavioral Flexibility**.

1. **Relative Performance:** The "purist" human World Record for this track is `53.76s`. The agent is currently hitting robust times of `~56s` and peak exploratory times of `54.45s`. It is driving at roughly 95-96% of the theoretical maximum human limit.
2. **Behavioral Generality:** The model has successfully learned the *universal physics* of the game (how to accelerate, how to drift, how to recover from bumps, how to steer smoothly).
3. **Avoidance of "Micro-Optimization":** It has *not yet* entered the phase where it hardcodes its inputs to the exact pixel and frame timings of `ESL-Hockolicious`. It is still relying on visual/feature inputs to react, rather than memorizing a rigid script.

## 🛑 Why branch now? (The Risks of 100% Mastery)
If we were to let the model train to 100% WR mastery on this single map before attempting to train it on other maps, we would encounter two massive RL hurdles:

### 1. Entropy Collapse (The "Overconfidence" Problem)
As an RL policy approaches 100% mastery, its internal uncertainty (entropy) collapses. The probability distribution over its actions becomes a massive spike. For example, at a specific corner, it becomes 99.99% confident that `Right + Gas` is the ONLY correct action. 
If we take this 100% mastered model and drop it on a *new* track that requires `Left + Brake`, the model will confidently execute `Right + Gas`. Because its probability of trying anything else is near 0%, it will take an astronomically long time to "unlearn" its confidence and start exploring again. 
**The 95% model still has enough internal variance/uncertainty to quickly adapt to new environments.**

### 2. Extreme Track Bias (Overfitting)
A 100% mastered model stops learning "how to drive a car" and starts learning "how to solve this exact sequence of turns." It becomes overfit to the exact visual features of `ESL-Hockolicious`. The 95% model is a **General Driving Foundation** that can be used to seed training on any track.

## ⚠️ What else are we avoiding? (Catastrophic Forgetting)
Even with the 95% foundational model, we cannot simply say "Okay, now let's train on Map B." 

In Neural Networks, training sequentially (Map A -> Map B -> Map C) leads to **Catastrophic Forgetting**. As the network updates its weights to optimize for Map B, it will completely overwrite and destroy the knowledge it gained for Map A.

**The Strategy for True Generalization:**
To build a truly generalized TrackMania AI, we must use this `hocko_v1_foundational_checkpoint` to initialize a new training run, but **we must train on a cycle of multiple maps simultaneously**. 
By interleaving Map A, Map B, and Map C in the training environment's `map_cycle`, the neural network is forced to find shared, generalized weights that work across all distinct geometries, rather than forgetting one to learn the other.

---
**Status Update (Post-10.82M Frames):**
1. **Single-Track Phase Concluded:** Training concluded at **10,826,633 frames** with a Personal Best of **54.000s** (240ms from human WR). Between 10.64M and 10.82M frames, the PB remained strictly invariant while the rolling mean drifted (54.9s → 55.6s), empirically confirming complete feature saturation under 50ms action quantization.
2. **Branch Initialized:** We are branching from `save/hocko_v1_foundational_checkpoint` (54.45s state) under the working hypothesis that it preserves higher behavioral plasticity and adaptability than the saturated 10.64M/10.82M model. This hypothesis will be verified empirically through comparative fine-tuning and zero-shot baselines against the late checkpoint.
3. **High-Speed Generalization Profile:**
   - **Collectors:** `gpu_collectors_count = 2` (both instances running parallel rollouts).
   - **Simulation Speed:** `running_speed = 80` (yielding ~197 aggregate FPS on local GPU workstation; headless CPU instances without GPU render at ~15 FPS).
   - **Replay Buffer:** `50,000` to `100,000` transitions (fast cycling to prioritize new track data).
   - **Step 1: Controlled Zero-Shot Baseline:** Run official maps in `map_cycle` with `(False, False)` across both checkpoints ($N \ge 10$ runs per track) measuring distance fraction completed and termination causes before updating weights.
   - **Step 2: Interleaved Curriculum:** Cycle between new official maps (A01, A03, etc.) and `ESL-Hockolicious` to evaluate adaptation rate and prevent catastrophic forgetting.

**Branch Recovery Note:** The first `gen_v1` training attempt was intentionally stopped after its fresh replay buffer caused rapid loss and gradient growth while current evaluations timed out across all maps. The foundational checkpoint remains untouched. A clean `gen_v2` branch was bootstrapped with model weights only, reset learner counters, no inherited optimizer/scaler state, a lower fine-tuning learning rate, higher initial exploration, and equal Hockolicious curriculum weighting.

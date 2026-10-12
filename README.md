# 🏎️ Linesight — TrackMania RL World Record Agent

> **Teaching an AI to drive faster than any human — 240 milliseconds at a time.**

An autonomous racing agent trained with **Deep Reinforcement Learning** (Implicit Quantile Networks) to master *TrackMania Nations Forever*. The agent learns entirely from pixels and game telemetry — no hardcoded racing lines, no replay imitation, no human demonstrations.

<!-- TODO: Replace with showcase website link once built -->
<!-- 🌐 **[View the Full Story →](https://stormy-1.github.io/linesight-trackmania-rl/)** -->

---

## 📊 Results

| Metric | Value |
|---|---|
| **Track** | `ESL-Hockolicious` |
| **Agent's Personal Best** | **`54.000s`** |
| **Human Purist World Record** | **`53.760s`** |
| **Gap to WR** | **0.24s (240 ms)** — 99.6% of human perfection |
| **Total Training Frames** | 10,650,000+ (~15 hours of accelerated sim time) |
| **Training Speed** | 80× real-time physics simulation |

The agent is actively training on an Azure cloud VM, hunting down the final 240 milliseconds.

---

## 🧠 How It Works

### Architecture
- **Algorithm:** [Implicit Quantile Network (IQN)](https://arxiv.org/abs/1806.06923) — a distributional RL method that learns the full distribution of future returns, not just the expected value.
- **Input:** 160×120 downscaled game frames + floating-point telemetry (speed, RPM, steering state).
- **Output:** Discrete action selection across all combinations of `{Steer Left, Steer Right, Gas, Brake, No Input}`.
- **Game Interface:** [TMInterface](https://donadigo.com/tminterface) — a C++/AngelScript injection API that provides direct memory access to TrackMania's physics engine.

### Training Pipeline
```
┌──────────────────────────────────────────────────────────────────────┐
│                         Training Loop                                │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐  │
│  │  Collector 1 │───▶│              │    │                         │  │
│  │  (TM @ 80x)  │    │   Replay     │───▶│   IQN Learner Process   │  │
│  ├─────────────┤    │   Buffer     │    │   (Gradient Updates)    │  │
│  │  Collector 2 │───▶│  (200K txns) │    │                         │  │
│  │  (TM @ 80x)  │    └──────────────┘    └─────────────────────────┘  │
│  └─────────────┘                                                     │
└──────────────────────────────────────────────────────────────────────┘
```

- **Dual collectors** run independent TrackMania instances at 80× speed, gathering experience rollouts.
- A **central replay buffer** (200,000 transitions) stores experiences for off-policy learning.
- The **IQN learner** samples mini-batches and performs gradient descent on the quantile Huber loss.
- Weights are shared asynchronously: collectors pull fresh network weights every few rollouts.

### Key Hyperparameters
| Parameter | Value | Notes |
|---|---|---|
| Discount (γ) | 1.0 | Infinite horizon (annealed from 0.9 to 1.0 over 2.5M frames) |
| Exploration (ε) | 0.03 | Floor reached at ~3M frames |
| Learning Rate | 5e-5 → 1e-5 | Cosine-like schedule with a warmup bump at 8.5M frames |
| IQN Quantiles (N/N'/K) | 8/8/32 | Symmetric tau sampling |
| Batch Size | 512 | |
| Buffer Size | 200,000 | Expanded from 50K at 5M frames |

---

## ☁️ Cloud Infrastructure

Training outgrew the local laptop at ~10M frames. The agent now trains on Azure:

| Component | Spec |
|---|---|
| **VM** | `Standard_E4as_v4` (4 vCPU AMD EPYC, 32 GiB RAM) |
| **OS** | Windows Server 2022 Datacenter |
| **GPU** | None (CPU-only — Azure Student 0-core GPU quota) |
| **Region** | East Asia |
| **Cost** | ~$0.52/hr ($100 Azure for Students budget) |

> **Why not Google Colab?** TrackMania is a 32-bit Windows DirectX 9 executable. It requires live DLL injection, socket IPC, and an active display surface — none of which are possible in Colab's headless Linux containers. See [`CLOUD_MIGRATION.md`](CLOUD_MIGRATION.md) for the full technical analysis.

---

## 🛡️ Generalization Strategy

Pushing to 100% mastery on a single map risks **entropy collapse** (the policy becomes so confident it can't explore on new tracks) and **catastrophic forgetting** (training on Map B destroys Map A knowledge).

Our approach:
1. **Save a 95% foundational checkpoint** (~54.45s) before the policy over-specializes.
2. **Beat the WR** on Hockolicious with the active run.
3. **Branch from the 95% checkpoint** and train on a 3–5 map interleaved curriculum to build a generalized driving AI.

See [`GENERALIZATION_STRATEGY.md`](GENERALIZATION_STRATEGY.md) for the full technical breakdown.

---

## 🚀 Quick Start

### Prerequisites
- Windows 10/11 or Windows Server 2022
- [TrackMania Nations Forever](https://www.trackmania.com/) (free)
- [TMInterface v2.1.0](https://donadigo.com/tminterface)
- [TMLoader](https://github.com/donadigo/TMLoader)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Setup
```powershell
# Create conda environment (MUST be Python 3.11 — newer versions break torchrl)
conda create -n linesight python=3.11.16 -y
conda activate linesight

# Install PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# OR for CPU-only:
# pip install torch torchvision

# Install dependencies (torchrl pinned to 0.2.1!)
pip install -r requirements_pip.txt
pip install -e . --no-deps
```

### Train
```powershell
# IMPORTANT: Close TMLoader GUI before running!
python scripts/train.py
```

### Monitor
```powershell
python -c "import joblib; s = joblib.load('save/hocko_v1/accumulated_stats.joblib'); print(f'Frames: {s[\"cumul_number_frames_played\"]:,} | PB: {s[\"alltime_min_ms\"][\"hock\"]/1000:.3f}s')"
```

---

## 📂 Project Structure

```
linesight-trackmania-rl/
├── config_files/           # Training hyperparameters & user configuration
│   ├── config.py           # Main config (copied to config_copy.py at runtime)
│   ├── user_config.py      # Local paths & TMNF credentials
│   ├── inputs_list.py      # Action space definition
│   └── state_normalization.py
├── trackmania_rl/          # Core RL library
│   ├── agents/
│   │   └── iqn.py          # Implicit Quantile Network implementation
│   ├── multiprocess/
│   │   ├── collector_process.py   # Experience collection workers
│   │   └── learner_process.py     # Gradient update loop
│   ├── tmi_interaction/
│   │   └── game_instance_manager.py  # TMInterface socket bridge
│   ├── buffer_utilities.py
│   └── analysis_metrics.py
├── scripts/
│   └── train.py            # Training entry point
├── GENERALIZATION_STRATEGY.md
├── CLOUD_MIGRATION.md
├── PROJECT_STORY.md
└── ATTRIBUTION.md
```

---

## 🙏 Attribution

This project builds on the **[Linesight](https://github.com/Linesight-RL/linesight)** framework by the Linesight-RL team. See [`ATTRIBUTION.md`](ATTRIBUTION.md) for full credits.

---

## 📄 License

This project inherits the license from the upstream [Linesight](https://github.com/Linesight-RL/linesight) repository. See the original project for license details.

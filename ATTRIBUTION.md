# Attribution

This project is built on top of the **[Linesight](https://github.com/Linesight-RL/linesight)** framework, an open-source Reinforcement Learning system for autonomous driving in TrackMania Nations Forever.

## Upstream Project

- **Repository:** [Linesight-RL/linesight](https://github.com/Linesight-RL/linesight)
- **Description:** A distributional RL agent (IQN) that learns to race in TrackMania from raw pixels and telemetry.
- **License:** See the [upstream repository](https://github.com/Linesight-RL/linesight) for license details.

## What I Built On Top

Starting from the upstream Linesight codebase, my contributions include:

1. **Dependency Stabilization**: Pinned `torchrl==0.2.1` and `tensordict==0.2.1` to fix `MemmapTensor` import crashes in newer versions.

2. **Full CPU/GPU Dynamic Compatibility**: Refactored every hardcoded `"cuda"` reference across the entire codebase (`iqn.py`, `learner_process.py`, `collector_process.py`, `buffer_utilities.py`, `analysis_metrics.py`) to dynamically detect hardware with `torch.cuda.is_available()`. This includes TorchScript-compatible device inference, `channels_last` memory format guards, AMP `GradScaler` CPU awareness, and `strict=False` state dict loading for legacy checkpoints.

3. **Self-Healing Auto-Recovery**: Extended `game_instance_manager.py` to automatically force-kill frozen TrackMania processes on TMInterface socket timeouts and relaunch fresh instances, enabling unattended cloud training without manual intervention.

4. **Reward Engineering**: Enabled and tuned the speedslide reward schedule to reward faster cornering on Hockolicious.

5. **Learning Rate Schedule Refinement**: Added a warmup bump at 8.5M frames to escape a local optimum, with safe decay back to baseline.

6. **Cloud Migration**: Ran the training pipeline on an Azure Windows VM, working around TMLoader CDN blocks, headless DirectX rendering and multi-instance coordination. The migration notes are not part of this repository.

7. **10.6M+ Frame Training Run**: Trained the agent on ESL-Hockolicious to a best training-evaluation time of **54.00 s** (`site/data/era1.json`). In the greedy benchmark it finished 5 of 10 runs, best 54.06 s (`eval_results/benchmark_20261014_190540.csv`).

8. **Hybrid self-driving stack (the current project)**: A classical robotics stack (map matching, speed profile, pure pursuit, safety layers), learned models replacing its hand-set numbers, and a residual TD3 policy on top, evaluated on held-out maps. See [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md) and [RESULTS.md](RESULTS.md).

## Thank You

A huge thank you to the Linesight-RL team for creating and open-sourcing this incredible framework. Without their foundational work on the IQN architecture, TMInterface bridge, and training pipeline, this project would not exist.

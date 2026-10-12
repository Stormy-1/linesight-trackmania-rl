# Attribution

This project is built on top of the **[Linesight](https://github.com/Linesight-RL/linesight)** framework — an open-source Reinforcement Learning system for autonomous driving in TrackMania Nations Forever.

## Upstream Project

- **Repository:** [Linesight-RL/linesight](https://github.com/Linesight-RL/linesight)
- **Description:** A distributional RL agent (IQN) that learns to race in TrackMania from raw pixels and telemetry.
- **License:** See the [upstream repository](https://github.com/Linesight-RL/linesight) for license details.

## What I Built On Top

Starting from the upstream Linesight codebase, my contributions include:

1. **Dependency Stabilization** — Pinned `torchrl==0.2.1` and `tensordict==0.2.1` to fix `MemmapTensor` import crashes in newer versions.

2. **Full CPU/GPU Dynamic Compatibility** — Refactored every hardcoded `"cuda"` reference across the entire codebase (`iqn.py`, `learner_process.py`, `collector_process.py`, `buffer_utilities.py`, `analysis_metrics.py`) to dynamically detect hardware with `torch.cuda.is_available()`. This includes TorchScript-compatible device inference, `channels_last` memory format guards, AMP `GradScaler` CPU awareness, and `strict=False` state dict loading for legacy checkpoints.

3. **Self-Healing Auto-Recovery** — Extended `game_instance_manager.py` to automatically force-kill frozen TrackMania processes on TMInterface socket timeouts and relaunch fresh instances, enabling unattended cloud training without manual intervention.

4. **Reward Engineering** — Enabled and tuned the speedslide reward schedule to incentivize optimal cornering technique for the World Record push.

5. **Learning Rate Schedule Refinement** — Added a warmup bump at 8.5M frames to escape a local optimum, with safe decay back to baseline.

6. **Cloud Migration & Documentation** — Full migration playbook for deploying the training pipeline to an Azure Windows VM, including solutions for TMLoader CDN blocks, headless DirectX rendering, and multi-instance coordination.

7. **10.6M+ Frame Training Run** — Trained the agent to a personal best of **54.000s** on ESL-Hockolicious (0.24s from the human World Record of 53.760s).

## Thank You

A huge thank you to the Linesight-RL team for creating and open-sourcing this incredible framework. Without their foundational work on the IQN architecture, TMInterface bridge, and training pipeline, this project would not exist.

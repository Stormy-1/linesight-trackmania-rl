# =======================================================================================================================
# Handle configuration file before running the actual content of train.py
# =======================================================================================================================
"""
Two files named "config.py" and "config_copy.py" coexist in the same folder.

At the beginning of training, parameters are copied from config.py to config_copy.py
During training, config_copy.py will be reloaded at regular time intervals.
config_copy.py is NOT tracked with git, as it is essentially a temporary file.

Training parameters modifications made during training in config_copy.py will be applied on the fly
without losing the existing content of the replay buffer.

The content of config.py may be modified after starting a run: it will have no effect on the ongoing run.
This setup provides the possibility to:
  1) Modify training parameters on the fly
  2) Continue to code, use git, and modify config.py without impacting an ongoing run.
"""

import shutil
from pathlib import Path


def copy_configuration_file():
    base_dir = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        base_dir / "config_files" / "config.py",
        base_dir / "config_files" / "config_copy.py",
    )


if __name__ == "__main__":
    copy_configuration_file()

# =======================================================================================================================
# Actual start of train.py, after copying config.py
# =======================================================================================================================

import ctypes
import joblib
import os
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from art import tprint
from torch.multiprocessing import Lock

from config_files import config_copy
from trackmania_rl.agents.iqn import make_untrained_iqn_network
from trackmania_rl.multiprocess.collector_process import collector_process_fn
from trackmania_rl.multiprocess.learner_process import learner_process_fn
from trackmania_rl.reproducibility import record_run_metadata
from trackmania_rl.utilities import set_random_seed

# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")
random_seed = 444
set_random_seed(random_seed)

def signal_handler(sig, frame):
    print("Received SIGINT signal. Killing all open Trackmania instances.")
    clear_tm_instances()

    for child in mp.active_children():
        child.kill()

    tprint("Bye bye!", font="tarty1")
    sys.exit()


def clear_tm_instances():
    if config_copy.is_linux:
        os.system("pkill -9 TmForever.exe")
    else:
        os.system("taskkill /F /IM TmForever.exe")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    base_dir = Path(__file__).resolve().parents[1]
    save_dir = base_dir / "save" / config_copy.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_base_dir = base_dir / "tensorboard"

    # The isolated generalization run must branch from its saved weights.  Do
    # this before starting collectors so a missing checkpoint cannot result in
    # an apparently healthy run using random weights.
    is_generalization_run = config_copy.run_name in {"gen_v1", "gen_v2"} or config_copy.run_name.startswith("gen_v3")
    metadata_path = record_run_metadata(
        base_dir,
        save_dir,
        config_copy,
        require_checkpoint=is_generalization_run,
        checkpoint_origin=("save/hocko_v1_foundational_checkpoint" if is_generalization_run else None),
    )
    print(f"Run metadata recorded at {metadata_path}")

    clear_tm_instances()

    # Copy Angelscript plugin to TMInterface dir
    shutil.copyfile(
        base_dir / "trackmania_rl" / "tmi_interaction" / "Python_Link.as",
        config_copy.target_python_link_path,
    )

    print("Run:\n\n")
    tprint(config_copy.run_name, font="tarty4")
    print("\n" * 2)
    tprint("Linesight", font="tarty1")
    print("\n" * 2)
    print("Training is starting!")

    if config_copy.is_linux:
        os.system(f"chmod +x {config_copy.linux_launch_game_path}")

    # Prepare multi process utilities
    shared_steps = mp.Value(ctypes.c_int64)
    shared_steps.value = 0
    rollout_queues = [mp.Queue(config_copy.max_rollout_queue_size) for _ in range(config_copy.gpu_collectors_count)]
    shared_network_lock = Lock()
    game_spawning_lock = Lock()
    stop_event = mp.Event()
    _, uncompiled_shared_network = make_untrained_iqn_network(jit=config_copy.use_jit, is_inference=False)
    uncompiled_shared_network.share_memory()

    # Seed the collector-visible network and schedule counter before workers
    # start.  Otherwise a collector can copy the still-random shared network
    # during its first rollout, before the learner has loaded the checkpoint.
    if is_generalization_run:
        from trackmania_rl import utilities

        utilities.load_model_checkpoint(
            uncompiled_shared_network,
            save_dir / "weights1.torch",
            map_location="cpu",
            allowed_missing_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
            allowed_unexpected_keys=utilities.NON_PERSISTENT_NORMALIZATION_KEYS,
        )
        shared_steps.value = joblib.load(save_dir / "accumulated_stats.joblib")["cumul_number_frames_played"]

    # Start worker process
    collector_processes = [
        mp.Process(
            target=collector_process_fn,
            args=(
                rollout_queue,
                uncompiled_shared_network,
                shared_network_lock,
                game_spawning_lock,
                shared_steps,
                base_dir,
                save_dir,
                config_copy.base_tmi_port + process_number,
                process_number,
                stop_event,
            ),
        )
        for rollout_queue, process_number in zip(rollout_queues, range(config_copy.gpu_collectors_count))
    ]
    for collector_process in collector_processes:
        collector_process.start()

    # Start learner process
    learner_process_fn(
        rollout_queues,
        uncompiled_shared_network,
        shared_network_lock,
        shared_steps,
        base_dir,
        save_dir,
        tensorboard_base_dir,
        stop_event,
    )  # Turn main process into learner process instead of starting a new one, this saves 1 CUDA context

    for collector_process in collector_processes:
        collector_process.join()

# Linesight TrackMania RL: Repository and Code Review

Review of Stormy-1/linesight-trackmania-rl against upstream Linesight-RL/linesight, prepared October 13, 2026.

## Scope and limits

This review covers the README, ATTRIBUTION.md, the full diff of config.py and the trackmania\_rl package against upstream main, the commit history of user\_config.py, and the launch\_game, is\_game\_running and close\_game functions. It does not cover the rest of game\_instance\_manager.py, the three long markdown files (CLOUD\_MIGRATION, GENERALIZATION\_STRATEGY, PROJECT\_STORY), the training logs, or the best replay. Anything said about the training result is therefore a statement about what the repository claims and shows, not about the run itself.

## Summary

The attribution is honest and the operations work is real: a device-agnostic port, automatic recovery from frozen game instances, and a long CPU-only run on a student cloud budget. The weakness is the gap between that work and how the repository presents it. The headline claims a world record agent that is faster than any human, while the results table shows it 240 ms short of the record. The code delta against upstream is small, the central result cannot be verified from the repository, and a fresh clone may not run. The worry about leaked credentials turned out to be unfounded.

## How large the contribution is

Against upstream main the code changes touch nine files with 62 insertions and 32 deletions, alongside roughly 750 added lines of markdown. In config.py the real changes are two: the speedslide reward went from 0 to 0.003, and the learning rate schedule gained a jump to 1e-4 at 8.5M frames. The prio\_alpha line has the same value as upstream (0); only its comment changed, so prioritized replay was not disabled by you, it was never on. The rest of the code diff is device plumbing plus one recovery block.

That is a legitimate body of ops and porting work, and ATTRIBUTION.md describes it fairly. It does not support words like tuned or refined for the reward and learning rate changes, because each is a one-line edit with no evidence yet of its effect. The honest framing is that you took a research framework, made it run unattended on cheap hardware, and ran a long single-map campaign.

## Headline claims versus the repository

The title says World Record Agent and the tagline says faster than any human, yet the results table puts the agent 240 ms behind the human record. Neither claim holds until a verified time beats 53.760 s.

The statements about no hardcoded racing lines and learning from pixels and telemetry only are overstated. The float inputs include a preview of 40 upcoming zone centers spaced 10 m apart, which is 400 m of the reference line, and the reward is progress along that line. The policy does not copy actions from a replay, but its observations and reward are built on a precomputed line. Whether the cl2 line is a plain centerline or derived from a human replay decides how strong this objection is.

Several README numbers disagree with config.py. The discount is described as annealed from 0.9 to 1.0 over 2.5M frames, while the config holds 0.999 until 1.5M and reaches 1.0 at 2.5M. The learning rate is described as cosine-like from 5e-5 to 1e-5 with a warmup bump, while the config starts at 1e-3 and is piecewise linear, and the bump is a ramp to 1e-4 over about 10k frames, held until 10M, which is a learning rate kick rather than a warmup. The buffer sizes of 50k, 100k and 200k look like upstream defaults rather than your change, and Boltzmann exploration is not mentioned at all. Because config\_copy.py is what actually runs, is hot-reloaded and is untracked, the committed config.py records intent rather than the run. Save a timestamped snapshot of the live config together with the frame counts at which it changed.

Some terminology needs fixing. Entropy collapse is a policy gradient concept; IQN with epsilon-greedy and Boltzmann exploration has no policy entropy to collapse. The real risks are overfitting to one map, loss of plasticity and a replay buffer dominated by one map. A discount of 1.0 is undiscounted episodic return, not infinite horizon.

## Reproducibility and verifiability

The diff statistics show maps/ESL-Hockolicious\_0.5m\_cl2.npy going from 193,616 bytes upstream to 0 bytes in your HEAD, which means it is absent from the committed tree even though your map\_cycle points at it. It may be excluded by the new .gitignore, so check with git ls-files. If it is missing, a clone cannot train, and you should commit it or document how to generate it.

Nothing in the repository proves the 54.000 s time. TrackMania replays are deterministic, so committing the best Replay.Gbx lets anyone verify the run. Link a source for the 53.760 s record as well, and say what purist means.

The committed user\_config.py carries laptop-specific paths under OneDrive, so anyone cloning it inherits paths that will not exist on their machine. A cleaner pattern is a user\_config.example.py in the repository with the real file ignored.

Add a LICENSE file with the upstream license text, since pointing to the upstream repository is not a license grant. Because your history has no merge base with upstream, the lineage is visible only through the README. Consider re-creating the repository as a real fork, or at least noting the upstream commit you started from.

## Code review

### Device handling

The direction is right, and a few details deserve attention. DEVICE is defined in three modules and local device variables appear elsewhere, so centralize it in one place. Disabling channels\_last on CPU may cost speed, since CPU convolutions often benefit from it with oneDNN, and CPU throughput is your bottleneck; benchmark it rather than assume. Disabling fp16 autocast on CPU means the run changed numerics when it moved from laptop to cloud, which confounds any story about what the later learning rate change did. Finally, ATTRIBUTION.md says every hardcoded cuda reference was replaced, and a search for the word cuda across all Python files will confirm that.

### Checkpoint loading with strict=False

The float normalization tensors became non-persistent buffers, so the state dict keys did not change and strict=False was not needed for that refactor. What strict=False does is let a mismatched or partial checkpoint load without any error. Combined with the inherited try and except that only prints on failure, a corrupted checkpoint after a crash could resume training with partly random layers, at a low exploration rate and a late learning rate schedule position, and nothing would flag it. Log the missing and unexpected keys and raise if either is non-empty. The optimizer loads kept the default strictness while the network loads were relaxed, which is inconsistent.

### Automatic recovery in game\_instance\_manager

The primary path calls close\_game, which issues taskkill on the tracked game process and then waits until the process is gone, so clearing the process id afterwards is sound there. The pid is that of TmForever.exe itself, located through Get-CimInstance, so killing a process tree is not needed.

The fallback branch, however, is close to unreachable and repeats the same mechanism. close\_game only asserts that the pid is set, os.system does not raise on failure, and the fallback runs the same taskkill command again. If it ever ran with an unset pid it would issue taskkill on the string None. The extra resets of the timeout and activation flags duplicate what close\_game already does.

The bigger issue is inherited: close\_game waits in a loop that calls sleep(0) and has no deadline. On a four vCPU machine running two game instances and a learner, that spin steals real compute, and if a process cannot be killed (for example access denied) the loop never exits and the collector hangs, which defeats the purpose of unattended recovery. Use a short sleep and a deadline, then escalate or raise so a supervisor can restart the collector.

Three points could not be settled from the excerpts. The launch\_game excerpt was truncated, so I cannot tell whether latest\_map\_path\_requested is reset on relaunch; if it is not, a relaunched game may never be told to load the map. Setting iface to None is safe only if no later code dereferences it before it is recreated. And removing ensure\_not\_minimized may hide a cause of freezes if they come from a minimized or disconnected remote desktop window, in which case you get a kill and relaunch loop instead of a fix. A counter of recoveries logged to TensorBoard would settle whether unattended operation actually works, and it is the evidence ATTRIBUTION.md currently lacks. Also consider whether a tmi\_protection\_timeout\_s of 500 seconds is acceptable, since each freeze can cost several minutes of collector time.

## Results and methodology

The result is a single run with no baseline and no learning curve in the repository. The question that matters is what unmodified upstream Linesight achieves on this map with its default config, because without that the speedslide reward and the learning rate kick cannot be credited with anything. The claim that the 8.5M bump escaped a local optimum is causal and needs a personal best versus frames plot with the config changes marked on it. The constant speedslide reward from frame zero is also worth an ablation, or a decay to zero late in training, since it can compete with lap time at the end.

The throughput figures do not reconcile. With 5 engine steps of 10 ms per action, each action is 50 ms. If a frame in the accounting is one action, 10.65M frames is about 148 hours of game time, which at a true 80 times speed would take under two hours per instance, not 15. So 80 times is the configured speed rather than measured throughput, which is plausible on this hardware. Report measured frames per second and state whether the 15 hours is wall-clock or game time.

The 50 ms action granularity also matters for the last stretch. The remaining 240 ms is about five actions, so the gap may be limited by action resolution as much as by exploration. Boltzmann exploration is a sensible tool for fine tuning, but a plateau here is still likely, and the project should not depend on beating the record.

## Generalization plan

Interleaved multi-map training is already supported by upstream through the map cycle, and the commented lines in your config show the pattern, so part of the plan is a configuration choice rather than new work. The valuable experiment is a controlled one: train on a few maps, hold out at least one, compare against a single-map baseline from the same checkpoint, and report the held-out gap. A saved 95 percent checkpoint is a sensible starting point, but the claim that it prevents forgetting is a hypothesis until measured.

## Resolved questions

The credentials worry is resolved. The history of user\_config.py contains a single commit of yours, which changed the username to stormy and two paths to OneDrive locations. At no point does the file contain a password or key, and the complete current file shows only a username, paths and a port. The README line saying the file holds TMNF credentials is therefore inaccurate and should say it holds a username and local paths. The account name is now in public history, which is low risk. One caveat is that the search only matched lines containing pass, user or key, but since the whole file was inspected the conclusion stands.

## Suggested priorities

Start with the claims, because they cost nothing: retitle the repository, fix the tagline, reconcile the README numbers with the config, and correct the credentials wording. Next make the result verifiable by committing the replay, the map file and a snapshot of the live config, and by adding the license. Then harden the code by logging key mismatches on checkpoint load, adding a deadline to the kill wait, removing the redundant fallback and counting recoveries. Finally strengthen the evidence with an upstream baseline, a progress versus frames plot with changes annotated, measured throughput, and one held-out map experiment. If the project is a portfolio piece, the strongest framing is the one already in ATTRIBUTION.md, stated at the top of the README.

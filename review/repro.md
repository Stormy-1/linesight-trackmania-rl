# Reproducibility check

Question: can someone regenerate every table in RESULTS.md from a clean clone, using only committed files?

Method: every command was run in a fresh git worktree of the branch (committed files only, no ignored files), with
the project's Python 3.11 environment. Nothing here needed the game.

## What regenerates from committed files

| RESULTS.md section | Command | Result in a clean checkout |
|---|---|---|
| §0 robustness table and pairs | `python scripts/report_robustness.py --tag v9_rl_rob --compare v7_rob` (also v8_rl_rob) | matches: 16 / 9 / 11 / 64, p = 0.824 |
| §0 map-level tests, bootstrap, power | `python scripts/reanalyze_clustered.py` (new, seed 20261117) | matches `review/clustered_reanalysis.json` |
| §1 single-run history | `python scripts/report_test_set.py --tag v4` … `--tag v8_rl` | matches all five aggregates |
| §2 dev single runs | `python scripts/report_test_set.py --set dev --tag v7a` and `--tag td3v2` | matches: 6/7, 93.1% and 93.7%, 97 and 196 recovery |
| §2b dev robustness | `python scripts/report_robustness.py --set dev --tag td3v3_rob --compare v7a_rob` | matches: 57 / 2 / 3 / 8, p = 1 |
| §3 rl_train | `python scripts/report_test_set.py --set rl_train --tag v7zero` and `--tag td3v2` | matches: 2/6, 69.5%, 574 and 3/6, 63.3%, 159 |
| §3 behaviour analysis | `python scripts/analyze_residual_policy.py --checkpoint eval_results/residual_rl/td3_v3/final.pt --runs "*_dev_td3v2" "*_rl_train_td3v2"` | runs; matches `td3_v3_behaviour.json` |
| §4 campaign | `python scripts/run_campaign_batch.py --report` | matches: 27 of 63 |

## Gaps

| Gap | Affects | Status |
|---|---|---|
| The completion bootstrap interval (−1.6 to +20.0) had no committed script and no seed | README, RESULTS §0, site | **Fixed.** `scripts/reanalyze_clustered.py` computes it with a fixed seed: −1.5 to +20.0. The docs now quote that value. |
| §3's td3_v3 training figures (C05 2/6 → 6/6, buffer 155k → 330k) have no command | RESULTS §3 | Open, but the source is committed: `eval_results/residual_rl/td3_v3/train_log.jsonl` (fields `map`, `episode`, `completion_pct`, `buffer_size`). |
| §5 era-1 benchmark has no command | RESULTS §5 | Open, but the source is committed: `eval_results/benchmark_20261014_190540.csv` (count `finished` by `checkpoint` and `track`). `scripts/build_site_data.py` reads it for the site. |
| §6 model validations can be read but not regenerated | RESULTS §6, ARCHITECTURE §6 to §8 | Open, by necessity. Rebuilding needs the game's official replays (`*.Replay.gbx`, game files, not redistributable) and the official-replay centerlines built from them (`maps/*_official_0.5m_cl.npy`, gitignored). `scripts/fit_grip_model.py` fails in a clean checkout on the first missing centerline. The validation JSONs, the trained models and the per-step live logs are committed. |
| The speed-model dataset (`eval_results/speed_model/dataset.npz`) is gitignored | ARCHITECTURE §6 "139,063 points" | Open. The count can't be checked from the repo. It is rebuilt by `scripts/build_speed_dataset.py` from the game's replays. |
| The era-1 learning curve comes from untracked TensorBoard logs (`tensorboard/hocko_v1*`) | site era-1 chart, ATTRIBUTION item 7 | Open. `site/data/era1.json` holds the extracted curve; the raw event files are not in the repo. |
| `trackmania_rl/hybrid_block_map.py` hard-coded the Steam install folder, and `scripts/build_speed_dataset.py`, `measure_gravity.py` and `build_range_sensor.py` used it with no override | rebuilding §6 on another machine | **Fixed.** `DEFAULT_GAME_DIR` now reads the `TMNF_GAME_DIR` environment variable first (README "Running it"). `build_official_centerlines.py` also takes `--game-dir`. |
| Hard-coded `C:\Users\chopi\...` paths in `scripts/tools/gbx_to_times_list.py`, `scripts/tools/video_stuff/a_v_widget_folder.py`, `trackmania_rl/run_to_video.py` | none of the results | Left alone: upstream Linesight tools, not used by this project's results. |
| Per-run `summary.json` files store model paths with Windows separators (`eval_results\\speed_model\\...`) | none of the reports | Left alone: the report scripts don't read those fields. They are a record of what was run. |
| 4 of 382 tests read gitignored centerlines in `maps/` (`test_hybrid_live.py::test_replays_real_a01_capture_without_losing_track`, `test_hybrid_predictive_features.py`, `test_hybrid_real_validation_readiness.py`, `test_hybrid_shadow_observer.py`) | the test suite in a clean clone: 375 pass, 4 fail, 3 skip | Open. They pass in a checkout that has the centerlines. They should skip when the file is missing, but that changes test code outside this review. |
| The pre-registered td3_v3 selection rule exists only in the docs | RESULTS §2b | Open. It was written in the gitignored project notes before the dev runs, so there is no committed timestamp proving the order. The rule and the dev data are both public, so anyone can check the rule was applied as written. |

## Seeds

| What | Seed | Where |
|---|---|---|
| Map split | 20261112 | `maps/map_split.json`, `scripts/make_map_split.py` |
| Evaluation steering noise | 0 | `--perturb-seed 0`, recorded in every run's `summary.json` |
| td3_v3 training noise | 1000 | `eval_results/residual_rl/td3_v3/train_config.json` |
| td3_v3 extra training maps | 20261113 | same file, `pool_seed` |
| TD3 learner, critic burn-in | 0 | `train_config.json`, `critic_burnin_v2.json` |
| Speed-model folds and training | `SEED` in `scripts/train_speed_model.py` | same |
| Map-level reanalysis | 20261117 | `scripts/reanalyze_clustered.py` |

Live runs are deterministic without noise and seeded with it, so the logged runs are the primary data. Re-driving a
map needs the game, TMInterface and the official-replay centerlines.

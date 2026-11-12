# Results

Every number below comes from a file in `eval_results/`. The source is named under each table, and the last section
shows how to regenerate the tables. How the components work is in [ARCHITECTURE.md](ARCHITECTURE.md).

Each run is one deterministic live drive of TrackMania Nations Forever at 10× game speed. A run ends at the finish,
after being stuck for 6 s, or at the time limit.

## 1. Held-out test set (the headline)

The test maps are 10 campaign maps drawn once from the White, Green, Blue and Red tiers (`maps/map_split.json`).
Nothing was tuned on them, and every map gets identical arguments.

| Version | What it added | Finished | Mean completion | Median time vs pro | Landings / bumps / walls / scrapes | Recovery steps |
|---|---|---|---|---|---|---|
| v4 | classical stack | 3/10 | 57.9% | 1.60× | 12 / 0 / 2 / 1 | 529 |
| v5 | + learned speed model | 3/10 | 52.3% | 1.63× | 14 / 0 / 11 / 4 | 734 |
| v6 | + learned grip limit | 2/10 | 54.0% | 1.50× | 19 / 0 / 2 / 0 | 677 |
| v7 | + measured gravity | 2/10 | 46.2% | 1.44× | 18 / 0 / 1 / 1 | 747 |
| **v8_rl** | **+ residual TD3** | **4/10** | **58.1%** | 1.50× | 17 / 0 / 1 / 1 | **431** |

Per map: finish time, or completion % when the car got stuck.

| Map | Tier | Pro replay | v4 | v5 | v6 | v7 | v8_rl |
|---|---|---|---|---|---|---|---|
| A09-Race | White | 25.98 s | 41.51 s | 42.7% | 66.2% | 69.2% | **39.56 s** |
| A15-Speed | White | 25.33 s | 41.39 s | 41.21 s | 40.11 s | **38.20 s** | 38.39 s |
| B04-Acrobatic | Green | 13.02 s | 54.2% | 54.9% | 54.2% | 54.2% | 54.2% |
| B07-Race | Green | 30.36 s | 44.97 s | 51.76 s | 42.93 s | 12.8% | **41.82 s** |
| B13-Obstacle | Green | 25.65 s | 8.0% | 8.0% | 8.0% | 8.3% | 8.2% |
| C02-Race | Blue | 42.47 s | 94.6% | 63.63 s | 94.8% | **57.60 s** | 62.37 s |
| C10-Acrobatic | Blue | 15.03 s | 49.6% | 45.0% | 45.0% | 45.0% | 46.1% |
| C13-Race | Blue | 44.87 s | 7.9% | 7.9% | 7.9% | 7.9% | 7.9% |
| D04-Race | Red | 52.86 s | 9.5% | 8.6% | 9.5% | 9.8% | 9.8% |
| D07-Race | Red | 60.37 s | 54.8% | 56.4% | 55.0% | 55.0% | 55.0% |

Source: `eval_results/test_set_v4_baseline.json`, `test_set_v5_speed_model.json`, `test_set_v6_grip_model.json`,
`test_set_v7_gravity.json` and `test_set_v8_rl_td3v2.json`. Per-step logs: `eval_results/hybrid_live/<map>_test_<tag>/`.

**Reading it honestly**

- **v5.** The speed model predicts pro speed far better than the hand rule, but on its own it made the car run
  wide: 11 wall hits. Pros drift through corners at speeds this car can't hold.
- **v6.** The grip cap fixed that: walls fell from 11 to 2, scrapes from 4 to 0, and the finished maps got faster.
  The two finishes it lost were airborne and landing events, which the grip model excludes by design.
- **v7.** Measured gravity made every surviving run faster (best A15 and C02), but the finish count didn't move.
- **The finished set churns from v4 to v7.** Each change flips one or two maps through landing events, where the car
  comes down a few metres off the line. The bottleneck became tracking precision around jumps and landings.
- **v8_rl.** The residual policy tipped both churning maps, A09 and B07, to the finish side, with best-ever times on
  both. It kept both v7 finishes, and recovery steps fell 42%. With 10 maps and one run each, this is a two-map
  swing: evidence, not proof. The median time ratio rises from 1.44 to 1.50 because two harder finishes enter the median.
- **Untouched failures.** B13, C13 and D04 stall in the first 10% in every version, and B04, C10 and D07 stall at the
  same spot every time. They look like line or geometry problems for the classical layer, not small-correction problems.

## 2. Dev maps: base vs base + RL

The dev maps are used for every design choice. The policy was never trained on them.

| Map | Pro replay | v7a (base) | td3v2 (base + RL) |
|---|---|---|---|
| A01-Race | 24.54 s | 42.76 s, 1 wall | 42.98 s, 0 walls |
| A02-Race | 16.25 s | 19.08 s | **18.62 s** |
| A03-Race | 18.75 s | 25.86 s | **25.75 s** |
| A05-Race | 16.91 s | 20.32 s | **19.55 s** |
| A07-Race | 29.14 s | 37.98 s | 38.21 s |
| A11-Race | 19.32 s | 24.80 s, 1 scrape | **24.42 s**, 0 scrapes |
| B09-Acrobatic | 13.99 s | 51.8% | 56.0% |
| **Set** | | 6/7, 93.1%, ratio 1.29, 1 wall, 97 recovery | 6/7, 93.7%, ratio 1.29, 0 walls, 196 recovery |

The policy was faster on 4 of the 6 finished maps and slower by about 0.2 s on 2. The six finish times sum to
170.80 s for the base and 169.53 s with RL. The cost is more recovery steps, on A01 and B09.
Source: `eval_results/hybrid_live/<map>_dev_v7a/` and `<map>_dev_td3v2/`.

## 3. Training maps: base vs base + RL

These are the 6 rl_train maps, driven by the deterministic policy with no exploration noise.

| Map | Pro replay | Base (v7, zero residual) | Base + RL (td3v2) |
|---|---|---|---|
| A06-Obstacle | 28.58 s | 98.5% | 15.8% |
| A08-Endurance | 64.48 s | 117.30 s | **116.47 s** |
| B11-Race | 31.44 s | 100.0%, stuck: the line ends before the finish | 54.7% |
| B12-Race | 45.50 s | 69.84 s | **68.00 s** |
| C05-Endurance | 116.39 s | 9.1% | **169.35 s** |
| D05-Race | 71.43 s | 9.2% | 9.2% |
| **Set** | | 2/6, 69.5%, 574 recovery | 3/6, 63.3%, 159 recovery |

Even on its own training maps the policy isn't uniformly better. On A06 and B11 it lowered the speed into a jump
that the base stack only just cleared. Single runs at knife-edge jumps flip on sub-m/s changes, which is why the
held-out results are reported as evidence, not proof.
Source: `eval_results/hybrid_live/<map>_rl_train_v7zero/` and `<map>_rl_train_td3v2/`.
Training logs: `eval_results/residual_rl/td3_v1/train_log.jsonl` and `td3_v2/train_log.jsonl`.

## 4. Full Nations campaign: the classical stack's generalisation

This is the v3b classical stack (before any learned component) on official-replay lines, one run per map.
It had been developed on 5 of these maps.

| Tier | Finished |
|---|---|
| White A01–A15 | 12 / 15 |
| Green B01–B15 | 9 / 15 |
| Blue C01–C15 | 4 / 15 |
| Red D01–D15 | 2 / 15 |
| Black E01–E03 | 0 / 3 |
| **Total** | **27 / 63** |

Source: `eval_results/hybrid_live/*_official_v3b/`. Table: `python scripts/run_campaign_batch.py --report`.

## 5. The pure-RL baseline

This is the original Linesight IQN agent, trained from pixels on Hockolicious, evaluated greedily, 10 runs per map
(`eval_results/benchmark_20261014_190540.csv`):

| Checkpoint | Hockolicious (trained) | A01 | A03 | A05 | A07 | A11 |
|---|---|---|---|---|---|---|
| 3.18M frames | 4/10 finished (best 54.70 s) | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| 10.64M frames | 5/10 finished (best 54.06 s) | 0/10 | 8/10 | 0/10 | 0/10 | 0/10 |

Across the five other maps, the agent finished 0 of 50 runs at 3.18M frames and 8 of 50 at 10.64M frames, all of them on A03.
The hybrid stack finished all five of those maps (§2 and `eval_results/hybrid_live/*_official_v3b/`).

## 6. Learned-model validation

The details are in ARCHITECTURE.md §6–8. Every split is by map.

| Model | Validation | Result | Baseline | Source |
|---|---|---|---|---|
| Speed model (MLP) | 5-fold CV by map | r 0.81, MAE 11.7 m/s | hand rule r 0.33, MAE 34.1 | `speed_model/validation_v1.json` |
| Speed model | test maps, scored once | r 0.84, MAE 10.3 m/s | hand rule r 0.21, MAE 31.1 | same |
| Grip limit (logistic) | leave-one-map-out, 54 maps | AUC 0.754 | speed alone 0.708 | `grip_model/grip_model_v1.json` |
| Grip limit | test maps, scored once | AUC 0.84 | | same |
| Virtual lidar | leave-one-map-out, 43 wall hits | contact read at 56% of hits vs 10% of ordinary steps | | `range_sensor/validation_v1.json` |
| Gravity | 2,328 free-fall samples, 53 maps | 24.87 m/s² (IQR 24.55–25.25) | old guess 15.4 | `gravity_measurement.json` |
| Critic burn-in | Q vs true return, 6 maps | r 0.89–1.00 | | `residual_rl/critic_burnin_v2.json` |

## Regenerating the tables

```bash
python scripts/report_test_set.py --tag v8_rl            # test set; also v4, v5, v6, v7
python scripts/report_test_set.py --set dev --tag td3v2  # dev; base is --tag v7a
python scripts/report_test_set.py --set rl_train --tag td3v2   # base is --tag v7zero
python scripts/run_campaign_batch.py --report
```

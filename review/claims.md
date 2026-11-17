# Claim ledger

Every quantitative or causal claim in README.md, RESULTS.md, ARCHITECTURE.md, ATTRIBUTION.md and the site
(`site/index.html`, `site/js/*`, `site/data/*.json`), as they stood at commit fff0802. Each number was recomputed
from the raw files where possible: per-run `summary.json` and `runXX.csv` in `eval_results/hybrid_live/`, the
model JSONs, the training logs and the benchmark CSV. "Recomputed" names the file that was read.

Verdicts: SUPPORTED, OVERSTATED (the number is right but the wording claims more than it shows), UNSUPPORTED
(no file in the repo backs it), WRONG, INCONSISTENT (files disagree). Line numbers refer to fff0802.

## Summary

| Verdict | Count |
|---|---|
| SUPPORTED | 45 |
| OVERSTATED | 19 |
| INCONSISTENT | 5 |
| UNSUPPORTED | 4 |
| WRONG | 5 |
| **Total** | **78** |

The numbers themselves hold up: no figure in a results table is wrong. The problems are wording and selection.
The WRONG items are three distinct errors. The claim that two "separately trained" policies replicate the B07/C02
trade appears in three files (H11 to H13); td3_v3 was initialised from td3_v2 and trained on a buffer holding
td3_v2's episodes. The site says "every claim here is measured on 10 held-out maps" (I3). RESULTS.md says v6 lost
"two finishes" where it lost one against v5 (T4). The most visible OVERSTATED item is the site hero stat
"25 → 27/100" without a p-value (H2).

Status on branch `review/claims`: every OVERSTATED, WRONG and INCONSISTENT item is fixed in the files it appears in.
Of the UNSUPPORTED items, E3 and E5 are reworded to what the repo shows. M2 (the dataset size) and E7 (the
fine-tuning collapse) are left as stated: the evidence exists outside the repo but isn't committed (see repro.md).

## Evaluation headline (test-set robustness runs)

| ID | Where | Wording | Recomputed | Verdict |
|---|---|---|---|---|
| H1 | README:65, RESULTS:38, site:48, site:344, ARCH:203 | 27/100 vs 25/100 runs finished | per-run summaries of `*_test_v9_rl_rob`, `*_test_v7_rob`: 27 and 25 | SUPPORTED as a number |
| H2 | site:48 | hero stat "Held-out runs finished, base vs + residual RL: 25 → 27/100", no p-value | p = 0.82 (pairs 9 vs 11); map-level sign-flip p = 1.0 (`review/clustered_reanalysis.json`) | OVERSTATED: reads as a gain |
| H3 | README:63-70 | "What it achieved" lists 27/100, 54.7%, −23% recovery, then the p-values at the end | same numbers; p = 0.82, CI includes 0 | OVERSTATED: gains first, null last |
| H4 | README:92-93, RESULTS:43-44, site:355 | pairs 16 / 9 / 11 / 64, exact sign test p = 0.82 | `eval_results/test_set_robustness_v7_v9rl.json`; recomputed from summaries | SUPPORTED |
| H5 | RESULTS:44 | v8_rl vs v7: 9 / 10 (p = 1) | `test_set_robustness_v7_v8rl.json` | SUPPORTED |
| H6 | README:66, RESULTS:38, site:345 | mean completion 48.0 → 54.7% | per-run completion means | SUPPORTED |
| H7 | README:69-70, RESULTS:49-50, site:345 | bootstrap over maps 95% CI −1.6 to +20.0 points | `scripts/reanalyze_clustered.py` (seed 20261117): −1.5 to +20.0. No committed script produced −1.6 | SUPPORTED (seed-dependent, see repro.md) |
| H8 | README:67, RESULTS:40, RESULTS:50, site:346-347 | recovery 6,543 → 5,043 (−23%); walls 41 → 32 (−22%) | robustness JSONs; 5043/6543 = 0.771, 32/41 = 0.780 | SUPPORTED |
| H9 | README:68 | "It reliably fixes B07 (9/10 vs 2/10, p = 0.016) but reliably breaks C02" | per-map discordant 0/7 and 7/0, p = 0.016 each uncorrected, ≈ 0.16 after ×10 | OVERSTATED: "reliably" with an uncorrected p |
| H10 | README:94-95 | "RL is consistently and significantly better on B07 ... and worse on C02" | same; the next sentence says neither survives correction | OVERSTATED and self-contradicting |
| H11 | README:95 | "Both policies show the same trade, so it is learned behaviour, not luck." | td3_v3 `train_config.json`: `init` = td3_v2/final.pt, `buffer_from` includes td3_v2/episodes | WRONG: not independent replications |
| H12 | RESULTS:47 | "The same direction in two separately trained policies makes chance an unlikely explanation." | same as H11 | WRONG |
| H13 | site:353-354 | "A real, learned trade ... td3_v2, trained separately, makes the same trade." | same as H11 | WRONG |
| H14 | RESULTS:28-37 | per-map finishes and completions for v7 / v8_rl / v9_rl, A09 pairs 2 / 4 | recomputed per map; all match | SUPPORTED |
| H15 | RESULTS:51-52 | "Lap times barely change ... A15's median difference is −0.07 s for td3_v2." | paired A15 median: td3_v2 −0.07 s (10 pairs), td3_v3 **+0.68 s** (10 pairs) | OVERSTATED: quotes td3_v2, the final policy is slower |
| H16 | RESULTS:54-57 | "That sensitivity explains both the single-run 'gain' and the churn" | consistent with the data, not tested as a cause | OVERSTATED (minor) |
| H17 | README:115-116 | "The same luck explains the churn from v4 to v7." | same as H16 | OVERSTATED (minor) |
| H18 | README:24-25 | the paired evaluation "showed which differences are real and which are luck" | 3 of 10 maps carry every discordant pair; effective sample ≈ 14 runs; the map-level test cannot reach p < 0.05 (`review/stats.md`) | OVERSTATED: it showed the single-run gain was luck, it cannot show a residual effect is real |
| H19 | RESULTS:21-23 | noise check on A07: 3/3 finished, 37.97 to 38.24 s vs 37.98 s | `A07-Race_dev_noise005_smoke/summary.json`: 38.24, 38.17, 37.97; `A07-Race_dev_v7a` 37.98 | SUPPORTED |
| H20 | README:99-100, RESULTS:24, RESULTS:135-150, site:359 | td3_v3 got its test look by a rule fixed before any dev result | the rule is recorded in the project notes before the dev runs; the dev JSONs show 60 > 59 and 97.4% > 94.4% | SUPPORTED (pre-registration is not timestamped in a committed file) |
| H21 | README:187-188, RESULTS:57, site:371 | 4 maps finish at least sometimes, 6 never finish in any version | robustness JSONs: B04 B13 C10 C13 D04 D07 0/10 in v7, v8_rl, v9_rl | SUPPORTED |
| H22 | README:191, RESULTS:111, site:371 | B13, C13, D04 stall in the first 10% in every version | completion 8.3/8.2/8.3, 7.9, 9.9/9.9/9.8 (robust); single runs 7.9 to 9.8 | SUPPORTED |

## Dev and training maps

| ID | Where | Wording | Recomputed | Verdict |
|---|---|---|---|---|
| D1 | README:71 | dev: 60/70 vs 59/70, completion 97.4% vs 95.0% | `dev_robustness_td3v3_rob_vs_v7a.json` | OVERSTATED: omits p = 1 and that walls rose 6 → 20 and recovery 1,395 → 1,704 |
| D2 | RESULTS:139-151 | dev table and "td3_v3 hits walls more often on dev (20 vs 6)" | same JSON | SUPPORTED |
| D3 | RESULTS:118-130 | dev single runs; sums 170.80 s and 169.53 s | per-run summaries `*_dev_v7a`, `*_dev_td3v2` | SUPPORTED |
| D4 | RESULTS:158-166 | rl_train 2/6, 69.5%, 574 vs 3/6, 63.3%, 159 | `report_test_set.py --set rl_train` for v7zero and td3v2 | SUPPORTED |
| D5 | README:72-73, site:281 | td3_v2 finished C05; td3_v3 finished it in its last 6 training episodes | `C05-Endurance_rl_train_td3v2` 169.35 s; td3_v3 train_log episodes 133 to 195 all 100% | SUPPORTED (training map) |
| D6 | RESULTS:179 | C05 2/6 in the first 100 episodes, 6/6 in the last 100 | train_log: episodes 13, 14, 15, 73, 74, 75 → 2 finishes; 133 to 195 → 6 | SUPPORTED |
| D7 | RESULTS:175, ARCH:190 | buffer 155k growing to about 330k | train_log `buffer_size` 155,107 → 330,769 | SUPPORTED |
| D8 | ARCH:217 | a transformer "would be oversized for 39 inputs and 155k transitions" | the final buffer is 330k | INCONSISTENT (minor) |

## Single-run test history

| ID | Where | Wording | Recomputed | Verdict |
|---|---|---|---|---|
| T1 | README:109-113, RESULTS:69-73 | v4 to v8_rl aggregates | `test_set_v{4..8}*.json` aggregates match every cell | SUPPORTED |
| T2 | README:115 | "42% fewer recovery steps" (v8_rl single run) | 431 vs 747 = −42% | SUPPORTED |
| T3 | RESULTS:97 | v6: walls 11 → 2, scrapes 4 → 0, finished maps faster | v5 vs v6 aggregates; A15 41.21 → 40.11 s, B07 51.76 → 42.93 s | SUPPORTED |
| T4 | RESULTS:98 | "The two finishes it lost were airborne and landing events" | v5 finished A15, B07, C02; v6 finished A15, B07. Against v5 it lost one map (C02). A09 was lost against v4 | WRONG as worded |
| T5 | RESULTS:99 | v7 made every surviving run faster | A15 40.11 → 38.20 s is the only map finished in both v6 and v7 | SUPPORTED (one map) |
| T6 | README:194, RESULTS (implicit), site:372 | finished maps take 1.4 to 1.6× the pro's time | single-run median ratios 1.44 to 1.63; robust v7 per-map medians 1.40 to 1.58 | SUPPORTED |
| T7 | site:242-243 | single-run finish count stayed at 2 to 3 of 10 | v4 to v7: 3, 3, 2, 2 | SUPPORTED |

## Test-set integrity

| ID | Where | Wording | Recomputed | Verdict |
|---|---|---|---|---|
| I1 | README:5 | "a held-out set of 10 maps that no part of the system was tuned on" | RESULTS:105-108: test failure logs led to the gravity fix | OVERSTATED |
| I2 | RESULTS:10 | "Nothing was tuned on them" | same; the caveat is 95 lines further down | OVERSTATED (minor) |
| I3 | site:43-45 | "Every claim here is measured on 10 held-out maps." | the page also shows the 63-map campaign, dev-map results, era-1 maps and training curves | WRONG |
| I4 | RESULTS:108-110 | the test set was looked at "five times" plus "a sixth look" | looks: v4, v5, v6, v7, v8_rl, robustness v7 + v8_rl, robustness v9_rl = 7 | INCONSISTENT: v9_rl missing from the count |
| I5 | README:177-180 | lists the looks without a count | 7 looks | SUPPORTED but uncounted |
| I6 | README:189-190, site:373 | the gravity leak | as stated | SUPPORTED, but below the fold |

## Classical stack and campaign

| ID | Where | Wording | Recomputed | Verdict |
|---|---|---|---|---|
| C1 | README:17-18, RESULTS:184-196, site:47, site:118 | 27 of 63 campaign maps, one run each, developed on 5 of them | `run_campaign_batch.py --report`: 27 of 63; tiers 12/9/4/2/0 | SUPPORTED |
| C2 | site:47 | hero stat "27/63" without "one run each" or the 5 development maps | as C1 | OVERSTATED (minor) |
| C3 | ARCH:26-27, site:129-130 | A05 first finish on the official line, 21.58 s | `A05_official_v3b/summary.json` 21.58 | SUPPORTED |
| C4 | ARCH:22-23, site:128 | official lines for 64 of 65 maps | `maps/*_official_0.5m_cl.npy` (64) | SUPPORTED |
| C5 | ARCH:174, README:57, site:259 | zero-initialised residual: 760 of 760 steps identical on A07 | `A07-Race_dev_v7a` vs `A07-Race_dev_v7a_residual0`: steering, throttle, action_idx equal on all 760 rows | SUPPORTED |

## Learned models

| ID | Where | Wording | Recomputed | Verdict |
|---|---|---|---|---|
| M1 | README:159, ARCH:102-107, site:219 | 18 features | `validation_v1.json` feature_names: 18 | SUPPORTED |
| M2 | ARCH:100, site:219 | 139,063 points | `eval_results/speed_model/dataset.npz` is not committed | UNSUPPORTED from the repo (see repro.md) |
| M3 | ARCH:111-115, RESULTS:217-218, site:220-221 | CV r 0.81 / MAE 11.7; linear 0.57 / 20.3; hand 0.33 / 34.1; test r 0.84 / 10.3 vs 0.21 / 31.1 | `validation_v1.json` cv_mlp 0.806 / 11.736, cv_linear 0.574 / 20.302, cv_hand_rule 0.328 / 34.12, test_once 0.84 / 10.25, 0.206 / 31.06 | SUPPORTED |
| M4 | ARCH:125-128 | 32,210 steps, 54 non-test maps, base rate 5.7% | `grip_model_v1.json` n_samples 32,210, 54 train_maps, base_rate 0.0574 | SUPPORTED |
| M5 | ARCH:130, 132-133 | coefficients; 56 → 23 m/s² at p = 0.2 | intercept −12.0, 0.132, 0.108; lateral_limit 56.0 and 23.4 | SUPPORTED |
| M6 | ARCH:134, RESULTS:219, site:226 | AUC 0.754 vs speed alone 0.708 | lomo auc 0.7544, auc_speed_only 0.7085, **auc_demand_only 0.7545** | OVERSTATED: the full model is no better than demand alone; the comparison shown uses the weaker baseline |
| M7 | ARCH:133 | "The speed term captures the controller's steering cap shrinking at high speed" | the speed term adds no AUC (M6) | OVERSTATED: an interpretation, not shown |
| M8 | ARCH:135, RESULTS:220 | test AUC 0.84 | test_score_once auc 0.840 | SUPPORTED |
| M9 | ARCH:56-58, RESULTS:222, site:232-233 | g = 24.87 m/s², 2,328 samples, 53 maps, IQR 24.55 to 25.25; ratio 0.80 → 1.01 | `gravity_measurement.json` | SUPPORTED |
| M10 | ARCH:151-153, RESULTS:221, site:238-239 | contact at 56% of hits vs 10%; AUC 0.62 vs 0.76 | `range_sensor/validation_v1.json` train_lomo 0.558, 0.103, 0.616, 0.762 | SUPPORTED |
| M11 | site:215 | "the maps it is scored on are never the maps it was fit on" | true for every validation listed | SUPPORTED |
| M12 | site:227 | "Single run on test maps: wall hits 11 → 2" | v5 11, v6 2 | SUPPORTED |

## Residual RL design and behaviour

| ID | Where | Wording | Recomputed | Verdict |
|---|---|---|---|---|
| R1 | README:58, ARCH:176-177, RESULTS:223, site:271 | critic burn-in Q vs return r 0.89 to 1.00 | `critic_burnin_v2.json` 0.89 to 0.999 | SUPPORTED |
| R2 | README:59, ARCH:178-181, site:275-276 | td3_v1 \|a\| ≈ 0.02; anchor weakened 10× | burn-in config anchor_alpha 2.5; td3_v3 config 25.0 | SUPPORTED |
| R3 | README:61, ARCH:188-190, site:286 | td3_v3: 20 maps (6 + 14), 200 episodes, σ 0.05 noise, seed 20261113 | `td3_v3/train_config.json` maps 20, pool_seed 20261113, steer_noise 0.05; train_log 200 rows | SUPPORTED |
| R4 | README:23 | "Three training generations fixed a stuck actor, weak exploration and narrow training data" | td3_v3's wider data did not change the held-out result (p = 0.82) | OVERSTATED: "addressed", not "fixed" |
| R5 | README:74-77, ARCH:194-200, site:297-298 | 11,608 states; td3_v3 \|a\| 0.13, r −0.22, 11 of 13; td3_v2 0.09 / 0.11, −0.13, 8 of 13; 0.2% saturated | `td3_v{2,3}_behaviour.json` | SUPPORTED |
| R6 | README:76-77, ARCH:200 | more maps with noise "made the behaviour more consistent" | two checkpoints that also differ in episodes and init; no ablation | OVERSTATED (minor): "coincided with" |
| R7 | ARCH:199 | "Both steer against the heading error, adding damping on top of pure pursuit" | pooled r −0.13 and −0.22: weak | OVERSTATED (minor) |
| R8 | ARCH:203-205 | td3_v3 "reliably fixes one map (B07) and reliably breaks another (C02)"; "raises completion by 6.7 points" without its interval | as H9, H7 | OVERSTATED |
| R9 | README:163 | evidence for the residual TD3 row points to `td3_v2/` | the final policy is td3_v3 | INCONSISTENT |
| R10 | README:242 | demo command uses `td3_v2/final.pt` | final policy td3_v3 (README:63, ARCH:207) | INCONSISTENT |
| R11 | README:245-249 | "Reproduce the headline table: report_test_set.py --tag v8_rl" | the headline is the robustness table (`report_robustness.py`) | INCONSISTENT |

## Era 1 and attribution

| ID | Where | Wording | Recomputed | Verdict |
|---|---|---|---|---|
| E1 | README:36-38, RESULTS:203-208, site:73, site:98 | Hock 5/10, best 54.06 s; 3.18M 0/50; 10.64M 8/50, all A03 | `benchmark_20261014_190540.csv` | SUPPORTED |
| E2 | site:85 | curve "falls from about 64 s to 54 s" | `site/data/era1.json` first 64.19 s, min 54.0 s | SUPPORTED |
| E3 | ATTRIBUTION:27 | "personal best of 54.000s ... (0.24s from the human World Record of 53.760s)" | 54.0 is the min of the training-eval curve (era1.json); the greedy benchmark best is 54.06 s; the world-record time has no source in the repo | UNSUPPORTED (WR figure) |
| E4 | ATTRIBUTION:21 | reward tuned "for the World Record push" | the WR chase was dropped; nothing in the repo measures it | OVERSTATED |
| E5 | ATTRIBUTION:25 | "Full migration playbook for deploying the training pipeline to an Azure Windows VM" | `CLOUD_MIGRATION.md` is gitignored, not in the public repo | UNSUPPORTED from the repo |
| E6 | ATTRIBUTION:17 | "Refactored every hardcoded "cuda" reference" | `git grep '"cuda"'`: remaining uses are `"cuda" if torch.cuda.is_available() else "cpu"` | SUPPORTED |
| E7 | README:16 | "It didn't generalise, and fine-tuning it on several maps made it collapse" | the fine-tuning collapse has no committed log | UNSUPPORTED from the repo (minor) |

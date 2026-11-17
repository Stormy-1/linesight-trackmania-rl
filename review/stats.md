# Statistical review: is there a residual-RL effect on the held-out maps?

Script: `scripts/reanalyze_clustered.py` (seed 20261117). Output: `review/clustered_reanalysis.json`.
Input: the per-run summaries of `<map>_test_v7_rob`, `<map>_test_v8_rl_rob` and `<map>_test_v9_rl_rob` in
`eval_results/hybrid_live/`, read through `scripts/report_robustness.py`'s loader. Tests:
`tests/test_reanalyze_clustered.py`.

## The problem with the pooled sign test

The headline comparison (RESULTS.md §0) is an exact sign test over 100 (map, run) pairs: 9 pairs finish only
without RL, 11 only with it, p = 0.82. That test treats the 100 pairs as independent. They are not. Runs on one map
share the map's geometry, and the noise only nudges the car around the same knife-edge events.

How strongly runs cluster, from the base version's finishes (one-way ANOVA ICC on the 0/1 outcome):

| | Value |
|---|---|
| Intraclass correlation of finishing | 0.686 |
| Design effect, 1 + (10 − 1) × ICC | 7.18 |
| Effective sample size of the 100 runs | 13.9 |

So the 100 runs carry about as much information as 14 independent ones.

## Which maps carry information

| Map | v7 | v8_rl | v9_rl | Discordant v8_rl vs v7 (only v7 / only RL) | Discordant v9_rl vs v7 |
|---|---|---|---|---|---|
| A09-Race | 5/10 | 5/10 | 7/10 | 3 / 3 | 2 / 4 |
| A15-Speed | 10/10 | 10/10 | 10/10 | 0 / 0 | 0 / 0 |
| B07-Race | 2/10 | 8/10 | 9/10 | 1 / 7 | 0 / 7 |
| C02-Race | 8/10 | 3/10 | 1/10 | 5 / 0 | 7 / 0 |
| B04, B13, C10, C13, D04, D07 | 0/10 each | 0/10 each | 0/10 each | 0 / 0 | 0 / 0 |

Seven of the ten maps contribute no discordant pair at all: six never finish in any version and A15 always does.
(The external review said six; A15 is the seventh.) Every discordant pair comes from A09, B07 and C02.

## Map as the unit

**Exact sign-flip test over maps.** The statistic is the sum of the per-map changes; the null distribution flips the
sign of each map's change, enumerating all patterns.

| Comparison | Per-map change in finishes | p (finishes) | Smallest p this design could give | p (mean completion) |
|---|---|---|---|---|
| v8_rl vs v7 | B07 +6, C02 −5, others 0 | 1.0 | 0.5 (2 maps changed) | 0.55 |
| v9_rl vs v7 | A09 +2, B07 +7, C02 −7, others 0 | 1.0 | 0.25 (3 maps changed) | 0.48 |

Even if every changed map had moved in RL's favour, the finish-count test could not have reached 0.05. That needs
at least 6 maps moving in the same direction (2/2⁶ = 0.031; with 5 maps the floor is 2/2⁵ = 0.0625).

**Cluster bootstrap over maps** (10,000 resamples of the 10 maps, percentile interval):

| Comparison | Finish-rate change, points [95% CI] | Mean-completion change, points [95% CI] |
|---|---|---|
| v8_rl vs v7 | +1.0 [−15.0, +18.0] | +4.9 [−3.0, +16.7] |
| v9_rl vs v7 | +2.0 [−19.0, +23.0] | +6.7 [−1.5, +20.0] |

The finish-rate interval runs from −19 to +23 points. RESULTS.md's completion interval (−1.6 to +20.0) is
reproduced to within resampling noise.

**Mixed-effects logistic model.** Not run. statsmodels is not installed in the project environment, and the model
would not add information here. Seven maps sit at 0% or 100% in every version, so their random intercepts run to
the boundary (complete separation within map), and the treatment effect would be estimated from A09, B07 and C02
alone, which is the same information the sign-flip test uses. With three informative clusters the random-effect
variance is not estimable in any useful sense.

## What effect could this design detect?

A power simulation (2,000 simulated studies per point): every map's finish odds are multiplied by an odds ratio,
starting from the base version's per-map rates smoothed as (k + 0.5) / 11 so that 0/10 and 10/10 maps can move.
Base and RL runs are drawn independently and the map-level sign-flip test is applied at α = 0.05.

| Odds ratio | Expected finish-rate gain, points | Power |
|---|---|---|
| 1.5 | +3.8 | 0.05 |
| 2 | +6.8 | 0.14 |
| 3 | +11.4 | 0.43 |
| 4 | +14.9 | 0.64 |
| **6** | **+20.5** | **0.90** |
| 8 | +24.8 | 0.98 |

The smallest effect with 80% power is an odds ratio between 4 and 6, roughly a +17 to +20 point gain in pooled
finish rate (from 25% to about 45%). This is optimistic: it assumes the six maps that never finish could be moved
by a residual at all, which the stall locations argue against (README "Limitations"). The observed v9_rl change is
+2 points. A real residual effect of the size the robustness numbers hint at (a few points) is far below what ten
maps can resolve, however many runs each map gets.

## The "two policies agree, so it isn't chance" argument

RESULTS.md §0 says: "The same direction in two separately trained policies makes chance an unlikely explanation."
README.md and the site say the same in other words.

The two policies are not independent replications:
- td3_v3 was initialised from td3_v2's final weights (`eval_results/residual_rl/td3_v3/train_config.json`,
  `"init": "eval_results\\residual_rl\\td3_v2\\final.pt"`).
- td3_v3's replay buffer was seeded with td3_v1's and td3_v2's episodes (`buffer_from`), 155,107 transitions at the
  start of its 200 episodes (`train_log.jsonl`).
- Both were evaluated with the same noise seed (0), so run k on C02 receives the identical steering-noise stream
  in v7, v8_rl and v9_rl.

So td3_v3 inherits td3_v2's behaviour by construction, and both face the same noise draws. Their agreement shows
that the B07 gain and the C02 loss are stable properties of one policy lineage under these noise draws. It does
not show that the trade would reappear in an independently trained policy, and it does not make chance in training
an unlikely explanation. A replication would need a policy trained from a fresh initialisation and a fresh buffer,
evaluated on a different noise seed.

What does hold: within each map, the B07 and C02 effects are large (7 of 7 discordant pairs each way, p = 0.016
uncorrected, about 0.16 after a ×10 correction). The direction on those two maps is unlikely to be evaluation
noise. Whether it generalises to other maps is unknown.

## Conclusions for the write-up

1. Report the residual-RL result as a null on the held-out set: map-level p = 1.0, finish-rate CI −19 to +23 points.
2. Say that the design cannot detect an effect smaller than roughly +17 to +20 points of finish rate, and that more
   runs per map would not change that. More maps would.
3. Drop "separately trained" and "chance is unlikely". Say the trade is consistent across two related policies.
4. Keep the per-map B07/C02 result, with its corrected p, as the one thing the residual demonstrably does.
5. A fresh held-out set exists: 26 campaign maps were never used for dev, test or RL training (23 outside the Black
   tier). They are not fully fresh. All of them are in the speed model's training data, the grip model's fitting
   logs and the v3b campaign run, and A12 and D01 were driven in the earlier showcase passes. They are fresh for
   RL and for design choices, not for the learned models.

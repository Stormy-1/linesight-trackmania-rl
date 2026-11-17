# Outline

Working title: *A classical driving stack generalises where end-to-end RL did not, and a residual policy on top
mostly reshuffles which maps finish.*

Every number in the paper is a LaTeX macro from `generated/numbers.tex` or a cell of a generated table, both written
by `make_tables.py`. The "Source" column names the file the script reads.

## Position

1. Classical robotics generalised where end-to-end RL didn't: 27/63 campaign maps (single run) versus an IQN agent
   that finished 0/50 runs off its training track (8/50 at its last checkpoint, all on one map).
2. On 10 held-out maps with paired noisy runs, a residual TD3 policy is a null result: 27/100 vs 25/100, map-level
   p = 1.0. It trades B07 for C02.
3. The evaluation design is the contribution: it shows what ten maps can and cannot resolve.

## Sections

| Section | Content | Numbers | Source |
|---|---|---|---|
| Abstract | position, headline null, design limit | 27/63; 25 vs 27/100; p 0.82, 1.0; ICC, ESS; power OR | campaign summaries; `test_set_robustness_v7_v9rl.json`; `review/clustered_reanalysis.json` (regenerated) |
| 1 Introduction | why TrackMania, the two eras, contributions list | IQN 0/50, 8/50; 27/63 | `benchmark_20261014_190540.csv`; `hybrid_live/*_official_v3b/summary.json` |
| 2 Related work | IQN, Linesight, pure pursuit, residual RL, TD3, TD3+BC; TODO cites for racing-game RL, MPC racing, sim-to-real, statistical practice in RL evaluation | none | refs.bib |
| 3 Method | sense/plan/act stack; learned speed, grip, gravity, lidar; residual TD3 (action bounds, zero init, burn-in, anchor) | speed CV r/MAE; grip AUC; g; lidar hit rate; critic r; residual bounds 0.25, 8 m/s; 760/760 identical steps | `speed_model/validation_v1.json`, `grip_model/grip_model_v1.json`, `gravity_measurement.json`, `range_sensor/validation_v1.json`, `residual_rl/critic_burnin_v2.json`, run `summary.json` residual_config, A07 zero-residual CSVs |
| 4 Protocol | fixed split; no per-map tuning; paired seeded noise; selection rule; td3_v3 training; map-level analysis | split seed, sizes; sigma, seed, runs per map; dev rule 59/60/60 of 70; td3_v3 maps, episodes, buffer | `maps/map_split.json`; run `summary.json`; `dev_robustness_*.json`; `td3_v3/train_config.json`, `train_log.jsonl` |
| 5 Results | 5.1 generalisation (Table: campaign tiers + IQN); 5.2 held-out robustness (Table per map); 5.3 map-level analysis and power (Table + Figure); 5.4 single-run history (Table) and why it misled; 5.5 what the residual does (behaviour stats, A15 lap time) | everything in the tables; pairs 16/9/11/64; recovery -23%, walls -22%; v8_rl single 4/10 vs 2/10, recovery -42%; behaviour r, abs a | robustness JSONs; `test_set_v{4..8}*.json`; `td3_v{2,3}_behaviour.json`; per-run summaries |
| 6 Limitations | the six required items, plus: car is slow (time ratio) | 7 looks; 4 m/s impact threshold; 3 of 10 informative maps; power floor; single run per campaign map | `review/stats.md`, `review/claims.md`, `test_set_v*.json` |
| 7 Conclusion | the position, restated as a finding with its limit, and the next experiment (fresh maps, independent policy) | none new | |

## Tables and figures (all generated)

- `tab_generalisation.tex`: campaign finishes per tier; IQN per checkpoint on Hockolicious and five other maps.
- `tab_models.tex`: learned-model validation with baselines.
- `tab_robustness.tex`: per-map finishes v7 / v8_rl / v9_rl, completion, discordant pairs; totals.
- `tab_maplevel.tex`: pooled sign test, map sign-flip test, attainable floor, cluster bootstrap, ICC, ESS.
- `tab_history.tex`: single-run v4 to v8_rl aggregates.
- `fig_maps_power.pdf`: (a) per-map change in finishes, td3_v3 vs base; (b) power of the map-level test vs
  effect size, with the observed change and its interval.

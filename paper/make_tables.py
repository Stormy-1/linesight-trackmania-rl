"""Build the paper's tables, number macros and figure from the committed result files.

Usage (from the repo root, no game needed):

    python paper/make_tables.py                      # reruns scripts/reanalyze_clustered.py (about 20 s)
    python paper/make_tables.py --reuse-reanalysis   # reads review/clustered_reanalysis.json instead

Writes to paper/generated/:
    numbers.tex                 one \\newcommand per number quoted in the text
    tab_*.tex                   tabular bodies, \\input by main.tex
    fig_maps_power.pdf          per-map change and the map-level power curve
    clustered_reanalysis.json   the reanalysis output the tables were built from (rerun mode only)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval_results"
LIVE = EVAL / "hybrid_live"
OUT = ROOT / "paper" / "generated"
REVIEW_JSON = ROOT / "review" / "clustered_reanalysis.json"
sys.path.insert(0, str(ROOT / "scripts"))

ALPHA = 0.05
TARGET_POWER = 0.8
N_TEST_MAPS_FOR_CORRECTION = 10
ERA1_OTHER_MAPS = ("A01", "A03", "A05", "A07", "A11")
VERSIONS = ("v7_rob", "v8_rl_rob", "v9_rl_rob")
TIER_ORDER = ("White", "Green", "Blue", "Red", "Black")
MINUS = "−"

# Palette (dataviz reference instance): diverging blue/orange poles, neutral grey, text inks.
BLUE, ORANGE, GREY, INK, INK_2 = "#2a78d6", "#eb6834", "#b4b2ac", "#0b0b0b", "#52514e"


# ---------------------------------------------------------------- helpers


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rnd(x: float, digits: int) -> str:
    """Round half up on the decimal text, so 10.25 prints as 10.3 (float formatting gives 10.2)."""

    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def fmt_p(p: float) -> str:
    if p >= 0.995:
        return "1.0"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def signed(x: float, digits: int = 1) -> str:
    text = f"{x:+.{digits}f}"
    return text.replace("-", "$-$") if x < 0 else text


def short(stem: str) -> str:
    return stem.split("-")[0]


def sign_test_p(k_a: int, k_b: int) -> float:
    """Exact two-sided sign test on discordant pairs."""

    n = k_a + k_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k_a, k_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"wrote {(OUT / name).relative_to(ROOT)}")


def tabular(spec: str, header: str, rows: list[str], footer: list[str] | None = None) -> str:
    body = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header + r" \\", "\\midrule"]
    body += [r + r" \\" for r in rows]
    if footer:
        body.append("\\midrule")
        body += [r + r" \\" for r in footer]
    body += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(body)


# ---------------------------------------------------------------- protocol


def protocol_macros() -> dict[str, str]:
    """Evaluation settings, read from every robustness run's summary.json and the map split."""

    settings = set()
    for version in VERSIONS:
        for path in LIVE.glob(f"*_test_{version}/summary.json"):
            s = load_json(path)
            settings.add((s["steering_noise_std"], s["perturbation_seed"], len(s["runs"]), s["game_speed"],
                          s["stuck_cutoff_s"], s["grip_risk"], s["top_speed_mps"]))
    if len(settings) != 1:
        raise SystemExit(f"robustness runs disagree on their settings: {settings}")
    sigma, seed, runs, speed, stuck, risk, top = settings.pop()
    rl = load_json(LIVE / "C02-Race_test_v9_rl_rob" / "summary.json")["residual_config"]
    split = load_json(ROOT / "maps" / "map_split.json")
    manifest = load_json(ROOT / "maps" / "official_manifest.json")
    return {
        "NoiseSigma": f"{sigma}", "NoiseSeed": f"{seed}", "RunsPerMap": f"{runs}", "GameSpeed": f"{speed:g}",
        "StuckCutoff": f"{stuck:g}", "GripRisk": f"{risk}", "TopSpeed": f"{top:g}",
        "ResSteer": f"{rl['max_steering_delta']}", "ResSpeed": f"{rl['max_target_speed_delta_mps']:g}",
        "SplitSeed": f"{split['seed']}", "NTest": f"{len(split['test'])}",
        "NDev": f"{sum(1 for m in split['dev'] if m in manifest)}",
        "NRLTrain": f"{len(split['rl_train'])}", "NOfflineTrain": f"{len(split['offline_train'])}",
    }


# ---------------------------------------------------------------- robustness (held-out test set)


def robustness(reanalysis: dict) -> dict[str, str]:
    v8 = load_json(EVAL / "test_set_robustness_v7_v8rl.json")
    v9 = load_json(EVAL / "test_set_robustness_v7_v9rl.json")
    base_maps, v8_maps, v9_maps = v9["compare"]["baseline_per_map"], v8["per_map"], v9["per_map"]
    per_map = reanalysis["v9_rl_rob_vs_v7_rob"]["per_map"]
    rows = []
    for stem, b in base_maps.items():
        d = per_map[stem]
        disc = f"{d['only_base']} / {d['only_other']}"
        rows.append(f"{short(stem)} & {b['tier']} & {b['finished']} & {v8_maps[stem]['finished']} & "
                    f"{v9_maps[stem]['finished']} & {b['mean_completion_pct']:.1f} & "
                    f"{v9_maps[stem]['mean_completion_pct']:.1f} & {disc}")
    bt, t8, t9, pairs = v9["compare"]["baseline_total"], v8["total"], v9["total"], v9["compare"]["pairs"]
    footer = [f"Total & & {bt['finished']} & {t8['finished']} & {t9['finished']} & "
              f"{bt['mean_completion_pct']:.1f} & {t9['mean_completion_pct']:.1f} & "
              f"{pairs['only_a']} / {pairs['only_b']}",
              f"Walls & & {bt['walls']} & {t8['walls']} & {t9['walls']} & & &",
              f"Recovery & & {thousands(bt['recovery_steps'])} & {thousands(t8['recovery_steps'])} & "
              f"{thousands(t9['recovery_steps'])} & & &"]
    header = (r"Map & Tier & v7 & v8\_rl & v9\_rl & \multicolumn{2}{c}{Compl.\ (\%)} & Disc. \\"
              "\n" r" & & \multicolumn{3}{c}{finished / 10} & v7 & v9\_rl & v7 / v9\_rl")
    write("tab_robustness.tex", tabular("@{}llrrrrrc@{}", header, rows, footer))
    return robustness_macros(v8, v9, bt, t9)


def robustness_macros(v8: dict, v9: dict, bt: dict, t9: dict) -> dict[str, str]:
    pairs, pairs8 = v9["compare"]["pairs"], v8["compare"]["pairs"]
    return {
        "TestRuns": f"{bt['runs']}", "BaseFin": f"{bt['finished']}", "VeightFin": f"{v8['total']['finished']}",
        "VnineFin": f"{t9['finished']}", "BaseCompl": f"{bt['mean_completion_pct']:.1f}",
        "VnineCompl": f"{t9['mean_completion_pct']:.1f}", "BaseMapsAny": f"{bt['maps_finished_at_least_once']}",
        "PairsBoth": f"{pairs['both']}", "PairsOnlyBase": f"{pairs['only_a']}", "PairsOnlyRL": f"{pairs['only_b']}",
        "PairsNeither": f"{pairs['neither']}", "SignP": fmt_p(v9["compare"]["sign_test_p"]),
        "VeightOnlyBase": f"{pairs8['only_a']}", "VeightOnlyRL": f"{pairs8['only_b']}",
        "VeightSignP": fmt_p(v8["compare"]["sign_test_p"]),
        "BaseWalls": f"{bt['walls']}", "VnineWalls": f"{t9['walls']}",
        "WallsDropPct": f"{round(100 * (1 - t9['walls'] / bt['walls']))}",
        "BaseRecovery": thousands(bt["recovery_steps"]), "VnineRecovery": thousands(t9["recovery_steps"]),
        "RecoveryDropPct": f"{round(100 * (1 - t9['recovery_steps'] / bt['recovery_steps']))}",
    }


def per_map_macros(reanalysis: dict) -> dict[str, str]:
    """B07, C02 and A09 outcomes and the per-map sign tests."""

    per_map = reanalysis["v9_rl_rob_vs_v7_rob"]["per_map"]
    macros = {}
    for stem, name in (("B07-Race", "Bseven"), ("C02-Race", "Ctwo"), ("A09-Race", "Anine")):
        d = per_map[stem]
        p = sign_test_p(d["only_base"], d["only_other"])
        macros |= {f"{name}Base": f"{d['base_finished']}", f"{name}RL": f"{d['other_finished']}",
                   f"{name}OnlyBase": f"{d['only_base']}", f"{name}OnlyRL": f"{d['only_other']}",
                   f"{name}P": fmt_p(p), f"{name}PCorr": f"{min(1.0, p * N_TEST_MAPS_FOR_CORRECTION):.2f}"}
    v8 = reanalysis["v8_rl_rob_vs_v7_rob"]["per_map"]
    macros |= {"BsevenVeight": f"{v8['B07-Race']['other_finished']}",
               "CtwoVeight": f"{v8['C02-Race']['other_finished']}"}
    unchanged = sum(1 for d in per_map.values() if d["base_finished"] == d["other_finished"])
    never = sum(1 for d in per_map.values() if d["base_finished"] == 0 and d["other_finished"] == 0)
    always = sum(1 for d in per_map.values() if d["base_finished"] == 10 and d["other_finished"] == 10)
    macros |= {"UnchangedMaps": f"{unchanged}", "NeverMaps": f"{never}", "AlwaysMaps": f"{always}",
               "InformativeMaps": f"{len(reanalysis['v9_rl_rob_vs_v7_rob']['maps_with_discordant_pairs'])}"}
    return macros


# ---------------------------------------------------------------- map-level reanalysis


def run_reanalysis(reuse: bool) -> dict:
    if reuse:
        return load_json(REVIEW_JSON)
    target = OUT / "clustered_reanalysis.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "reanalyze_clustered.py"), "--json", str(target)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    result = load_json(target)
    if result != load_json(REVIEW_JSON):
        print("WARNING: regenerated reanalysis differs from review/clustered_reanalysis.json")
    return result


def power_bracket(curve: dict) -> tuple[str, dict, str, dict]:
    """The grid odds ratios just below and at or above the target power."""

    ordered = sorted(curve.items(), key=lambda kv: float(kv[0]))
    above = next(i for i, (_, v) in enumerate(ordered) if v["power"] >= TARGET_POWER)
    return (*ordered[above - 1], *ordered[above])


def maplevel(reanalysis: dict) -> dict[str, str]:
    clus = reanalysis["clustering_v7"]
    rows = []
    for key, label in (("v8_rl_rob_vs_v7_rob", "v8\\_rl"), ("v9_rl_rob_vs_v7_rob", "v9\\_rl")):
        r = reanalysis[key]
        f, c = r["bootstrap_finish_rate_diff_pts"], r["bootstrap_completion_diff_pts"]
        rows.append(f"{label} & {fmt_p(r['map_sign_flip_p_finishes'])} & {fmt_p(r['min_attainable_p_finishes'])} & "
                    f"{signed(f[0])} [{signed(f[1])}, {signed(f[2])}] & {fmt_p(r['map_sign_flip_p_completion'])} & "
                    f"{signed(c[0])} [{signed(c[1])}, {signed(c[2])}]")
    header = (r"vs v7 & $p_{\mathrm{fin}}$ & floor & $\Delta$ finish rate & $p_{\mathrm{compl}}$ & "
              r"$\Delta$ completion")
    write("tab_maplevel.tex", tabular("@{}lrrcrc@{}", header, rows))

    lo_or, lo, hi_or, hi = power_bracket(reanalysis["power_map_level"]["curve"])
    floor_maps = next(n for n in range(1, 20) if 2 / 2**n < ALPHA)
    r9 = reanalysis["v9_rl_rob_vs_v7_rob"]
    f9, c9 = r9["bootstrap_finish_rate_diff_pts"], r9["bootstrap_completion_diff_pts"]
    return {
        "ICC": f"{clus['icc_finish']:.2f}", "DesignEffect": f"{clus['design_effect']:.1f}",
        "ESS": f"{clus['effective_sample_size']:.0f}", "MapP": fmt_p(r9["map_sign_flip_p_finishes"]),
        "MapPFloor": fmt_p(r9["min_attainable_p_finishes"]), "FloorMaps": f"{floor_maps}",
        "FloorMapsP": f"{2 / 2**floor_maps:.3f}",
        "FinDiff": signed(f9[0]), "FinLo": signed(f9[1]), "FinHi": signed(f9[2]),
        "ComplDiff": signed(c9[0]), "ComplLo": signed(c9[1]), "ComplHi": signed(c9[2]),
        "PowerLoOR": f"{float(lo_or):g}", "PowerLo": f"{lo['power']:.2f}",
        "PowerLoGain": f"{lo['expected_finish_rate_gain_pts']:.1f}",
        "PowerHiOR": f"{float(hi_or):g}", "PowerHi": f"{hi['power']:.2f}",
        "PowerHiGain": f"{hi['expected_finish_rate_gain_pts']:.1f}",
        "PowerSims": "2{,}000", "BootResamples": "10{,}000", "ReanalysisSeed": f"{reanalysis['seed']}",
    }


# ---------------------------------------------------------------- lap times (paired, per-run logs)


def paired_median_delta(base: list, other: list) -> tuple[float, int]:
    deltas = [o.race_time_s - b.race_time_s for b, o in zip(base, other) if b.finished and o.finished]
    return statistics.median(deltas), len(deltas)


def lap_time_macros() -> dict[str, str]:
    from report_robustness import collect_runs

    runs = {v: collect_runs("test", v) for v in VERSIONS}
    stem = "A15-Speed"
    d8, _ = paired_median_delta(runs["v7_rob"][stem], runs["v8_rl_rob"][stem])
    d9, n9 = paired_median_delta(runs["v7_rob"][stem], runs["v9_rl_rob"][stem])
    return {"AfifteenDeltaVeight": signed(d8, 2), "AfifteenDeltaVnine": signed(d9, 2), "AfifteenPairs": f"{n9}"}


# ---------------------------------------------------------------- single-run history and dev maps


def history() -> dict[str, str]:
    files = (("v4", "test_set_v4_baseline.json", "classical stack"),
             ("v5", "test_set_v5_speed_model.json", "+ learned speed"),
             ("v6", "test_set_v6_grip_model.json", "+ grip limit"),
             ("v7", "test_set_v7_gravity.json", "+ measured $g$"),
             ("v8\\_rl", "test_set_v8_rl_td3v2.json", "+ residual TD3"))
    rows, aggs = [], {}
    for tag, name, added in files:
        a = load_json(EVAL / name)["aggregate"]
        aggs[tag] = a
        rows.append(f"{tag} & {added} & {a['finished']}/{a['maps']} & {a['mean_completion_pct']:.1f} & "
                    f"{a['median_time_ratio']:.2f} & {a['walls']} & {a['recovery_steps']}")
    header = r"Ver. & Change & Fin. & Compl.\,\% & $t/t_{\mathrm{pro}}$ & Walls & Recov."
    write("tab_history.tex", tabular("@{}llrrrrr@{}", header, rows))
    v7, v8 = aggs["v7"], aggs["v8\\_rl"]
    ratios = [a["median_time_ratio"] for a in aggs.values()]
    return {"SingleVsevenFin": f"{v7['finished']}", "SingleVeightFin": f"{v8['finished']}",
            "SingleVsevenRec": f"{v7['recovery_steps']}", "SingleVeightRec": f"{v8['recovery_steps']}",
            "SingleRecDropPct": f"{round(100 * (1 - v8['recovery_steps'] / v7['recovery_steps']))}",
            "SingleWallsVfour": f"{aggs['v4']['walls']}", "SingleWallsVfive": f"{aggs['v5']['walls']}", "SingleWallsVsix": f"{aggs['v6']['walls']}",
            "TimeRatioLo": f"{min(ratios):.2f}", "TimeRatioHi": f"{max(ratios):.2f}"}


def dev_macros() -> dict[str, str]:
    d2 = load_json(EVAL / "dev_robustness_td3v2_rob_vs_v7a.json")
    d3 = load_json(EVAL / "dev_robustness_td3v3_rob_vs_v7a.json")
    base, t2, t3, pairs = d3["compare"]["baseline_total"], d2["total"], d3["total"], d3["compare"]["pairs"]
    return {"DevRuns": f"{base['runs']}", "DevBaseFin": f"{base['finished']}", "DevVtwoFin": f"{t2['finished']}",
            "DevVthreeFin": f"{t3['finished']}", "DevBaseCompl": f"{base['mean_completion_pct']:.1f}",
            "DevVtwoCompl": f"{t2['mean_completion_pct']:.1f}", "DevVthreeCompl": f"{t3['mean_completion_pct']:.1f}",
            "DevBaseWalls": f"{base['walls']}", "DevVthreeWalls": f"{t3['walls']}",
            "DevOnlyBase": f"{pairs['only_a']}", "DevOnlyRL": f"{pairs['only_b']}",
            "DevSignP": fmt_p(d3["compare"]["sign_test_p"])}


# ---------------------------------------------------------------- generalisation: campaign and era 1


def campaign() -> tuple[Counter, Counter, dict]:
    from run_campaign_batch import drivable_maps, load_summary, run_name

    manifest = load_json(ROOT / "maps" / "official_manifest.json")
    driven, finished, by_stem = Counter(), Counter(), {}
    for stem in drivable_maps(manifest):
        summary = load_summary(run_name(stem, "v3b"))
        if summary is None:
            continue
        ok = bool(summary["runs"][0]["finished"])
        driven[manifest[stem]["group"]] += 1
        finished[manifest[stem]["group"]] += int(ok)
        by_stem[short(stem)] = ok
    return driven, finished, by_stem


def era1() -> tuple[Counter, dict]:
    """Finishes per (checkpoint, map) and best finish time from the greedy benchmark."""

    counts, best = Counter(), {}
    with open(EVAL / "benchmark_20261014_190540.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["checkpoint"], row["track"])
            counts[key, "runs"] += 1
            if row["finished"] == "True":
                counts[key, "fin"] += 1
                best[key] = min(best.get(key, math.inf), float(row["race_time_s"]))
    return counts, best


def generalisation() -> dict[str, str]:
    driven, finished, by_stem = campaign()
    counts, best = era1()
    checkpoints = sorted({k[0][0] for k in counts}, key=lambda c: float(c.split("_")[1].rstrip("M")))
    rows = [f"Stack (v3b), campaign & {t} & {finished[t]}/{driven[t]}" for t in TIER_ORDER if driven[t]]
    rows.append(f"Stack (v3b), campaign & all & {sum(finished.values())}/{sum(driven.values())}")
    other = {}
    for ckpt in checkpoints:
        frames = ckpt.split("_")[1]
        hock = (ckpt, "hock")
        other[ckpt] = (sum(counts[(ckpt, m), "fin"] for m in ERA1_OTHER_MAPS),
                       sum(counts[(ckpt, m), "runs"] for m in ERA1_OTHER_MAPS))
        rows.append(f"IQN {frames}, training map & Hock. & {counts[hock, 'fin']}/{counts[hock, 'runs']}")
        rows.append(f"IQN {frames}, other maps & 5 maps & {other[ckpt][0]}/{other[ckpt][1]}")
    write("tab_generalisation.tex", tabular("@{}llr@{}", r"Agent & Maps & Finished", rows))
    first, last = checkpoints[0], checkpoints[-1]
    return {"CampaignFin": f"{sum(finished.values())}", "CampaignMaps": f"{sum(driven.values())}",
            "IQNFirstFrames": first.split("_")[1], "IQNLastFrames": last.split("_")[1],
            "IQNFirstOther": f"{other[first][0]}", "IQNLastOther": f"{other[last][0]}",
            "IQNOtherRuns": f"{other[last][1]}", "IQNOtherMaps": f"{len(ERA1_OTHER_MAPS)}", "IQNLastAthree": f"{counts[(last, 'A03'), 'fin']}",
            "IQNHockFin": f"{counts[(last, 'hock'), 'fin']}", "IQNHockRuns": f"{counts[(last, 'hock'), 'runs']}",
            "IQNHockBest": f"{best[(last, 'hock')]:.2f}",
            "StackEraOneMapsFin": f"{sum(by_stem.get(m, False) for m in ERA1_OTHER_MAPS)}"}


# ---------------------------------------------------------------- learned models and the residual policy


def models() -> dict[str, str]:
    sp = load_json(EVAL / "speed_model" / "validation_v1.json")
    gr = load_json(EVAL / "grip_model" / "grip_model_v1.json")
    rs = load_json(EVAL / "range_sensor" / "validation_v1.json")["train_lomo"]
    g = load_json(EVAL / "gravity_measurement.json")
    mlp, hand, tm, th = sp["cv_mlp"], sp["cv_hand_rule"], sp["test_once"]["mlp"], sp["test_once"]["hand_rule"]
    rows = [
        f"Speed MLP & CV by map & $r$ {rnd(mlp['r'], 2)}, MAE {rnd(mlp['mae_mps'], 1)} & "
        f"hand rule $r$ {rnd(hand['r'], 2)}, MAE {rnd(hand['mae_mps'], 1)}",
        f"Speed MLP & test, once & $r$ {rnd(tm['r'], 2)}, MAE {rnd(tm['mae_mps'], 1)} & "
        f"hand rule $r$ {rnd(th['r'], 2)}, MAE {rnd(th['mae_mps'], 1)}",
        f"Grip logistic & LOMO, {len(gr['train_maps'])} maps & AUC {gr['lomo']['auc']:.3f} & "
        f"$v^2|\\kappa|$ alone {gr['lomo']['auc_demand_only']:.3f}",
        f"Grip logistic & test, once & AUC {gr['test_score_once']['auc']:.2f} & ",
        f"Virtual lidar & LOMO, {rs['wall_hits_scored']} hits & contact at {100 * rs['hits_within_contact_m']:.0f}\\% "
        f"of hits & {100 * rs['ordinary_within_contact_m']:.0f}\\% of other steps",
        f"Gravity & {thousands(g['n_samples'])} samples & {g['gravity_mps2']:.2f} m/s$^2$ & "
        f"IQR {g['iqr_mps2'][0]:.2f}--{g['iqr_mps2'][1]:.2f}",
    ]
    write("tab_models.tex", tabular("@{}llll@{}", r"Model & Validation & Result & Reference", rows))
    critic_r = [c["pearson_r"] for c in load_json(EVAL / "residual_rl" / "critic_burnin_v2.json")["train_fit"].values()]
    return {
        "SpeedFeatures": f"{len(sp['feature_names'])}", "SpeedCVr": rnd(mlp["r"], 2),
        "SpeedLinr": rnd(sp["cv_linear"]["r"], 2), "SpeedHandr": rnd(hand["r"], 2),
        "GripSamples": thousands(gr["n_samples"]), "GripMaps": f"{len(gr['train_maps'])}",
        "GripBaseRate": f"{100 * gr['lomo']['base_rate']:.1f}", "GripAUC": f"{gr['lomo']['auc']:.3f}",
        "GripAUCDemand": f"{gr['lomo']['auc_demand_only']:.3f}", "GripAUCSpeed": f"{gr['lomo']['auc_speed_only']:.3f}",
        "GripIntercept": f"{gr['intercept']:.1f}", "GripDemandCoef": f"{gr['demand_coef']:.3f}",
        "GripSpeedCoef": f"{gr['speed_coef']:.3f}",
        "Gravity": f"{g['gravity_mps2']:.2f}", "GravityMaps": f"{g['n_maps']}",
        "GravityOldRatio": f"{g['detector_over_pro_speed_median']['15.4']:.2f}",
        "GravityNewRatio": f"{g['detector_over_pro_speed_median']['measured']:.2f}",
        "LidarHitPct": f"{100 * rs['hits_within_contact_m']:.0f}",
        "LidarOrdPct": f"{100 * rs['ordinary_within_contact_m']:.0f}",
        "LidarAUC": f"{rs['auc_warning_range']:.2f}", "LateralAUC": f"{rs['auc_warning_lateral']:.2f}",
        "CriticRLo": f"{min(critic_r):.2f}", "CriticRHi": f"{max(critic_r):.2f}", "CriticMaps": f"{len(critic_r)}",
    }


def behaviour() -> dict[str, str]:
    macros = {}
    for name, label in (("td3_v2", "Vtwo"), ("td3_v3", "Vthree")):
        b = load_json(EVAL / "residual_rl" / f"{name}_behaviour.json")
        negative = sum(1 for r in b["per_run"].values() if r["corr_steer_vs_sin_heading_error"] < 0)
        macros |= {f"Beh{label}SteerA": f"{b['pooled']['mean_abs_action'][0]:.2f}",
                   f"Beh{label}R": signed(b["pooled"]["corr_steer_vs_sin_heading_error"], 2),
                   f"Beh{label}Neg": f"{negative}", f"Beh{label}Sat": f"{100 * b['pooled']['saturated_share']:.1f}",
                   "BehRuns": f"{len(b['per_run'])}", "BehStates": thousands(b["pooled"]["steps"])}
    return macros


def training() -> dict[str, str]:
    cfg = load_json(EVAL / "residual_rl" / "td3_v3" / "train_config.json")
    lines = (EVAL / "residual_rl" / "td3_v3" / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    log = [json.loads(line) for line in lines if line.strip()]
    split = load_json(ROOT / "maps" / "map_split.json")
    return {"TrainMaps": f"{len(cfg['maps'])}", "TrainExtraMaps": f"{len(set(cfg['maps']) - set(split['rl_train']))}",
            "TrainEpisodes": f"{len(log)}", "TrainNoise": f"{cfg['steer_noise']}",
            "TrainNoiseSeed": f"{cfg['perturb_seed']}", "TrainPoolSeed": f"{cfg['pool_seed']}",
            "BufferStart": thousands(log[0]["buffer_size"]), "BufferEnd": thousands(log[-1]["buffer_size"]),
            "AnchorAlpha": f"{cfg['td3']['anchor_alpha']:g}", "ExploreSigma": f"{cfg['training']['exploration_noise']}",
            "NoiseHold": f"{cfg['training']['noise_hold_steps']}"}


def design_constants() -> dict[str, str]:
    """Design constants quoted in the method section, read from the code that uses them."""

    sys.path.insert(0, str(ROOT))
    from trackmania_rl.hybrid_live import HybridLivePolicyConfig
    from trackmania_rl.range_sensor import DEFAULT_MAX_RANGE_M
    from trackmania_rl.residual_observation import OBSERVATION_SIZE, RAY_FEATURES
    from trackmania_rl.speed_model import HIDDEN_UNITS

    td3 = load_json(EVAL / "residual_rl" / "td3_v3" / "train_config.json")["td3"]
    return {"ObsSize": f"{OBSERVATION_SIZE}", "NRays": f"{len(RAY_FEATURES)}", "RayRange": f"{DEFAULT_MAX_RANGE_M:g}",
            "SpeedHidden": f"{HIDDEN_UNITS}", "TDHidden": f"{td3['hidden_units']}", "TDGamma": f"{td3['gamma']}",
            "ImpactDrop": f"{HybridLivePolicyConfig().impact_speed_drop_mps:g}"}


def zero_residual_check() -> dict[str, str]:
    """Live A07 runs with no residual and with a zero-initialised one: identical commands per step."""

    def rows(name: str) -> list[dict]:
        with open(LIVE / name / "run01.csv", newline="") as handle:
            return list(csv.DictReader(handle))

    base, zero = rows("A07-Race_dev_v7a"), rows("A07-Race_dev_v7a_residual0")
    fields = ("steering", "throttle", "action_idx")
    same = sum(1 for a, b in zip(base, zero) if all(a[f] == b[f] for f in fields))
    times = [float(r["race_time_s"]) for r in base]
    step_ms = statistics.median(b - a for a, b in zip(times, times[1:])) * 1000
    return {"ZeroSame": f"{same}", "ZeroSteps": f"{max(len(base), len(zero))}", "StepMs": f"{step_ms:.0f}"}


def fresh_maps() -> dict[str, str]:
    """Drivable campaign tracks never used for dev, test or residual-RL training (stats.md, conclusion 5)."""

    from run_campaign_batch import drivable_maps

    manifest = load_json(ROOT / "maps" / "official_manifest.json")
    split = load_json(ROOT / "maps" / "map_split.json")
    trained = load_json(EVAL / "residual_rl" / "td3_v3" / "train_config.json")["maps"]
    used = set(split["dev"]) | set(split["test"]) | set(split["rl_train"]) | set(trained)
    fresh = [m for m in drivable_maps(manifest) if m not in used]
    return {"FreshMaps": f"{len(fresh)}",
            "FreshMapsNonBlack": f"{sum(1 for m in fresh if manifest[m]['group'] != 'Black')}"}


# ---------------------------------------------------------------- figure


def plot_per_map(ax, per_map: dict) -> None:
    stems = list(per_map)[::-1]
    deltas = [per_map[s]["other_finished"] - per_map[s]["base_finished"] for s in stems]
    colours = [BLUE if d > 0 else ORANGE if d < 0 else GREY for d in deltas]
    ax.barh(range(len(stems)), deltas, color=colours, height=0.6)
    ax.axvline(0, color=INK_2, linewidth=0.6)
    ax.set_yticks(range(len(stems)), [short(s) for s in stems])
    for y, d in enumerate(deltas):
        if d:
            ax.text(d + (0.4 if d > 0 else -0.4), y, f"{d:+d}".replace("-", MINUS), va="center",
                    ha="left" if d > 0 else "right", color=INK, fontsize=6)
    ax.set_xlim(-10, 10)
    ax.set_xlabel("change in finishes /10")
    ax.set_title("(a) v9_rl − v7, per map", fontsize=7, color=INK)


def plot_power(ax, reanalysis: dict) -> None:
    curve = sorted((v["expected_finish_rate_gain_pts"], v["power"])
                   for v in reanalysis["power_map_level"]["curve"].values())
    gain, lo, hi = reanalysis["v9_rl_rob_vs_v7_rob"]["bootstrap_finish_rate_diff_pts"]
    ax.axvspan(lo, hi, color=GREY, alpha=0.35, linewidth=0)
    ax.axvline(gain, color=ORANGE, linewidth=1.2)
    ax.axhline(TARGET_POWER, color=INK_2, linewidth=0.6, linestyle="--")
    ax.plot([c[0] for c in curve], [c[1] for c in curve], color=BLUE, linewidth=1.5, marker="o", markersize=2.5)
    ax.text(lo + 1, 0.55, "observed\n95% CI", color=INK, fontsize=6, va="top")
    ax.set_xlim(lo - 2, 58)
    ax.set_xticks([-20, 0, 20, 40])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("finish-rate gain (points)")
    ax.set_ylabel("power")
    ax.set_title("(b) map-level test power", fontsize=7, color=INK)


def figure(reanalysis: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw={"width_ratios": [1, 1.25]})
    plot_per_map(ax_a, reanalysis["v9_rl_rob_vs_v7_rob"]["per_map"])
    plot_power(ax_b, reanalysis)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_maps_power.pdf")
    plt.close(fig)
    print(f"wrote {(OUT / 'fig_maps_power.pdf').relative_to(ROOT)}")


# ---------------------------------------------------------------- main


def write_macros(macros: dict[str, str]) -> None:
    bad = [k for k in macros if not k.isalpha()]
    if bad:
        raise SystemExit(f"macro names must be letters only: {bad}")
    lines = ["% Generated by paper/make_tables.py. Do not edit by hand.", "\\RequirePackage{xspace}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}\\xspace}}" for k, v in sorted(macros.items())]
    write("numbers.tex", "\n".join(lines) + "\n")


def build_overleaf_zip() -> None:
    """Everything Overleaf needs to compile main.tex. IEEEtran.cls is left out: Overleaf ships it."""

    import zipfile

    paper = ROOT / "paper"
    files = [paper / "main.tex", paper / "refs.bib", *sorted(OUT.glob("*.tex")), OUT / "fig_maps_power.pdf"]
    with zipfile.ZipFile(paper / "overleaf.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(paper).as_posix())
    print(f"wrote paper/overleaf.zip ({len(files)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse-reanalysis", action="store_true",
                        help="read review/clustered_reanalysis.json instead of rerunning the script")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    reanalysis = run_reanalysis(args.reuse_reanalysis)
    macros = protocol_macros()
    macros |= robustness(reanalysis)
    macros |= per_map_macros(reanalysis)
    macros |= maplevel(reanalysis)
    macros |= lap_time_macros()
    macros |= history()
    macros |= dev_macros()
    macros |= generalisation()
    macros |= models()
    macros |= behaviour()
    macros |= training()
    macros |= zero_residual_check()
    macros |= design_constants()
    macros |= fresh_maps()
    write_macros(macros)
    figure(reanalysis)
    build_overleaf_zip()


if __name__ == "__main__":
    main()

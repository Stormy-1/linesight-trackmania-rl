"""Summarize recorded traces for offline hybrid speed-profile calibration.

This report is descriptive only. It does not modify controller parameters or
claim that speeds observed from an RL run are safe targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p25": None, "p50": None, "p75": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def _section(records: list[dict[str, Any]], section_index: int, bins: int, total_length: float) -> dict[str, Any]:
    start = section_index / bins
    end = (section_index + 1) / bins
    selected = [
        item
        for item in records
        if start <= float(item["progress_m"]) / total_length < end
        or section_index == bins - 1 and start <= float(item["progress_m"]) / total_length <= end
    ]
    speeds = [float(item["speed_mps"]) for item in selected]
    targets = [float(item["target_speed_mps"]) for item in selected]
    curvatures = [abs(float(item["local_curvature"])) for item in selected]
    gaps = [float(item["speed_mps"]) - float(item["target_speed_mps"]) for item in selected]
    return {
        "bin_index": section_index,
        "progress_fraction_start": start,
        "progress_fraction_end": end,
        "samples": len(selected),
        "actual_speed_mps": _percentiles(speeds),
        "nominal_target_speed_mps": _percentiles(targets),
        "absolute_curvature": _percentiles(curvatures),
        "speed_gap_actual_minus_nominal_mps": _percentiles(gaps),
        "fraction_above_nominal_target": float(sum(gap > 0.0 for gap in gaps) / len(gaps)) if gaps else 0.0,
        "mean_abs_lateral_error_m": float(np.mean([abs(float(item["lateral_error_m"])) for item in selected])) if selected else None,
        "mean_abs_heading_error_rad": float(np.mean([abs(float(item["heading_error_rad"])) for item in selected])) if selected else None,
        "mean_abs_elevation_error_m": float(np.mean([abs(float(item["elevation_error_m"])) for item in selected])) if selected else None,
        "recovery_rate": float(sum(bool(item["recovery"]) for item in selected) / len(selected)) if selected else 0.0,
    }


def summarize_trace(path: Path, bins: int = 10) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = list(data["states"])
    total_length = max(float(data["end_progress_m"]), 1e-9)
    speeds = [float(item["speed_mps"]) for item in records]
    targets = [float(item["target_speed_mps"]) for item in records]
    return {
        "file": str(path),
        "samples": len(records),
        "duration_s": data["duration_s"],
        "progress_gain_m": data["progress_gain_m"],
        "mean_actual_speed_mps": float(np.mean(speeds)),
        "mean_nominal_target_speed_mps": float(np.mean(targets)),
        "mean_actual_minus_nominal_mps": float(np.mean(np.asarray(speeds) - np.asarray(targets))),
        "fraction_above_nominal_target": float(np.mean(np.asarray(speeds) > np.asarray(targets))),
        "sections": [_section(records, index, bins, total_length) for index in range(bins)],
    }


def build_report(paths: list[Path], bins: int = 10) -> dict[str, Any]:
    traces = [summarize_trace(path, bins=bins) for path in paths]
    return {
        "scope": "offline_recorded_speed_and_section_calibration",
        "controller_modified": False,
        "training_started": False,
        "traces": traces,
        "interpretation": [
            "Observed speeds come from the existing RL evaluation policy, not the conservative controller.",
            "Observed speeds are evidence for calibration, not automatically safe target speeds.",
            "No controller parameter was changed by this report.",
        ],
        "next_offline_action": "Compare section speed and curvature gaps, then choose bounded target-speed candidates for simulator-only testing.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.analysis, bins=args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "traces": len(report["traces"]), "bins": args.bins}, indent=2))


if __name__ == "__main__":
    main()

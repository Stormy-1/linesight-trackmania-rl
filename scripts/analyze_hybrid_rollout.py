"""Analyze a recorded telemetry rollout against a known centerline offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trackmania_rl.hybrid_rollout import analyze_rollout, load_rollout_csv
from trackmania_rl.hybrid_state import build_centerline_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centerline", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-radius", type=int, default=32)
    parser.add_argument("--finish-progress-m", type=float, default=None)
    parser.add_argument("--finish-distance-limit-m", type=float, default=4.0)
    parser.add_argument("--finish-elevation-tolerance-m", type=float, default=8.0)
    args = parser.parse_args()

    geometry = build_centerline_geometry(np.load(args.centerline))
    report = analyze_rollout(
        load_rollout_csv(args.rollout),
        geometry,
        search_radius=args.search_radius,
        finish_progress_m=args.finish_progress_m,
        finish_distance_limit_m=args.finish_distance_limit_m,
        finish_elevation_tolerance_m=args.finish_elevation_tolerance_m,
    )
    report["centerline"] = str(args.centerline)
    report["rollout"] = str(args.rollout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": report["samples"],
                "progress_gain_m": report["progress_gain_m"],
                "recovery_rate": report["recovery_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

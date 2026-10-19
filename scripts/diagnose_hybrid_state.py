"""Generate an offline structured-state readiness report.

This script only reads centerline ``.npy`` files and optional aggregate
evaluation CSVs.  It does not launch TrackMania, load checkpoints, or claim
controller completion from aggregate evidence that contains no trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from trackmania_rl.hybrid_state import (
    STRUCTURED_STATE_SCHEMA_VERSION,
    build_centerline_geometry,
    curvature_target_speed_mps,
    sample_centerline_lookahead,
)


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def summarize_centerline(path: Path) -> dict[str, Any]:
    points = np.load(path)
    geometry = build_centerline_geometry(points)
    lookahead = sample_centerline_lookahead(geometry, 0.0, [10.0, 25.0, 50.0])
    target_speeds = np.asarray(
        [curvature_target_speed_mps(value) for value in geometry.point_curvature], dtype=np.float64
    )
    return {
        "file": path.name,
        "points": int(len(points)),
        "segments": int(len(geometry.segment_lengths)),
        "length_m": float(geometry.cumulative_lengths[-1]),
        "curvature": _finite_summary(geometry.point_curvature),
        "target_speed_mps": _finite_summary(target_speeds),
        "lookahead_shape": list(lookahead.points.shape),
        "finite_geometry": bool(
            np.isfinite(geometry.segment_lengths).all()
            and np.isfinite(geometry.point_curvature).all()
            and np.isfinite(lookahead.points).all()
        ),
    }


def summarize_evaluation_csv(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_track: dict[str, dict[str, Any]] = {}
    for track in sorted({row.get("track_name", "unknown") for row in rows}):
        track_rows = [row for row in rows if row.get("track_name", "unknown") == track]
        percentages = np.asarray([float(row["dist_pct"]) for row in track_rows], dtype=np.float64)
        finished = sum(row.get("finished", "").lower() == "true" for row in track_rows)
        by_track[track] = {
            "episodes": len(track_rows),
            "finished": int(finished),
            "finish_rate": float(finished / len(track_rows)) if track_rows else 0.0,
            "distance_pct": _finite_summary(percentages),
        }
    return {
        "file": path.name,
        "episodes": len(rows),
        "tracks": by_track,
        "trajectory_data_available": False,
        "note": "Aggregate evaluation CSV; insufficient for state-trajectory validation.",
    }


def build_report(maps_dir: Path, evaluation_csvs: list[Path]) -> dict[str, Any]:
    centerlines = sorted(maps_dir.glob("*.npy"))
    if not centerlines:
        raise ValueError(f"no centerline .npy files found in {maps_dir}")
    return {
        "schema_version": STRUCTURED_STATE_SCHEMA_VERSION,
        "scope": "offline_centerline_geometry_and_aggregate_evidence",
        "trackmania_launched": False,
        "training_started": False,
        "checkpoints_loaded": False,
        "centerlines": [summarize_centerline(path) for path in centerlines],
        "evaluations": [summarize_evaluation_csv(path) for path in evaluation_csvs if path.exists()],
        "state_trajectory_validation": {
            "status": "blocked_by_missing_recorded_pose_telemetry",
            "required_fields": ["timestamp_s", "position_xyz", "heading_xyz", "speed_mps"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", type=Path, default=Path("maps"))
    parser.add_argument("--eval-csv", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("eval_results/hybrid_state_diagnostics.json"))
    args = parser.parse_args()
    report = build_report(args.maps_dir, args.eval_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "centerlines": len(report["centerlines"]), "evaluations": len(report["evaluations"])}, indent=2))


if __name__ == "__main__":
    main()

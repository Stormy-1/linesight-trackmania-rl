"""Calibrate the offline hybrid simulator's finish gates.

The diagnostic deliberately stays inside the deterministic software-in-the-loop
path.  It reads saved ``.npy`` centerlines, adds a synthetic straight-line
finish probe, and reports whether projected progress, the strict 3D gate, or
the elevation-tolerant gate accepted each run.  It does not start TrackMania,
training, checkpoint evaluation, or live control.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

# Make the documented ``python scripts/...`` invocation work from the
# workspace root as well as ``python -m scripts...`` without installing the
# project or consulting anything outside the workspace.
if __package__ in (None, ""):
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

from trackmania_rl.hybrid_controller import (
    ConservativeCenterlineController,
    ConservativeControllerConfig,
)
from trackmania_rl.hybrid_simulator import (
    ClosedLoopConfig,
    ClosedLoopResult,
    simulate_closed_loop,
)
from trackmania_rl.hybrid_state import CenterlineGeometry, build_centerline_geometry


DEFAULT_MAPS_DIR = Path("maps")
DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_finish_gate_diagnostic.json")
DEFAULT_CENTERLINES = {
    "A01": "A01-Race_0.5m_cl2.npy",
    "A07": "A07-Race_0.5m_cl2.npy",
}

# These are deliberately explicit rather than generated from or written back to
# ClosedLoopConfig.  The first entry is the current default gate; the others
# are diagnostic comparisons only.
FINISH_GATE_MATRIX: tuple[dict[str, Any], ...] = (
    {
        "name": "default_strict_gate",
        "finish_distance_limit_m": 4.0,
        "finish_elevation_tolerance_m": 8.0,
        "is_default": True,
    },
    {
        "name": "tight_distance_gate",
        "finish_distance_limit_m": 2.0,
        "finish_elevation_tolerance_m": 8.0,
        "is_default": False,
    },
    {
        "name": "zero_elevation_tolerance_gate",
        "finish_distance_limit_m": 4.0,
        "finish_elevation_tolerance_m": 0.0,
        "is_default": False,
    },
    {
        "name": "wide_elevation_tolerance_gate",
        "finish_distance_limit_m": 4.0,
        "finish_elevation_tolerance_m": 12.0,
        "is_default": False,
    },
)

PROFILE_NAME = "default_conservative_controller"


def synthetic_straight_centerline(*, length_m: float = 80.0, spacing_m: float = 2.0) -> np.ndarray:
    """Return a small, deterministic straight centerline fixture.

    This API is intentionally independent of files so tests and other offline
    diagnostics can construct the same geometry without touching map data.
    """

    if not np.isfinite([length_m, spacing_m]).all() or length_m <= 0.0 or spacing_m <= 0.0:
        raise ValueError("length_m and spacing_m must be finite and positive")
    x = np.arange(0.0, length_m, spacing_m, dtype=np.float64)
    if len(x) == 0 or x[-1] < length_m:
        x = np.append(x, float(length_m))
    else:
        x[-1] = float(length_m)
    return np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))


def synthetic_straight_fixture(
    *, length_m: float = 80.0, spacing_m: float = 2.0, elevation_offset_m: float = 6.0
) -> tuple[CenterlineGeometry, np.ndarray, np.ndarray]:
    """Build the near-finish elevation probe used by :func:`build_report`.

    Starting just before the finish keeps the probe focused on gate geometry:
    it reaches projected progress while retaining a known elevation offset.
    """

    if not np.isfinite(float(elevation_offset_m)):
        raise ValueError("elevation_offset_m must be finite")
    geometry = build_centerline_geometry(
        synthetic_straight_centerline(length_m=length_m, spacing_m=spacing_m)
    )
    initial_position = geometry.points[-1].copy()
    initial_position[0] -= min(0.5, geometry.segment_lengths[-1] * 0.5)
    initial_position[1] += float(elevation_offset_m)
    return geometry, initial_position, geometry.segment_tangents[-1].copy()


def _load_geometry(path: Path) -> CenterlineGeometry:
    return build_centerline_geometry(np.load(path, allow_pickle=False))


def _resolve_centerlines(maps_dir: Path) -> list[tuple[str, Path]]:
    """Prefer the requested A01/A07 inputs, with a useful temporary-fixture fallback."""

    requested = [(label, maps_dir / filename) for label, filename in DEFAULT_CENTERLINES.items()]
    present = [(label, path) for label, path in requested if path.is_file()]
    if present:
        return present
    fallback = sorted(maps_dir.glob("*.npy"))
    if not fallback:
        raise ValueError(f"no A01/A07 or temporary .npy centerlines found in {maps_dir}")
    return [(path.stem, path) for path in fallback]


def _profile_config() -> ConservativeControllerConfig:
    return ConservativeControllerConfig()


def _run_metrics(result: ClosedLoopResult, gate: dict[str, Any]) -> dict[str, Any]:
    final_state = result.final_state
    final_horizontal = float(final_state.horizontal_distance_to_centerline_m)
    final_3d = float(final_state.distance_to_centerline_m)
    final_elevation = float(final_state.elevation_error_m)
    projected_progress = bool(result.progress_complete)
    strict_finish = bool(result.strict_3d_finished)
    tolerant_finish = bool(result.elevation_tolerant_finished)
    return {
        "gate": {
            "name": str(gate["name"]),
            "finish_distance_limit_m": float(gate["finish_distance_limit_m"]),
            "finish_elevation_tolerance_m": float(gate["finish_elevation_tolerance_m"]),
            "is_default": bool(gate["is_default"]),
        },
        "projected_progress_completion": projected_progress,
        "progress_complete": projected_progress,
        "strict_finish": strict_finish,
        "strict_3d_finish": strict_finish,
        "elevation_tolerant_finish": tolerant_finish,
        "termination_reason": str(result.termination_reason),
        "steps": int(len(result.steps)),
        "final_errors": {
            "horizontal_m": final_horizontal,
            "distance_3d_m": final_3d,
            "elevation_m": final_elevation,
            "absolute_elevation_m": abs(final_elevation),
        },
        # Flat aliases make the one-row-per-map/profile/gate output convenient
        # for CSV-like consumers while retaining the grouped representation.
        "final_horizontal_error_m": final_horizontal,
        "final_3d_error_m": final_3d,
        "final_elevation_error_m": final_elevation,
        "final_abs_elevation_error_m": abs(final_elevation),
        "final_progress_m": float(final_state.progress_m),
        "track_length_m": float(result.track_length_m),
    }


def _run_one(
    geometry: CenterlineGeometry,
    *,
    initial_position: np.ndarray,
    initial_heading: np.ndarray,
    controller: ConservativeCenterlineController,
    gate: dict[str, Any],
    max_steps: int,
    search_radius: int,
) -> dict[str, Any]:
    config = ClosedLoopConfig(
        max_steps=max_steps,
        search_radius=search_radius,
        finish_distance_limit_m=float(gate["finish_distance_limit_m"]),
        finish_elevation_tolerance_m=float(gate["finish_elevation_tolerance_m"]),
    )
    result = simulate_closed_loop(
        geometry,
        initial_position=initial_position,
        initial_heading=initial_heading,
        initial_speed_mps=0.0,
        controller=controller,
        config=config,
    )
    return _run_metrics(result, gate)


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(bool(row[field]) for row in rows) / len(rows)) if rows else 0.0


def _pass_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["profile"]), str(row["gate"]["name"])), []).append(row)
    return [
        {
            "profile": profile,
            "gate": gate,
            "evaluations": len(grouped[(profile, gate)]),
            "projected_progress_completion_rate": _rate(grouped[(profile, gate)], "projected_progress_completion"),
            "strict_finish_rate": _rate(grouped[(profile, gate)], "strict_finish"),
            "elevation_tolerant_finish_rate": _rate(grouped[(profile, gate)], "elevation_tolerant_finish"),
        }
        for profile, gate in sorted(grouped)
    ]


def build_report(
    maps_dir: Path = DEFAULT_MAPS_DIR,
    *,
    max_steps: int = 3000,
    search_radius: int = 64,
) -> dict[str, Any]:
    """Run the finish-gate matrix against A01/A07 and the synthetic probe."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if search_radius < 0:
        raise ValueError("search_radius must be non-negative")
    maps_dir = Path(maps_dir)
    defaults_before = ClosedLoopConfig()
    controller_config = _profile_config()
    rows: list[dict[str, Any]] = []
    map_reports: list[dict[str, Any]] = []

    synthetic_geometry, synthetic_position, synthetic_heading = synthetic_straight_fixture()
    synthetic_runs: list[dict[str, Any]] = []
    for gate in FINISH_GATE_MATRIX:
        metrics = _run_one(
            synthetic_geometry,
            initial_position=synthetic_position,
            initial_heading=synthetic_heading,
            controller=ConservativeCenterlineController(controller_config),
            gate=gate,
            max_steps=max_steps,
            search_radius=search_radius,
        )
        row = {"map": "synthetic_straight", "profile": PROFILE_NAME, **metrics}
        rows.append(row)
        synthetic_runs.append(metrics)
    map_reports.append(
        {
            "map": "synthetic_straight",
            "source": "synthetic_straight_fixture()",
            "length_m": float(synthetic_geometry.cumulative_lengths[-1]),
            "initial_elevation_offset_m": 6.0,
            "runs": synthetic_runs,
        }
    )

    for label, path in _resolve_centerlines(maps_dir):
        geometry = _load_geometry(path)
        runs: list[dict[str, Any]] = []
        for gate in FINISH_GATE_MATRIX:
            metrics = _run_one(
                geometry,
                initial_position=geometry.points[0],
                initial_heading=geometry.segment_tangents[0],
                controller=ConservativeCenterlineController(controller_config),
                gate=gate,
                max_steps=max_steps,
                search_radius=search_radius,
            )
            row = {"map": label, "profile": PROFILE_NAME, "source": path.name, **metrics}
            rows.append(row)
            runs.append(metrics)
        map_reports.append(
            {
                "map": label,
                "source": path.name,
                "length_m": float(geometry.cumulative_lengths[-1]),
                "runs": runs,
            }
        )

    defaults_after = ClosedLoopConfig()
    rates = _pass_rates(rows)
    report = {
        "scope": "offline_hybrid_finish_gate_calibration_diagnostic",
        "offline_only": True,
        "safety": {
            "offline_only": True,
            "trackmania_launched": False,
            "training_started": False,
            "checkpoints_loaded": False,
            "live_control_used": False,
            "no_trackmania": True,
            "no_training": True,
            "no_checkpoint": True,
            "no_live_control": True,
        },
        "trackmania_launched": False,
        "training_started": False,
        "checkpoints_loaded": False,
        "live_control_used": False,
        "controller_defaults_modified": defaults_before != defaults_after,
        "simulator": {
            "dt_s": float(defaults_before.dt_s),
            "max_steps": int(max_steps),
            "search_radius_segments": int(search_radius),
            "map_match_horizontal_only": bool(defaults_before.map_match_horizontal_only),
            "initial_speed_mps": 0.0,
        },
        "default_closed_loop_config": asdict(defaults_before),
        "profiles": {
            PROFILE_NAME: {
                "name": PROFILE_NAME,
                "controller": "ConservativeCenterlineController",
                "config": asdict(controller_config),
            }
        },
        "gate_matrix": [dict(gate) for gate in FINISH_GATE_MATRIX],
        "maps": map_reports,
        "results": rows,
        "finish_gate_pass_rates": rates,
        "summary": {
            "map_count": len(map_reports),
            "evaluations": len(rows),
            "projected_progress_completion_count": sum(bool(row["projected_progress_completion"]) for row in rows),
            "strict_finish_count": sum(bool(row["strict_finish"]) for row in rows),
            "elevation_tolerant_finish_count": sum(bool(row["elevation_tolerant_finish"]) for row in rows),
            "finish_gate_pass_rates": rates,
        },
        "limitations": [
            "The deterministic simulator is a bounded kinematic model, not TrackMania physics.",
            "Projected progress is a map-matching result and does not by itself prove a finish.",
            "The synthetic straight case is a near-finish elevation-offset probe, not a live run.",
        ],
    }
    # Keep this assertion local so a future diagnostic edit cannot silently
    # change the simulator defaults while still producing a report.
    if defaults_before != defaults_after:
        raise AssertionError("finish-gate diagnostic modified ClosedLoopConfig defaults")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--search-radius", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.maps_dir, max_steps=args.max_steps, search_radius=args.search_radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evaluations": report["summary"]["evaluations"],
                "strict_finish_count": report["summary"]["strict_finish_count"],
                "elevation_tolerant_finish_count": report["summary"]["elevation_tolerant_finish_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

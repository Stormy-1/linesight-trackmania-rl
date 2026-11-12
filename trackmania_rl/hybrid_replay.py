"""Deterministic offline replay of deliberately disturbed rollout poses.

This module is the small seam between recorded :class:`RolloutSample` values
and the pure closed-loop simulator.  It does not import TrackMania, training,
or checkpoint code.  Disturbances are applied in the same signed lateral
coordinate convention as :mod:`trackmania_rl.hybrid_state` and every replay
returns JSON-safe summary metrics for report scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping

import numpy as np

from trackmania_rl.hybrid_controller import ConservativeCenterlineController, ControllerCommand
from trackmania_rl.hybrid_rollout import RolloutSample
from trackmania_rl.hybrid_simulator import ClosedLoopConfig, ClosedLoopResult, simulate_closed_loop
from trackmania_rl.hybrid_state import (
    CenterlineGeometry,
    TrackRelativeState,
    make_track_relative_state,
    project_to_centerline,
)


# Projection and finish-gate calculations can differ by a few floating-point
# micrometres at the last centerline sample.  Treat only larger reverse motion
# as a meaningful backward-progress event in replay metrics.
BACKWARD_PROGRESS_TOLERANCE_M = 1e-5
STABLE_REJOIN_CONSECUTIVE_SAMPLES = 3


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _validate_section_bins(section_bins: int) -> int:
    if not isinstance(section_bins, int) or isinstance(section_bins, bool) or section_bins <= 0:
        raise ValueError("section_bins must be a positive integer")
    return section_bins


@dataclass(frozen=True)
class DisturbanceSpec:
    """One deterministic pose perturbation in track-relative coordinates."""

    name: str
    lateral_offset_m: float = 0.0
    heading_offset_rad: float = 0.0
    speed_delta_mps: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        for field_name in ("lateral_offset_m", "heading_offset_rad", "speed_delta_mps"):
            _finite(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class DisturbedPose:
    """A replay-ready pose plus its source and projected track metadata."""

    source_index: int
    timestamp_s: float
    progress_m: float
    section_bin: int
    position: np.ndarray
    heading: np.ndarray
    speed_mps: float
    disturbance_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_index, int) or isinstance(self.source_index, bool) or self.source_index < 0:
            raise ValueError("source_index must be a non-negative integer")
        _finite(self.timestamp_s, "timestamp_s")
        _finite(self.progress_m, "progress_m")
        if not isinstance(self.section_bin, int) or isinstance(self.section_bin, bool) or self.section_bin < 0:
            raise ValueError("section_bin must be a non-negative integer")
        for value, name in ((self.position, "position"), (self.heading, "heading")):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must contain three finite values")
        if _finite(self.speed_mps, "speed_mps") < 0.0:
            raise ValueError("speed_mps must be non-negative")
        if not isinstance(self.disturbance_name, str) or not self.disturbance_name.strip():
            raise ValueError("disturbance_name must be a non-empty string")

    @property
    def name(self) -> str:
        """Short alias useful to report scripts."""

        return self.disturbance_name

    @property
    def disturbed_position(self) -> np.ndarray:
        return self.position

    @property
    def disturbed_heading(self) -> np.ndarray:
        return self.heading

    @property
    def disturbed_speed_mps(self) -> float:
        return self.speed_mps


@dataclass(frozen=True)
class ReplayOutcome:
    """Closed-loop result, initial disturbed pose, and JSON-safe metrics."""

    result: ClosedLoopResult
    initial_pose: DisturbedPose
    metrics: dict[str, Any]

    @property
    def closed_loop_result(self) -> ClosedLoopResult:
        return self.result

    @property
    def disturbed_pose(self) -> DisturbedPose:
        return self.initial_pose


@dataclass(frozen=True)
class MidRolloutInjection:
    """Apply one disturbance immediately before the indexed controller command."""

    step_index: int
    disturbance: DisturbanceSpec

    def __post_init__(self) -> None:
        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool) or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if not isinstance(self.disturbance, DisturbanceSpec):
            raise TypeError("disturbance must be a DisturbanceSpec")


@dataclass(frozen=True)
class MidRolloutOutcome:
    """The uninterrupted rollout, injection snapshots, and suffix diagnostics."""

    result: ClosedLoopResult
    injection: MidRolloutInjection
    pre_injection_state: TrackRelativeState
    post_injection_state: TrackRelativeState
    disturbed_pose: DisturbedPose
    metrics: dict[str, Any]


class InjectionNotReached(RuntimeError):
    """The simulator terminated before the requested injection step."""

    def __init__(self, result: ClosedLoopResult, step_index: int) -> None:
        self.result = result
        self.step_index = step_index
        super().__init__(
            f"injection step {step_index} was not reached: "
            f"rollout terminated after {len(result.steps)} steps ({result.termination_reason})"
        )


def _sample_progress(sample: RolloutSample, geometry: CenterlineGeometry) -> float:
    segment, fraction, _, _, _ = project_to_centerline(sample.position, geometry)
    return float(geometry.cumulative_lengths[segment] + fraction * geometry.segment_lengths[segment])


def select_source_samples(
    samples: Iterable[RolloutSample],
    geometry: CenterlineGeometry,
    target_progress_m: float,
    *,
    count: int = 1,
) -> list[tuple[int, RolloutSample]]:
    """Select source samples nearest to a requested projected progress.

    Ties are resolved by source index, making selection stable across runs and
    independent of the order used by a report script's later processing.
    """

    target = _finite(target_progress_m, "target_progress_m")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    indexed = list(enumerate(samples))
    if not indexed:
        raise ValueError("samples must contain at least one sample")
    scored = sorted(
        ((abs(_sample_progress(sample, geometry) - target), index, sample) for index, sample in indexed),
        key=lambda item: (item[0], item[1]),
    )
    return [(index, sample) for _, index, sample in scored[:count]]


def _section_bin(progress_m: float, track_length_m: float, section_bins: int) -> int:
    fraction = min(max(progress_m / track_length_m, 0.0), 1.0)
    return min(int(fraction * section_bins), section_bins - 1)


def disturb_sample(
    sample: RolloutSample,
    geometry: CenterlineGeometry,
    disturbance: DisturbanceSpec,
    source_index: int,
    section_bins: int = 10,
    *,
    reference_state: TrackRelativeState | None = None,
    horizontal_only: bool = False,
    search_radius: int | None = None,
) -> DisturbedPose:
    """Apply a lateral, yaw, and speed disturbance to one rollout sample.

    A positive lateral offset produces a positive ``lateral_error_m`` under the
    existing state convention.  For a +X tangent this therefore moves toward
    -Z, matching ``make_track_relative_state`` exactly.

    A supplied reference state fixes the displacement frame to the caller's
    matched tangent and anchors an optional local rematch. Defaults retain the
    historical global 3D projection and non-negative speed clamp.
    """

    if not isinstance(sample, RolloutSample):
        raise TypeError("sample must be a RolloutSample")
    if not isinstance(geometry, CenterlineGeometry):
        raise TypeError("geometry must be a CenterlineGeometry")
    if not isinstance(disturbance, DisturbanceSpec):
        raise TypeError("disturbance must be a DisturbanceSpec")
    if not isinstance(source_index, int) or isinstance(source_index, bool) or source_index < 0:
        raise ValueError("source_index must be a non-negative integer")
    if reference_state is not None and not isinstance(reference_state, TrackRelativeState):
        raise TypeError("reference_state must be a TrackRelativeState or None")
    if not isinstance(horizontal_only, bool):
        raise ValueError("horizontal_only must be a boolean")
    if search_radius is not None and (
        not isinstance(search_radius, int) or isinstance(search_radius, bool) or search_radius < 0
    ):
        raise ValueError("search_radius must be a non-negative integer or None")
    bins = _validate_section_bins(section_bins)

    position = np.asarray(sample.position, dtype=np.float64)
    heading = np.asarray(sample.heading, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("sample position must contain three finite values")
    if heading.shape != (3,) or not np.isfinite(heading).all():
        raise ValueError("sample heading must contain three finite values")
    timestamp = _finite(sample.timestamp_s, "sample timestamp_s")
    speed = _finite(sample.speed_mps, "sample speed_mps")

    state = reference_state if reference_state is not None else make_track_relative_state(
        position, heading, speed, geometry, horizontal_only=horizontal_only
    )
    tangent = np.asarray(state.tangent, dtype=np.float64)
    horizontal_norm = math.hypot(float(tangent[0]), float(tangent[2]))
    if horizontal_norm <= 1e-9:
        raise ValueError("track tangent must have a non-zero horizontal component")
    # n is the signed normal for lateral_error_m: tangent_z * offset_x -
    # tangent_x * offset_z.  It is intentionally not the usual left normal.
    lateral_normal = np.asarray([tangent[2], 0.0, -tangent[0]], dtype=np.float64) / horizontal_norm
    position = position + lateral_normal * float(disturbance.lateral_offset_m)

    heading_horizontal_norm = math.hypot(float(heading[0]), float(heading[2]))
    if heading_horizontal_norm <= 1e-9:
        raise ValueError("sample heading must have a non-zero horizontal component")
    heading_yaw = math.atan2(float(heading[2]), float(heading[0])) + float(disturbance.heading_offset_rad)
    heading = np.asarray(
        [
            heading_horizontal_norm * math.cos(heading_yaw),
            float(heading[1]),
            heading_horizontal_norm * math.sin(heading_yaw),
        ],
        dtype=np.float64,
    )
    speed = max(0.0, speed + float(disturbance.speed_delta_mps))
    disturbed_state = make_track_relative_state(
        position,
        heading,
        speed,
        geometry,
        previous_segment=state.segment_index if reference_state is not None else None,
        search_radius=search_radius,
        horizontal_only=horizontal_only,
    )
    progress = float(disturbed_state.progress_m)
    return DisturbedPose(
        source_index=source_index,
        timestamp_s=timestamp,
        progress_m=progress,
        section_bin=_section_bin(progress, float(geometry.cumulative_lengths[-1]), bins),
        position=position.copy(),
        heading=heading.copy(),
        speed_mps=speed,
        disturbance_name=disturbance.name,
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _progress_accounting(progress: list[float]) -> tuple[int, float]:
    backward_distances = [max(previous - current, 0.0) for previous, current in zip(progress, progress[1:])]
    meaningful = [distance for distance in backward_distances if distance > BACKWARD_PROGRESS_TOLERANCE_M]
    return len(meaningful), float(sum(meaningful))


def _stability_event(record: Mapping[str, Any], *, kind: str, reason: str) -> dict[str, Any]:
    event: dict[str, Any] = {
        "kind": kind,
        "reason": reason,
        "step_index": int(record["step_index"]),
        "timestamp_s": float(record["timestamp_s"]),
        "progress_m": float(record["progress_m"]),
    }
    for field in (
        "lateral_error_m",
        "heading_error_rad",
        "horizontal_distance_to_centerline_m",
        "elevation_error_m",
    ):
        if field in record:
            event[field] = float(record[field])
    return event


def _recovery_stability_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    stable_rejoin_samples: int = STABLE_REJOIN_CONSECUTIVE_SAMPLES,
) -> dict[str, Any]:
    """Summarize stable rejoin and post-rejoin recovery episodes.

    This is additive to the historical first-completion heuristic.  It requires
    a short consecutive non-recovery window before calling a rejoin stable, so
    one flapping sample cannot be mistaken for a completed recovery.
    """

    if (
        not isinstance(stable_rejoin_samples, int)
        or isinstance(stable_rejoin_samples, bool)
        or stable_rejoin_samples <= 0
    ):
        raise ValueError("stable_rejoin_samples must be a positive integer")
    ordered = list(records)
    activation_index = next(
        (index for index, record in enumerate(ordered) if bool(record.get("recovery", False))),
        None,
    )
    stable_index: int | None = None
    if activation_index is not None:
        false_streak = 0
        for index in range(activation_index + 1, len(ordered)):
            if bool(ordered[index].get("recovery", False)):
                false_streak = 0
                continue
            false_streak += 1
            if false_streak >= stable_rejoin_samples:
                stable_index = index
                break

    stable_event = None
    latency_samples = None
    latency_s = None
    if stable_index is not None and activation_index is not None:
        stable_event = _stability_event(
            ordered[stable_index], kind="stable_rejoin", reason="tracking_rejoined_stably"
        )
        stable_event["stable_rejoin_samples"] = stable_rejoin_samples
        latency_samples = int(ordered[stable_index]["step_index"]) - int(ordered[activation_index]["step_index"])
        latency_s = float(ordered[stable_index]["timestamp_s"]) - float(ordered[activation_index]["timestamp_s"])

    first_redivergence = None
    redivergence_count = 0
    if stable_index is not None:
        was_recovering = False
        for index in range(stable_index + 1, len(ordered)):
            recovering = bool(ordered[index].get("recovery", False))
            if recovering and not was_recovering:
                redivergence_count += 1
                if first_redivergence is None:
                    first_redivergence = _stability_event(
                        ordered[index], kind="redivergence", reason="post_rejoin_recovery"
                    )
            was_recovering = recovering

    return {
        "stable_rejoin_required_samples": stable_rejoin_samples,
        "stable_rejoin": stable_event is not None,
        "stable_rejoin_event": stable_event,
        "first_redivergence_event": first_redivergence,
        "redivergence_count": redivergence_count,
        "recovery_latency_samples": latency_samples,
        "recovery_latency_s": latency_s,
    }


def _replay_metrics(result: ClosedLoopResult, initial_progress_m: float) -> dict[str, Any]:
    states = [step.state for step in result.steps]
    states.append(result.final_state)
    lateral = [abs(float(state.lateral_error_m)) for state in states]
    horizontal = [float(state.horizontal_distance_to_centerline_m) for state in states]
    heading = [abs(float(state.heading_error_rad)) for state in states]
    progress = [float(initial_progress_m)] + [float(state.progress_m) for state in states]
    backward_count, backward_distance = _progress_accounting(progress)
    progress_start = progress[0]
    progress_final = float(result.final_state.progress_m)
    progress_gain = progress_final - progress_start
    progress_fraction = progress_final / max(result.track_length_m, 1e-9)
    events = result.divergence_recovery
    records = [
        {
            "step_index": step.step_index,
            "timestamp_s": step.timestamp_s,
            "progress_m": step.state.progress_m,
            "lateral_error_m": step.state.lateral_error_m,
            "heading_error_rad": step.state.heading_error_rad,
            "horizontal_distance_to_centerline_m": step.state.horizontal_distance_to_centerline_m,
            "elevation_error_m": step.state.elevation_error_m,
            "recovery": step.command.recovery,
        }
        for step in result.steps
    ]
    stability = _recovery_stability_metrics(records)
    activation = events["first_recovery_activation"]
    completion = events["first_recovery_completion"]
    recovery_success: bool | None
    if activation is None:
        recovery_success = None
    else:
        recovery_success = bool(completion is not None and backward_count == 0)

    completion_flags = {
        "finished": bool(result.finished),
        "strict_3d_finished": bool(result.strict_3d_finished),
        "elevation_tolerant_finished": bool(result.elevation_tolerant_finished),
        "progress_complete": bool(result.progress_complete),
        "termination_reason": str(result.termination_reason),
    }
    metrics: dict[str, Any] = {
        "recovery_activation": activation,
        "recovery_completion": completion,
        "recovery_activated": bool(activation is not None),
        "recovery_completed": bool(completion is not None),
        "recovery_success": recovery_success,
        **stability,
        "mean_abs_lateral_error_m": _mean(lateral),
        "max_abs_lateral_error_m": max(lateral, default=0.0),
        "final_abs_lateral_error_m": abs(float(result.final_state.lateral_error_m)),
        "mean_horizontal_error_m": _mean(horizontal),
        "max_horizontal_error_m": max(horizontal, default=0.0),
        "final_horizontal_error_m": float(result.final_state.horizontal_distance_to_centerline_m),
        "mean_horizontal_distance_to_centerline_m": _mean(horizontal),
        "max_horizontal_distance_to_centerline_m": max(horizontal, default=0.0),
        "final_horizontal_distance_to_centerline_m": float(result.final_state.horizontal_distance_to_centerline_m),
        "mean_abs_heading_error_rad": _mean(heading),
        "max_abs_heading_error_rad": max(heading, default=0.0),
        "final_abs_heading_error_rad": abs(float(result.final_state.heading_error_rad)),
        "progress_start_m": progress_start,
        "progress_final_m": progress_final,
        "progress_gain_m": progress_gain,
        "progress_fraction": float(progress_fraction),
        "start_progress_m": progress_start,
        "final_progress_m": progress_final,
        "backward_progress_event_count": int(backward_count),
        "backward_progress_distance_m": backward_distance,
        "progress_continuity": bool(backward_count == 0),
        "completion": completion_flags,
        "completion_flags": completion_flags,
        "finished": completion_flags["finished"],
        "strict_3d_finished": completion_flags["strict_3d_finished"],
        "elevation_tolerant_finished": completion_flags["elevation_tolerant_finished"],
        "progress_complete": completion_flags["progress_complete"],
        "termination_reason": completion_flags["termination_reason"],
        "elapsed_s": float(result.elapsed_s),
        "steps": len(result.steps),
        "section_metrics": result.section_metrics,
        "recovery": {
            "activation": activation,
            "completion": completion,
            "success": recovery_success,
            "samples": int(result.recovery_samples),
            **stability,
        },
        "progress": {
            "start_m": progress_start,
            "final_m": progress_final,
            "gain_m": progress_gain,
            "track_length_m": float(result.track_length_m),
            "fraction": float(progress_fraction),
            "backward_event_count": int(backward_count),
            "backward_distance_m": backward_distance,
            "continuity": bool(backward_count == 0),
        },
    }
    return metrics


def summarize_rollout_suffix(
    result: ClosedLoopResult,
    start_step_index: int,
    *,
    dt_s: float,
) -> dict[str, Any]:
    """Return historical replay metrics from a recorded global step onward.

    The first suffix observation is the progress baseline, so a displacement
    into that observation is not counted as backward driving. Event indices
    and timestamps stay global; elapsed time covers only suffix updates.
    """

    if not isinstance(result, ClosedLoopResult):
        raise TypeError("result must be a ClosedLoopResult")
    if not isinstance(start_step_index, int) or isinstance(start_step_index, bool) or start_step_index < 0:
        raise ValueError("start_step_index must be a non-negative integer")
    dt = _finite(dt_s, "dt_s")
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")
    start = next(
        (index for index, step in enumerate(result.steps) if step.step_index == start_step_index),
        None,
    )
    if start is None:
        raise ValueError("start_step_index must identify a recorded step in a nonempty rollout")
    suffix = result.steps[start:]
    suffix_result = replace(result, steps=suffix, elapsed_s=len(suffix) * dt)
    return _replay_metrics(suffix_result, suffix[0].state.progress_m)


def _injection_snapshot(
    state: TrackRelativeState,
    *,
    step_index: int,
    timestamp_s: float,
    command: ControllerCommand | None,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "step_index": step_index,
        "timestamp_s": timestamp_s,
        "segment_index": int(state.segment_index),
    }
    for field in (
        "progress_m",
        "lateral_error_m",
        "distance_to_centerline_m",
        "horizontal_distance_to_centerline_m",
        "elevation_error_m",
        "heading_error_rad",
        "speed_mps",
        "local_curvature",
    ):
        snapshot[field] = float(getattr(state, field))
    snapshot["recovery"] = bool(command.recovery) if command is not None else None
    for field in ("steering", "throttle", "brake", "target_speed_mps"):
        snapshot[field] = float(getattr(command, field)) if command is not None else None
    return snapshot


def replay_disturbed_pose(
    geometry: CenterlineGeometry | DisturbedPose,
    disturbed_pose: DisturbedPose | CenterlineGeometry,
    *,
    controller: ConservativeCenterlineController | None = None,
    config: ClosedLoopConfig | None = None,
) -> ReplayOutcome:
    """Replay one disturbed pose through the deterministic closed-loop model.

    The normal spelling is ``replay_disturbed_pose(geometry, pose, ...)``;
    accepting ``(pose, geometry, ...)`` as well keeps the helper convenient for
    report code that starts from a pose returned by :func:`disturb_sample`.
    """

    if isinstance(geometry, DisturbedPose) and isinstance(disturbed_pose, CenterlineGeometry):
        geometry, disturbed_pose = disturbed_pose, geometry
    if not isinstance(geometry, CenterlineGeometry):
        raise TypeError("geometry must be a CenterlineGeometry")
    if not isinstance(disturbed_pose, DisturbedPose):
        raise TypeError("disturbed_pose must be a DisturbedPose")
    result = simulate_closed_loop(
        geometry,
        initial_position=disturbed_pose.position,
        initial_heading=disturbed_pose.heading,
        initial_speed_mps=disturbed_pose.speed_mps,
        controller=controller,
        config=config,
    )
    return ReplayOutcome(
        result=result,
        initial_pose=disturbed_pose,
        metrics=_replay_metrics(result, disturbed_pose.progress_m),
    )


def replay_midrollout_disturbance(
    geometry: CenterlineGeometry,
    *,
    initial_position: np.ndarray,
    initial_heading: np.ndarray,
    initial_speed_mps: float,
    injection: MidRolloutInjection,
    controller: ConservativeCenterlineController | None = None,
    config: ClosedLoopConfig | None = None,
) -> MidRolloutOutcome:
    """Inject once before a command in one uninterrupted simulator loop.

    The controller instance, map-match history, global clock, and total step
    budget survive the injection. A zero disturbance observes the target step
    without touching its pose. An out-of-bounds disturbed speed is rejected,
    including negative requested speeds (unlike historical ``disturb_sample``).
    Early termination raises :class:`InjectionNotReached` with the full result.

    Metrics include scalar pre/post snapshots, requested deltas and realized
    pose under ``injection``, historical suffix-only fields under
    ``post_rollout``, its ``recovery`` alias, and ``total_elapsed_s``.
    """

    if not isinstance(geometry, CenterlineGeometry):
        raise TypeError("geometry must be a CenterlineGeometry")
    if not isinstance(injection, MidRolloutInjection):
        raise TypeError("injection must be a MidRolloutInjection")
    active_config = config or ClosedLoopConfig()
    if injection.step_index >= active_config.max_steps:
        raise ValueError("injection step_index must be less than max_steps")
    disturbance = injection.disturbance
    requested = {
        field: float(getattr(disturbance, field))
        for field in ("lateral_offset_m", "heading_offset_rad", "speed_delta_mps")
    }
    modifies_pose = any(value != 0.0 for value in requested.values())
    pre_state: TrackRelativeState | None = None
    pose: DisturbedPose | None = None

    def inject(
        step_index: int,
        timestamp_s: float,
        position: np.ndarray,
        heading: np.ndarray,
        speed_mps: float,
        state: TrackRelativeState,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        nonlocal pre_state, pose
        if step_index != injection.step_index:
            return None
        pre_state = state
        if not modifies_pose:
            pose = DisturbedPose(
                source_index=step_index,
                timestamp_s=timestamp_s,
                progress_m=state.progress_m,
                section_bin=_section_bin(state.progress_m, float(geometry.cumulative_lengths[-1]), active_config.section_bins),
                position=position.copy(),
                heading=heading.copy(),
                speed_mps=speed_mps,
                disturbance_name=disturbance.name,
            )
            return None
        disturbed_speed = speed_mps + requested["speed_delta_mps"]
        if not math.isfinite(disturbed_speed) or not 0.0 <= disturbed_speed <= active_config.max_speed_mps:
            raise ValueError("disturbed speed must be within the simulator speed bounds")
        pose = disturb_sample(
            RolloutSample(timestamp_s, position, heading, speed_mps),
            geometry,
            disturbance,
            source_index=step_index,
            section_bins=active_config.section_bins,
            reference_state=state,
            horizontal_only=active_config.map_match_horizontal_only,
            search_radius=active_config.search_radius,
        )
        return pose.position, pose.heading, pose.speed_mps

    result = simulate_closed_loop(
        geometry,
        initial_position=initial_position,
        initial_heading=initial_heading,
        initial_speed_mps=initial_speed_mps,
        controller=controller,
        config=active_config,
        _pose_mutator=inject,
    )
    if pre_state is None or pose is None:
        raise InjectionNotReached(result, injection.step_index)
    post_step = result.steps[injection.step_index]
    post_state = post_step.state
    # Use the simulator's actual rematch, which retains its prior search window,
    # and the normalized heading that its pose validator passed to the command.
    realized_heading = pose.heading.copy()
    if modifies_pose:
        yaw = math.atan2(float(pose.heading[2]), float(pose.heading[0]))
        realized_heading = np.asarray([math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float64)
    pose = replace(
        pose,
        progress_m=post_state.progress_m,
        section_bin=_section_bin(post_state.progress_m, result.track_length_m, result.section_bins),
        heading=realized_heading,
        speed_mps=post_state.speed_mps,
    )
    post_metrics = summarize_rollout_suffix(result, injection.step_index, dt_s=active_config.dt_s)
    metrics = {
        "injection": {
            "step_index": injection.step_index,
            "timestamp_s": pose.timestamp_s,
            "disturbance_name": disturbance.name,
            "requested": requested,
            "realized_pose": {
                "position": pose.position.tolist(),
                "heading": pose.heading.tolist(),
                "speed_mps": float(pose.speed_mps),
                "progress_m": float(pose.progress_m),
                "section_bin": int(pose.section_bin),
            },
            "progress_jump_m": float(post_state.progress_m - pre_state.progress_m),
        },
        "pre_injection": _injection_snapshot(
            pre_state, step_index=injection.step_index, timestamp_s=pose.timestamp_s, command=None
        ),
        "post_injection": _injection_snapshot(
            post_state, step_index=injection.step_index, timestamp_s=pose.timestamp_s, command=post_step.command
        ),
        "post_rollout": post_metrics,
        "recovery": post_metrics["recovery"],
        "total_elapsed_s": float(result.elapsed_s),
    }
    return MidRolloutOutcome(result, injection, pre_state, post_state, pose, metrics)

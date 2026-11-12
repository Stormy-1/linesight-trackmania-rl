"""Pure planning and metric helpers for the staged ``gen_v3`` experiment.

This module deliberately does not launch TrackMania, mutate checkpoints, or
start training.  It keeps the Stage 1 map cycle and behavioral gates in one
small, testable place so the runtime configuration cannot silently fall back
to the failed six-map ``gen_v2`` curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


MapCycleEntry = tuple[str, str, str, bool, bool]
WHEEL_STATE_SIZE = 32


@dataclass(frozen=True)
class StageSpec:
    """A bounded gen_v3 stage and the map introduced by that stage."""

    name: str
    new_map: str
    map_path: str
    centerline_file: str


@dataclass(frozen=True)
class EvaluationGate:
    """Behavioral limits used at a bounded evaluation boundary."""

    episodes_per_map: int
    interval_frames: int
    stage_frame_budget: int
    replay_warmup_rollouts_per_map: int
    min_replay_memories_before_learning: int
    minimum_hock_completion_rate: float
    maximum_hock_timeout_rate: float
    minimum_new_map_completion_rate: float
    minimum_new_map_completion_improvement: float
    rollback_if_both_maps_timeout: bool = True


GEN_V3_STAGES = (
    StageSpec("stage1_hock_a01", "A01", '"Official Maps\\A01-Race.Challenge.Gbx"', "A01-Race_0.5m_cl2.npy"),
    StageSpec("stage2_hock_a03", "A03", '"Official Maps\\A03-Race.Challenge.Gbx"', "A03-Race_0.5m_cl2.npy"),
    StageSpec("stage3_hock_a05", "A05", '"Official Maps\\A05-Race.Challenge.Gbx"', "A05-Race_0.5m_cl2.npy"),
    StageSpec("stage4_hock_a07", "A07", '"Official Maps\\A07-Race.Challenge.Gbx"', "A07-Race_0.5m_cl2.npy"),
    StageSpec("stage5_hock_a11", "A11", '"Official Maps\\A11-Race.Challenge.Gbx"', "A11-Race_0.5m_cl2.npy"),
)

STAGE_1 = GEN_V3_STAGES[0]

GEN_V3_GATE = EvaluationGate(
    episodes_per_map=5,
    interval_frames=250_000,
    stage_frame_budget=1_000_000,
    replay_warmup_rollouts_per_map=8,
    min_replay_memories_before_learning=50_000,
    minimum_hock_completion_rate=0.80,
    maximum_hock_timeout_rate=0.20,
    minimum_new_map_completion_rate=0.20,
    minimum_new_map_completion_improvement=0.10,
)


def stage_map_cycle(stage: StageSpec = STAGE_1) -> list[Iterator[MapCycleEntry]]:
    """Return a replay-preserving cycle for Hockolicious and one new map.

    Exploratory entries fill replay.  Greedy entries are bounded evaluation
    passes and do not fill replay, preventing evaluation trajectories from
    contaminating the learning distribution.
    """

    hock_entry = ("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_0.5m_cl2.npy")
    new_entry = (stage.new_map, stage.map_path, stage.centerline_file)
    return [
        repeat((*hock_entry, True, True), 2),
        repeat((*new_entry, True, True), 2),
        repeat((*hock_entry, False, False), 1),
        repeat((*new_entry, False, False), 1),
    ]


def materialize_cycle(cycle: Iterable[Iterator[MapCycleEntry]]) -> list[MapCycleEntry]:
    """Materialize a finite configured cycle for tests and reproducibility."""

    entries: list[MapCycleEntry] = []
    for iterator in cycle:
        entries.extend(iterator)
    return entries


def map_cycle_signature(cycle: Iterable[Iterable[MapCycleEntry]]) -> tuple[MapCycleEntry, ...]:
    """Return a stable logical signature instead of ``repr(itertools.repeat)``."""

    return tuple(entry for group in cycle for entry in group)


def summarize_map_results(results: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    """Aggregate the behavioral metrics required by the gen_v3 handoff."""

    if not results:
        raise ValueError("At least one evaluation result is required")
    finished = sum(bool(result.get("finished", False)) for result in results)
    timed_out = sum(bool(result.get("timed_out", False)) for result in results)
    infrastructure_failures = sum(bool(result.get("infrastructure_failure", False)) for result in results)
    contact_failures = [
        int(result["contact_or_barrier_failures"])
        for result in results
        if result.get("contact_metric_available", False) and result.get("contact_or_barrier_failures") is not None
    ]
    completion = [float(result.get("completion_pct", 0.0)) for result in results]
    return {
        "episodes": len(results),
        "finish_rate": finished / len(results),
        "completion_pct_mean": sum(completion) / len(completion),
        "timeout_rate": timed_out / len(results),
        "infrastructure_failure_rate": infrastructure_failures / len(results),
        "contact_metric_available": bool(contact_failures),
        "contact_or_barrier_failures": sum(contact_failures) if contact_failures else None,
    }


def derive_episode_metrics(
    zone_centers: Sequence[object],
    rollout_results: Mapping[str, object],
    end_race_stats: Mapping[str, object],
    *,
    rollout_crashed: bool = False,
) -> dict[str, object]:
    """Normalize one rollout into the metrics used by the stage controller.

    Exact barrier collisions are not exposed by the current TMInterface
    telemetry, so contact metrics are explicitly marked unavailable rather
    than being reported as a misleading zero.
    """

    finish_zone_idx = max(1, len(zone_centers) - 1000)
    furthest_zone_idx = int(rollout_results.get("furthest_zone_idx", 0) or 0)
    finished = bool(end_race_stats.get("race_finished", False))
    infrastructure_failure = rollout_crashed or bool(end_race_stats.get("tmi_protection_cutoff", False))
    if finished:
        termination_reason = "finished"
        completion_pct = 100.0
    elif infrastructure_failure:
        termination_reason = "tmi_protection_timeout"
        completion_pct = min(100.0, 100.0 * furthest_zone_idx / finish_zone_idx)
    else:
        termination_reason = "rollout_timeout"
        completion_pct = min(100.0, 100.0 * furthest_zone_idx / finish_zone_idx)
    contact_proxy = summarize_wheel_contact_proxies(rollout_results.get("car_gear_and_wheels", []))
    return {
        "finished": finished,
        "completion_pct": completion_pct,
        "timed_out": not finished and not infrastructure_failure,
        "termination_reason": termination_reason,
        "infrastructure_failure": infrastructure_failure,
        "contact_metric_available": False,
        "contact_or_barrier_failures": None,
        "contact_proxy": contact_proxy,
    }


def derive_wheel_contact_proxies(
    car_gear_and_wheels: Sequence[float] | np.ndarray,
    previous_car_gear_and_wheels: Sequence[float] | np.ndarray | None = None,
) -> dict[str, object]:
    """Decode current wheel telemetry without claiming exact collisions.

    The game exposes wheel sliding, ground contact, and broad material groups,
    but not a barrier-collision event or impact impulse.  All returned values
    are therefore telemetry/proxy values, not exact crash counts.
    """

    current = np.asarray(car_gear_and_wheels, dtype=np.float32)
    if current.shape != (WHEEL_STATE_SIZE,) or not np.isfinite(current).all():
        raise ValueError("A normal wheel telemetry sample must contain 32 finite values")

    sliding = tuple(bool(value >= 0.5) for value in current[0:4])
    ground_contact = tuple(bool(value >= 0.5) for value in current[4:8])
    material_groups: list[int | None] = []
    for wheel in range(4):
        material = current[16 + 4 * wheel : 20 + 4 * wheel]
        if not np.isfinite(material).all() or np.any((material < -1e-5) | (material > 1.00001)):
            material_groups.append(None)
        elif np.all(material < 0.5):
            material_groups.append(4)
        elif np.count_nonzero(material >= 0.5) == 1:
            material_groups.append(int(np.argmax(material)))
        else:
            material_groups.append(None)

    contact_loss_count = None
    contact_gain_count = None
    material_transition_count = None
    if previous_car_gear_and_wheels is not None:
        previous = derive_wheel_contact_proxies(previous_car_gear_and_wheels)
        previous_contact = previous["ground_contact"]
        previous_material = previous["material_groups"]
        contact_loss_count = sum(old and not new for old, new in zip(previous_contact, ground_contact))
        contact_gain_count = sum(not old and new for old, new in zip(previous_contact, ground_contact))
        material_transition_count = sum(
            old is not None
            and new is not None
            and old != new
            and old_contact
            and new_contact
            for old, new, old_contact, new_contact in zip(previous_material, material_groups, previous_contact, ground_contact)
        )

    grounded_sliding = tuple(slide and contact for slide, contact in zip(sliding, ground_contact))
    return {
        "ground_contact": ground_contact,
        "sliding": sliding,
        "grounded_sliding": grounded_sliding,
        "material_groups": tuple(material_groups),
        "ground_contact_count": sum(ground_contact),
        "airborne_wheel_count": 4 - sum(ground_contact),
        "sliding_count": sum(sliding),
        "grounded_sliding_count": sum(grounded_sliding),
        "contact_loss_count": contact_loss_count,
        "contact_gain_count": contact_gain_count,
        "material_transition_count": material_transition_count,
        "exact_barrier_collisions": None,
    }


def summarize_wheel_contact_proxies(samples: Iterable[Sequence[float] | np.ndarray]) -> dict[str, object]:
    """Summarize valid wheel telemetry, ignoring the scalar finish sentinel."""

    summary = {
        "telemetry_samples": 0,
        "ground_contact_samples": 0,
        "airborne_wheel_samples": 0,
        "sliding_samples": 0,
        "grounded_sliding_samples": 0,
        "ground_contact_loss_count": 0,
        "ground_contact_gain_count": 0,
        "material_transition_count": 0,
        "exact_barrier_collisions": None,
    }
    previous = None
    for sample in samples:
        values = np.asarray(sample)
        if values.shape != (WHEEL_STATE_SIZE,) or not np.isfinite(values).all():
            continue
        metrics = derive_wheel_contact_proxies(values, previous)
        summary["telemetry_samples"] += 1
        summary["ground_contact_samples"] += metrics["ground_contact_count"]
        summary["airborne_wheel_samples"] += metrics["airborne_wheel_count"]
        summary["sliding_samples"] += metrics["sliding_count"]
        summary["grounded_sliding_samples"] += metrics["grounded_sliding_count"]
        summary["ground_contact_loss_count"] += metrics["contact_loss_count"] or 0
        summary["ground_contact_gain_count"] += metrics["contact_gain_count"] or 0
        summary["material_transition_count"] += metrics["material_transition_count"] or 0
        previous = values
    summary["proxy_available"] = summary["telemetry_samples"] > 0
    return summary


class StageController:
    """Bounded Stage 1 gatekeeper owned by the learner process."""

    def __init__(self, stage_start_frames: int, gate: EvaluationGate = GEN_V3_GATE):
        self.stage_start_frames = stage_start_frames
        self.gate = gate
        self.next_evaluation_frame = stage_start_frames + gate.interval_frames
        self.windows: dict[str, list[Mapping[str, object]]] = {}
        self.baseline_new_map_finish_rate: float | None = None
        self.last_metrics: dict[str, dict[str, float | int | bool | None]] | None = None
        self.decision: str | None = None
        self.reason: str | None = None

    def stage_frames(self, cumulative_frames: int) -> int:
        return max(0, cumulative_frames - self.stage_start_frames)

    def record_episode(self, map_name: str, metrics: Mapping[str, object], cumulative_frames: int, *, is_evaluation: bool) -> str | None:
        if self.decision is not None or not is_evaluation:
            return self.decision
        required = self.gate.episodes_per_map
        window = self.windows.setdefault(map_name, [])
        if len(window) < required:
            window.append(dict(metrics))
        if any(len(self.windows.get(name, [])) < required for name in ("hock", "A01")):
            return None

        hock_metrics = summarize_map_results(self.windows["hock"][:required])
        new_map_metrics = summarize_map_results(self.windows["A01"][:required])
        self.last_metrics = {"hock": hock_metrics, "A01": new_map_metrics}
        self.windows = {}

        baseline_was_missing = self.baseline_new_map_finish_rate is None
        infrastructure_failure = (
            new_map_metrics["infrastructure_failure_rate"] > 0
            or hock_metrics["infrastructure_failure_rate"] > 0
        )
        if infrastructure_failure:
            self.decision = "infrastructure_failure"
            self.reason = "evaluation window contained a TMInterface/game infrastructure failure"
        elif baseline_was_missing:
            if rollback_required(hock_metrics, new_map_metrics, self.gate):
                self.decision = "rollback"
                self.reason = initial_rollback_reason(hock_metrics, new_map_metrics, self.gate)
            else:
                self.baseline_new_map_finish_rate = float(new_map_metrics["finish_rate"])
        elif rollback_required(hock_metrics, new_map_metrics, self.gate):
            self.decision = "rollback"
            self.reason = "Hockolicious regression or simultaneous map timeouts"
        elif (
            cumulative_frames >= self.next_evaluation_frame
            and stage_accepted(self.baseline_new_map_finish_rate, hock_metrics, new_map_metrics, self.gate)
        ):
            self.decision = "accepted"
            self.reason = "new map improved while Hockolicious remained functional"

        if baseline_was_missing and self.baseline_new_map_finish_rate is not None:
            self.next_evaluation_frame = cumulative_frames + self.gate.interval_frames
        elif not baseline_was_missing and cumulative_frames >= self.next_evaluation_frame:
            self.next_evaluation_frame += self.gate.interval_frames
        return self.decision

    def budget_exhausted(self, cumulative_frames: int) -> bool:
        if self.decision is not None:
            return False
        if self.stage_frames(cumulative_frames) >= self.gate.stage_frame_budget:
            self.decision = "budget_exhausted"
            self.reason = "stage frame budget reached before acceptance"
            return True
        return False


def initial_rollback_reason(
    hock_metrics: Mapping[str, float | int],
    new_map_metrics: Mapping[str, float | int],
    gate: EvaluationGate = GEN_V3_GATE,
) -> str:
    """Explain why an initial behavioral window cannot establish a baseline."""

    if hock_metrics["timeout_rate"] == 1 and new_map_metrics["timeout_rate"] == 1:
        return "initial evaluation window unusable: Hockolicious and A01 timed out in every episode"
    return "initial evaluation window failed the Hockolicious completion/timeout gate"


def rollback_required(
    hock_metrics: Mapping[str, float | int],
    new_map_metrics: Mapping[str, float | int],
    gate: EvaluationGate = GEN_V3_GATE,
) -> bool:
    """Return whether the current stage should stop and restore its baseline."""

    both_timeout = hock_metrics["timeout_rate"] == 1 and new_map_metrics["timeout_rate"] == 1
    hock_regressed = (
        hock_metrics["finish_rate"] < gate.minimum_hock_completion_rate
        or hock_metrics["timeout_rate"] > gate.maximum_hock_timeout_rate
    )
    return (gate.rollback_if_both_maps_timeout and both_timeout) or hock_regressed


def stage_accepted(
    baseline_new_map_finish_rate: float,
    hock_metrics: Mapping[str, float | int],
    new_map_metrics: Mapping[str, float | int],
    gate: EvaluationGate = GEN_V3_GATE,
) -> bool:
    """Return whether a bounded evaluation can promote the current stage."""

    if rollback_required(hock_metrics, new_map_metrics, gate):
        return False
    return (
        new_map_metrics["finish_rate"] >= gate.minimum_new_map_completion_rate
        and new_map_metrics["finish_rate"] - baseline_new_map_finish_rate
        >= gate.minimum_new_map_completion_improvement
    )

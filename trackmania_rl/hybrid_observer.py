"""Deterministic shadow observer for predictive planner warnings.

The observer consumes already-computed :class:`PredictiveRiskFeatures` and
emits signal samples plus hysteresis events. It has no controller or action
interface by design, so it can be run beside a later real capture without
changing driving behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from trackmania_rl.hybrid_planner import PredictiveRiskFeatures

SIGNAL_NAMES = (
    "speed_margin",
    "lateral_acceleration_margin",
    "tangent_change",
    "forward_alignment_margin",
)


def _finite(value: object, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class PredictiveObserverConfig:
    """Thresholds and dwell counts for a non-intervening shadow observer."""

    name: str
    speed_margin_threshold_mps: float | None = None
    lateral_acceleration_margin_threshold_mps2: float | None = None
    tangent_change_threshold_rad: float | None = None
    forward_alignment_margin_threshold_rad: float | None = None
    activation_samples: int = 3
    release_samples: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        thresholds = (
            (self.speed_margin_threshold_mps, "speed_margin_threshold_mps"),
            (self.lateral_acceleration_margin_threshold_mps2, "lateral_acceleration_margin_threshold_mps2"),
            (self.tangent_change_threshold_rad, "tangent_change_threshold_rad"),
            (self.forward_alignment_margin_threshold_rad, "forward_alignment_margin_threshold_rad"),
        )
        if not any(value is not None for value, _ in thresholds):
            raise ValueError("at least one observer threshold is required")
        for value, field_name in thresholds:
            if value is not None:
                _finite(value, field_name)
        for value, field_name in (
            (self.activation_samples, "activation_samples"),
            (self.release_samples, "release_samples"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True)
class ObserverStep:
    """One shadow-observer sample; it contains no action or command."""

    sample_index: int
    timestamp_s: float
    active: bool
    signals: tuple[str, ...]
    signal_count: int


@dataclass(frozen=True)
class ObserverEvent:
    """One hysteresis-delimited predictive warning interval."""

    start_index: int
    end_index: int
    start_timestamp_s: float
    end_timestamp_s: float
    duration_s: float
    peak_signal_count: int
    signals: tuple[str, ...]


def observer_signals(features: PredictiveRiskFeatures, config: PredictiveObserverConfig) -> tuple[str, ...]:
    """Return active warning names for one feature vector."""

    if not isinstance(features, PredictiveRiskFeatures):
        raise TypeError("features must be PredictiveRiskFeatures")
    active: list[str] = []
    if config.speed_margin_threshold_mps is not None and (
        features.speed_margin_to_min_target_mps < config.speed_margin_threshold_mps
    ):
        active.append("speed_margin")
    if config.lateral_acceleration_margin_threshold_mps2 is not None and (
        features.lateral_acceleration_margin_mps2 < config.lateral_acceleration_margin_threshold_mps2
    ):
        active.append("lateral_acceleration_margin")
    if config.tangent_change_threshold_rad is not None and (
        features.max_abs_tangent_change_rad > config.tangent_change_threshold_rad
    ):
        active.append("tangent_change")
    if config.forward_alignment_margin_threshold_rad is not None and (
        features.one_step_forward_alignment_margin_rad < config.forward_alignment_margin_threshold_rad
    ):
        active.append("forward_alignment_margin")
    return tuple(active)


class PredictiveShadowObserver:
    """Track warning events using activation and release hysteresis."""

    def __init__(self, config: PredictiveObserverConfig):
        if not isinstance(config, PredictiveObserverConfig):
            raise TypeError("config must be PredictiveObserverConfig")
        self.config = config
        self.steps: list[ObserverStep] = []
        self.events: list[ObserverEvent] = []
        self._active = False
        self._pending_active: list[ObserverStep] = []
        self._active_steps: list[ObserverStep] = []
        self._pending_release: list[ObserverStep] = []

    def observe(self, sample_index: int, timestamp_s: float, features: PredictiveRiskFeatures) -> ObserverStep:
        if not isinstance(sample_index, int) or isinstance(sample_index, bool) or sample_index < 0:
            raise ValueError("sample_index must be a non-negative integer")
        timestamp = _finite(timestamp_s, "timestamp_s")
        signals = observer_signals(features, self.config)
        step = ObserverStep(sample_index, timestamp, self._active, signals, len(signals))
        if self._active:
            if signals:
                self._active_steps.append(step)
                self._pending_release.clear()
            else:
                self._pending_release.append(step)
                if len(self._pending_release) >= self.config.release_samples:
                    self._close_event()
                    step = ObserverStep(sample_index, timestamp, False, signals, len(signals))
        else:
            if signals:
                self._pending_active.append(step)
                if len(self._pending_active) >= self.config.activation_samples:
                    self._active = True
                    pending_steps = list(self._pending_active)
                    self._pending_active.clear()
                    self._pending_release.clear()
                    step = ObserverStep(sample_index, timestamp, True, signals, len(signals))
                    pending_steps[-1] = step
                    self._active_steps = pending_steps
            else:
                self._pending_active.clear()
        self.steps.append(step)
        return step

    def _close_event(self) -> None:
        if not self._active_steps:
            self._active = False
            self._pending_release.clear()
            return
        start = self._active_steps[0]
        end = self._active_steps[-1]
        signals = tuple(sorted({signal for step in self._active_steps for signal in step.signals}))
        self.events.append(
            ObserverEvent(
                start_index=start.sample_index,
                end_index=end.sample_index,
                start_timestamp_s=start.timestamp_s,
                end_timestamp_s=end.timestamp_s,
                duration_s=max(0.0, end.timestamp_s - start.timestamp_s),
                peak_signal_count=max(step.signal_count for step in self._active_steps),
                signals=signals,
            )
        )
        self._active = False
        self._active_steps.clear()
        self._pending_release.clear()

    def finish(self) -> tuple[ObserverEvent, ...]:
        """Close a trailing event and return all completed events."""

        if self._active:
            self._close_event()
        return tuple(self.events)

    def report(self) -> dict[str, Any]:
        """Return JSON-safe observer diagnostics without action fields."""

        self.finish()
        return {
            "config": {
                "name": self.config.name,
                "speed_margin_threshold_mps": self.config.speed_margin_threshold_mps,
                "lateral_acceleration_margin_threshold_mps2": self.config.lateral_acceleration_margin_threshold_mps2,
                "tangent_change_threshold_rad": self.config.tangent_change_threshold_rad,
                "forward_alignment_margin_threshold_rad": self.config.forward_alignment_margin_threshold_rad,
                "activation_samples": self.config.activation_samples,
                "release_samples": self.config.release_samples,
            },
            "sample_count": len(self.steps),
            "active_sample_count": sum(step.active for step in self.steps),
            "events": [
                {
                    "start_index": event.start_index,
                    "end_index": event.end_index,
                    "start_timestamp_s": event.start_timestamp_s,
                    "end_timestamp_s": event.end_timestamp_s,
                    "duration_s": event.duration_s,
                    "peak_signal_count": event.peak_signal_count,
                    "signals": list(event.signals),
                }
                for event in self.events
            ],
        }

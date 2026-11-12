"""Parser for opt-in pose telemetry joined with discrete controller actions."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from trackmania_rl.hybrid_rollout import RolloutSample, parse_rollout_rows
from trackmania_rl.hybrid_telemetry import ACTION_TELEMETRY_COLUMNS, RANDOMIZED_ACTION_TELEMETRY_COLUMNS

ACTION_ROLLOUT_COLUMNS = ACTION_TELEMETRY_COLUMNS


@dataclass(frozen=True)
class ActionTelemetrySample:
    """One pose sample and the action selected for that sample."""

    telemetry: RolloutSample
    action_idx: int
    action_accelerate: bool
    action_brake: bool
    action_left: bool
    action_right: bool
    action_was_greedy: bool
    action_is_randomized: bool | None = None

    @property
    def action_inputs(self) -> tuple[bool, bool, bool, bool]:
        return (
            self.action_accelerate,
            self.action_brake,
            self.action_left,
            self.action_right,
        )


def _parse_bool(row_number: int, row: Mapping[str, str], column: str) -> bool:
    value = row.get(column)
    if value is None:
        raise ValueError(f"row {row_number}: {column} is required")
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"row {row_number}: {column} must be boolean")


def _parse_action_idx(row_number: int, row: Mapping[str, str]) -> int:
    value = row.get("action_idx")
    try:
        action_idx = int(value) if value is not None else -1
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: action_idx must be an integer") from exc
    if action_idx < 0:
        raise ValueError(f"row {row_number}: action_idx must be non-negative")
    return action_idx


def parse_action_rollout_rows(rows: Iterable[Mapping[str, str]]) -> list[ActionTelemetrySample]:
    """Parse and validate action-extended rollout rows."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("action rollout must contain at least one sample")
    base_samples = parse_rollout_rows(materialized)
    has_randomization_column = any("action_is_randomized" in row for row in materialized)
    if has_randomization_column and not all("action_is_randomized" in row for row in materialized):
        raise ValueError("action_is_randomized must be present on every row")
    parsed: list[ActionTelemetrySample] = []
    for row_number, (row, telemetry) in enumerate(zip(materialized, base_samples), start=2):
        action_inputs = {
            "accelerate": _parse_bool(row_number, row, "action_accelerate"),
            "brake": _parse_bool(row_number, row, "action_brake"),
            "left": _parse_bool(row_number, row, "action_left"),
            "right": _parse_bool(row_number, row, "action_right"),
        }
        if action_inputs["left"] and action_inputs["right"]:
            raise ValueError(f"row {row_number}: left and right cannot both be active")
        parsed.append(
            ActionTelemetrySample(
                telemetry=telemetry,
                action_idx=_parse_action_idx(row_number, row),
                action_accelerate=action_inputs["accelerate"],
                action_brake=action_inputs["brake"],
                action_left=action_inputs["left"],
                action_right=action_inputs["right"],
                action_was_greedy=_parse_bool(row_number, row, "action_was_greedy"),
                action_is_randomized=(
                    _parse_bool(row_number, row, "action_is_randomized") if has_randomization_column else None
                ),
            )
        )
    return parsed


def load_action_rollout_csv(path: str | Path) -> list[ActionTelemetrySample]:
    """Load an action-extended rollout and reject legacy-only files."""

    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in ACTION_ROLLOUT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"action rollout CSV is missing required columns: {', '.join(missing)}")
        return parse_action_rollout_rows(reader)


def load_randomized_action_rollout_csv(path: str | Path) -> list[ActionTelemetrySample]:
    """Load a capture that includes an explicit intervention provenance flag."""

    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in RANDOMIZED_ACTION_TELEMETRY_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"randomized action rollout CSV is missing required columns: {', '.join(missing)}")
        samples = parse_action_rollout_rows(reader)
    if any(sample.action_is_randomized is None for sample in samples):
        raise ValueError("randomized action rollout requires action_is_randomized on every row")
    return samples

"""Optional CSV writer for privileged rollout telemetry.

The writer is inert unless a caller explicitly constructs it.  The live
collector can therefore remain unchanged during normal training, while a
deliberately planned visible run can export the pose fields required by the
offline hybrid-state analyzer.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Self, TextIO

import numpy as np

TELEMETRY_COLUMNS = (
    "timestamp_s",
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "heading_x",
    "heading_y",
    "heading_z",
    "speed_mps",
)

ACTION_TELEMETRY_COLUMNS = TELEMETRY_COLUMNS + (
    "action_idx",
    "action_accelerate",
    "action_brake",
    "action_left",
    "action_right",
    "action_was_greedy",
)
RANDOMIZED_ACTION_TELEMETRY_COLUMNS = ACTION_TELEMETRY_COLUMNS + ("action_is_randomized",)


class HybridTelemetryWriter:
    """Write validated pose samples in the Phase 1 rollout schema."""

    def __init__(
        self,
        path: str | Path,
        *,
        include_action_columns: bool = False,
        include_randomization_columns: bool = False,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(include_action_columns, bool):
            raise TypeError("include_action_columns must be a boolean")
        if not isinstance(include_randomization_columns, bool):
            raise TypeError("include_randomization_columns must be a boolean")
        if include_randomization_columns and not include_action_columns:
            raise ValueError("randomization columns require include_action_columns=True")
        self.include_action_columns = include_action_columns
        self.include_randomization_columns = include_randomization_columns
        if include_randomization_columns:
            self.columns = RANDOMIZED_ACTION_TELEMETRY_COLUMNS
        elif include_action_columns:
            self.columns = ACTION_TELEMETRY_COLUMNS
        else:
            self.columns = TELEMETRY_COLUMNS
        self._handle: TextIO = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.columns)
        self._writer.writeheader()
        self._handle.flush()
        self._closed = False

    def write_sample(
        self,
        timestamp_s: float,
        position: np.ndarray,
        heading: np.ndarray,
        speed_mps: float,
        *,
        action_idx: int | None = None,
        action_inputs: Mapping[str, bool] | None = None,
        action_was_greedy: bool | None = None,
        action_is_randomized: bool | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("telemetry writer is closed")
        values = [float(timestamp_s), float(speed_mps)]
        position_array = np.asarray(position, dtype=np.float64)
        heading_array = np.asarray(heading, dtype=np.float64)
        if position_array.shape != (3,) or heading_array.shape != (3,):
            raise ValueError("position and heading must have shape (3,)")
        if not np.isfinite(position_array).all() or not np.isfinite(heading_array).all() or not all(math.isfinite(value) for value in values):
            raise ValueError("telemetry values must be finite")
        horizontal_norm = float(np.linalg.norm(heading_array[[0, 2]]))
        if horizontal_norm <= 1e-9:
            raise ValueError("heading must have a non-zero horizontal component")
        heading_array = heading_array / max(float(np.linalg.norm(heading_array)), 1e-9)
        row = {
            "timestamp_s": values[0],
            "position_x_m": position_array[0],
            "position_y_m": position_array[1],
            "position_z_m": position_array[2],
            "heading_x": heading_array[0],
            "heading_y": heading_array[1],
            "heading_z": heading_array[2],
            "speed_mps": values[1],
        }
        if self.include_action_columns:
            self._validate_action(action_idx, action_inputs, action_was_greedy)
            if self.include_randomization_columns:
                if not isinstance(action_is_randomized, (bool, np.bool_)):
                    raise TypeError("action_is_randomized must be boolean")
            elif action_is_randomized is not None:
                raise ValueError("randomization fields require include_randomization_columns=True")
            assert action_inputs is not None
            row.update(
                {
                    "action_idx": action_idx,
                    "action_accelerate": action_inputs["accelerate"],
                    "action_brake": action_inputs["brake"],
                    "action_left": action_inputs["left"],
                    "action_right": action_inputs["right"],
                    "action_was_greedy": action_was_greedy,
                }
            )
            if self.include_randomization_columns:
                row["action_is_randomized"] = action_is_randomized
        elif any(value is not None for value in (action_idx, action_inputs, action_was_greedy, action_is_randomized)):
            raise ValueError("action fields require include_action_columns=True")
        self._writer.writerow(row)
        self._handle.flush()

    @staticmethod
    def _validate_action(
        action_idx: int | None,
        action_inputs: Mapping[str, bool] | None,
        action_was_greedy: bool | None,
    ) -> None:
        if not isinstance(action_idx, Integral) or isinstance(action_idx, bool) or action_idx < 0:
            raise ValueError("action_idx must be a non-negative integer")
        if action_inputs is None or set(action_inputs) != {"accelerate", "brake", "left", "right"}:
            raise ValueError("action_inputs must contain exactly the four boolean input fields")
        if any(not isinstance(value, bool) for value in action_inputs.values()):
            raise ValueError("action_inputs values must be boolean")
        if action_inputs["left"] and action_inputs["right"]:
            raise ValueError("action_inputs cannot press left and right together")
        if not isinstance(action_was_greedy, (bool, np.bool_)):
            raise TypeError("action_was_greedy must be boolean")

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

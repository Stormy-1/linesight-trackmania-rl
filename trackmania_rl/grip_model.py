"""Learned grip limit of the car + pure-pursuit controller (system identification from live logs).

Physics: in a corner of curvature ``k`` at speed ``v`` the car needs lateral
acceleration ``a = v^2 |k|``.  The live logs show the achieved lateral
acceleration follows that demand up to ~35-40 m/s^2 and then saturates while
the car drifts to the outside of the curve (a friction circle, plus the
controller's speed-dependent steering cap).

Model: logistic regression for "the car runs wide within the next second"
from the demand and the speed::

    logit P(wide) = b0 + b_demand * v^2 |k| + b_speed * v

For a chosen risk ``p`` this gives a speed-dependent lateral limit
``a_max(v) = (logit(p) - b0 - b_speed v) / b_demand`` and, solving the
quadratic ``b_demand |k| v^2 + b_speed v + b0 - logit(p) = 0``, a corner speed
limit for every centerline point.  Three numbers, fitted from our own runs.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Labelling and sample filters (chosen from the dev-map logs; see HANDOVER §69).
WIDE_HORIZON_STEPS = 20  # 1.0 s at 50 ms per step
WIDE_RISE_M = 1.5  # outward lateral error grows by this much ...
WIDE_LEVEL_M = 2.5  # ... and ends outside this offset
FREE_FALL_ACCEL_MPS2 = -8.0  # vertical accel below this = airborne
MAX_VERTICAL_ACCEL_MPS2 = 30.0  # above this = landing / impact
MIN_SPEED_MPS = 10.0
MIN_CURVATURE = 0.004  # 1/m; straighter sections carry no grip information
MIN_DEMAND_MPS2 = 15.0
MIN_RUN_STEPS = 10
NEWTON_ITERS = 50
L2 = 1e-4


@dataclass(frozen=True)
class GripModel:
    intercept: float
    demand_coef: float
    speed_coef: float

    def probability(self, demand_mps2: np.ndarray, speed_mps: np.ndarray) -> np.ndarray:
        z = self.intercept + self.demand_coef * np.asarray(demand_mps2) + self.speed_coef * np.asarray(speed_mps)
        return 1.0 / (1.0 + np.exp(-z))

    def lateral_limit_mps2(self, speed_mps: float, risk: float) -> float:
        return (_logit(risk) - self.intercept - self.speed_coef * speed_mps) / self.demand_coef

    def speed_limit_mps(self, curvature: np.ndarray, risk: float) -> np.ndarray:
        """Speed at which running wide reaches ``risk`` on curvature ``k`` (inf on a straight)."""

        if self.demand_coef <= 0.0:
            raise ValueError("grip model needs a positive demand coefficient")
        k = np.abs(np.asarray(curvature, dtype=float))
        c = self.intercept - _logit(risk)
        a = self.demand_coef * k
        b = self.speed_coef
        straight = -c / b if b > 0.0 else math.inf  # limit when k -> 0
        with np.errstate(divide="ignore", invalid="ignore"):
            root = (-b + np.sqrt(np.maximum(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)
        return np.where(a > 1e-12, root, straight)


def _logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk must be in (0, 1), got {p}")
    return math.log(p / (1.0 - p))


def save_grip_model(model: GripModel, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**asdict(model), **(extra or {})}, indent=2))


def load_grip_model(path: Path) -> GripModel:
    data = json.loads(Path(path).read_text())
    return GripModel(float(data["intercept"]), float(data["demand_coef"]), float(data["speed_coef"]))


@dataclass(frozen=True)
class GripSamples:
    demand_mps2: np.ndarray
    speed_mps: np.ndarray
    wide: np.ndarray  # bool


def run_samples(run_csv: Path, line_progress_m: np.ndarray, line_curvature: np.ndarray) -> GripSamples:
    """Grip samples (on ground, cornering, driving) from one live run log."""

    with open(run_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    cols = {c: np.array([_num(r[c]) for r in rows]) for c in ("race_time_s", "x", "y", "z", "progress_m", "lateral_error_m")}
    driving = np.array([r["recovery"] != "True" and r.get("maneuver", "drive") == "drive" for r in rows])
    impact = np.array([r["impact_suspect"] == "True" for r in rows])
    return samples_from_arrays(cols, driving & ~impact, line_progress_m, line_curvature)


def samples_from_arrays(cols: dict, driving: np.ndarray, line_progress_m: np.ndarray, line_curvature: np.ndarray) -> GripSamples:
    t = cols["race_time_s"]
    empty = GripSamples(np.zeros(0), np.zeros(0), np.zeros(0, bool))
    if len(t) < MIN_RUN_STEPS or np.any(np.diff(t) <= 0):
        return empty
    vx, vz, vy = (np.gradient(cols[c], t) for c in ("x", "z", "y"))
    speed = np.hypot(vx, vz)
    vertical_accel = np.gradient(vy, t)
    k = np.interp(cols["progress_m"], line_progress_m, line_curvature)
    demand = speed**2 * np.abs(k)
    outward = np.sign(k) * cols["lateral_error_m"]  # measured: positive = outside of the curve
    wide = _runs_wide(outward)
    keep = (
        driving
        & (vertical_accel > FREE_FALL_ACCEL_MPS2)
        & (np.abs(vertical_accel) < MAX_VERTICAL_ACCEL_MPS2)
        & (speed > MIN_SPEED_MPS)
        & (np.abs(k) > MIN_CURVATURE)
        & (demand > MIN_DEMAND_MPS2)
        & np.isfinite(demand)
    )
    return GripSamples(demand[keep], speed[keep], wide[keep])


def _runs_wide(outward: np.ndarray) -> np.ndarray:
    n = len(outward)
    future_max = np.array([np.nanmax(outward[i : i + WIDE_HORIZON_STEPS]) for i in range(n)])
    return (future_max - outward > WIDE_RISE_M) & (future_max > WIDE_LEVEL_M)


def _num(text: str) -> float:
    return float(text) if text not in ("", "nan") else math.nan


def fit_grip_model(samples: GripSamples) -> GripModel:
    """Logistic regression (Newton's method on standardised inputs)."""

    if len(samples.wide) < 10 or samples.wide.all() or not samples.wide.any():
        raise ValueError("need both wide and non-wide samples to fit")
    x = np.c_[samples.demand_mps2, samples.speed_mps]
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-9)
    z = np.c_[np.ones(len(x)), (x - mean) / std]
    y = samples.wide.astype(float)
    w = np.zeros(3)
    for _ in range(NEWTON_ITERS):
        p = 1.0 / (1.0 + np.exp(-z @ w))
        hessian = z.T @ (z * (p * (1.0 - p))[:, None]) + L2 * np.eye(3)
        step = np.linalg.solve(hessian, z.T @ (p - y) + L2 * w)
        w = w - step
        if np.max(np.abs(step)) < 1e-8:
            break
    demand_coef, speed_coef = w[1] / std[0], w[2] / std[1]
    intercept = w[0] - demand_coef * mean[0] - speed_coef * mean[1]
    return GripModel(float(intercept), float(demand_coef), float(speed_coef))


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-9, 1 - 1e-9)
    return float(-np.mean(np.where(labels, np.log(p), np.log(1 - p))))

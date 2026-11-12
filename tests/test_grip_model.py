import math

import numpy as np
import pytest

from trackmania_rl.grip_model import (
    GripModel,
    GripSamples,
    auc,
    fit_grip_model,
    load_grip_model,
    samples_from_arrays,
    save_grip_model,
)
from trackmania_rl.hybrid_state import build_centerline_geometry
from trackmania_rl.hybrid_track_profile import TrackProfileConfig, build_track_speed_profile

MODEL = GripModel(intercept=-12.0, demand_coef=0.13, speed_coef=0.11)


def test_fit_recovers_synthetic_coefficients():
    rng = np.random.default_rng(0)
    demand = rng.uniform(15, 80, 40000)
    speed = rng.uniform(20, 75, 40000)
    wide = rng.random(40000) < MODEL.probability(demand, speed)

    fitted = fit_grip_model(GripSamples(demand, speed, wide))

    assert fitted.demand_coef == pytest.approx(MODEL.demand_coef, rel=0.15)
    assert fitted.speed_coef == pytest.approx(MODEL.speed_coef, rel=0.15)


def test_speed_limit_sits_exactly_at_the_chosen_risk():
    curvature = np.array([0.005, 0.01, 0.02, 0.05])

    limit = MODEL.speed_limit_mps(curvature, 0.2)

    assert np.allclose(MODEL.probability(limit**2 * curvature, limit), 0.2)
    assert np.all(np.diff(limit) < 0)  # tighter corners -> slower


def test_speed_limit_on_a_straight_is_finite_and_sign_agnostic():
    straight = MODEL.speed_limit_mps(np.array([0.0]), 0.2)[0]

    assert straight == pytest.approx((math.log(0.25) + 12.0) / 0.11)
    assert MODEL.speed_limit_mps(np.array([-0.02]), 0.2)[0] == MODEL.speed_limit_mps(np.array([0.02]), 0.2)[0]


def test_higher_risk_allows_more_speed():
    assert MODEL.speed_limit_mps(np.array([0.02]), 0.3)[0] > MODEL.speed_limit_mps(np.array([0.02]), 0.1)[0]
    assert MODEL.lateral_limit_mps2(60.0, 0.2) < MODEL.lateral_limit_mps2(30.0, 0.2)


def test_invalid_risk_is_rejected():
    with pytest.raises(ValueError):
        MODEL.speed_limit_mps(np.array([0.01]), 1.0)


def test_save_and_load_round_trip(tmp_path):
    save_grip_model(MODEL, tmp_path / "grip.json", {"note": "x"})
    assert load_grip_model(tmp_path / "grip.json") == MODEL


def _circle_run(outward_drift_m_per_s: float) -> dict:
    """A car circling a 50 m radius left turn at 30 m/s, drifting outward."""

    t = np.arange(0, 6, 0.05)
    angle = 30.0 / 50.0 * t
    return {
        "race_time_s": t,
        "x": 50 * np.cos(angle),
        "y": np.zeros_like(t),
        "z": 50 * np.sin(angle),
        "progress_m": 30.0 * t,
        "lateral_error_m": outward_drift_m_per_s * t,  # positive curvature: outward = positive error
    }


def test_samples_label_outward_drift_as_wide():
    line_progress = np.linspace(0, 400, 801)
    line_curvature = np.full(801, 1 / 50)

    drifting = samples_from_arrays(_circle_run(3.0), np.ones(120, bool), line_progress, line_curvature)
    holding = samples_from_arrays(_circle_run(0.0), np.ones(120, bool), line_progress, line_curvature)

    assert drifting.demand_mps2 == pytest.approx(np.full(len(drifting.demand_mps2), 18.0), rel=0.02)
    assert drifting.wide[:40].all()
    assert not holding.wide.any()


def test_auc_perfect_and_random():
    labels = np.array([False, False, True, True])
    assert auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_grip_limit_caps_the_corner_limit_in_the_profile():
    angle = np.linspace(0, math.pi, 400)
    centerline = np.c_[60 * np.cos(angle), np.zeros(400), 60 * np.sin(angle)]  # 60 m radius
    geometry = build_centerline_geometry(centerline)
    base = np.full(len(geometry.cumulative_lengths), 90.0)

    plain = build_track_speed_profile(geometry, TrackProfileConfig(max_speed_mps=100.0), base)
    capped = build_track_speed_profile(geometry, TrackProfileConfig(max_speed_mps=100.0), base, lambda k: MODEL.speed_limit_mps(k, 0.2))

    middle = len(base) // 2
    expected = MODEL.speed_limit_mps(np.array([1 / 60]), 0.2)[0]
    assert plain.corner_limit_mps[middle] == pytest.approx(90.0)
    assert capped.corner_limit_mps[middle] == pytest.approx(expected, rel=0.05)

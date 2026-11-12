import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_speed_dataset import pro_speed_along_path, pro_speed_on_centerline
from trackmania_rl.hybrid_state import build_centerline_geometry
from trackmania_rl.hybrid_track_profile import TrackProfileConfig, build_track_speed_profile
from trackmania_rl.speed_features import FEATURE_NAMES, track_point_features
from trackmania_rl.speed_model import learned_base_speed, load_speed_model, save_speed_model, train_speed_model


def straight_then_turn(n=400, spacing=0.5):
    points = []
    heading = 0.0
    x = z = 0.0
    for i in range(n):
        if i > n // 2:
            heading += 0.01  # constant-radius right-hand curve (r = 50 m)
        x += spacing * np.cos(heading)
        z += spacing * np.sin(heading)
        points.append((x, 10.0, z))
    return build_centerline_geometry(np.array(points))


def test_features_have_one_row_per_point_and_see_the_curve_ahead():
    geometry = straight_then_turn()
    features = track_point_features(geometry)
    assert features.shape == (len(geometry.points), len(FEATURE_NAMES))
    ahead = features[:, FEATURE_NAMES.index("max_abs_curvature_ahead_15_40")]
    here = features[:, FEATURE_NAMES.index("abs_curvature")]
    i = 200 - 50  # 25 m before the curve starts
    assert here[i] < 1e-3 and ahead[i] == pytest.approx(0.02, rel=0.1)


def test_pro_speed_from_ghost_positions():
    raw = np.array([[0.0, 0.0, 5.0 * i] for i in range(20)])  # 5 m per 100 ms = 50 m/s
    arclength, speed = pro_speed_along_path(raw)
    assert arclength[-1] == pytest.approx(95.0) and np.allclose(speed, 50.0)
    assert np.allclose(pro_speed_on_centerline(raw, np.linspace(0.0, 95.0, 191)), 50.0)


def test_mismatched_line_lengths_are_rejected():
    raw = np.array([[0.0, 0.0, 5.0 * i] for i in range(20)])
    with pytest.raises(ValueError):
        pro_speed_on_centerline(raw, np.linspace(0.0, 80.0, 161))


def test_model_learns_a_simple_rule_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 3))
    y = 50.0 + 10.0 * x[:, 0]
    model = train_speed_model(x, y, ("a", "b", "c"), epochs=30)
    assert np.mean(np.abs(model.predict(x) - y)) < 2.0
    save_speed_model(model, tmp_path / "m.pt")
    assert np.allclose(load_speed_model(tmp_path / "m.pt").predict(x[:5]), model.predict(x[:5]), atol=1e-5)


def test_learned_base_speed_replaces_the_corner_rule_within_limits():
    geometry = straight_then_turn()
    base = np.full(len(geometry.points), 200.0)
    profile = build_track_speed_profile(geometry, TrackProfileConfig(max_speed_mps=90.0), base)
    assert profile.corner_limit_mps.max() == 90.0
    with pytest.raises(ValueError):
        build_track_speed_profile(geometry, None, base[:-1])


def test_speed_scale_is_bounded():
    geometry = straight_then_turn()
    model = train_speed_model(np.zeros((10, len(FEATURE_NAMES))), np.full(10, 40.0), FEATURE_NAMES, epochs=1)
    with pytest.raises(ValueError):
        learned_base_speed(geometry, model, 0.0)

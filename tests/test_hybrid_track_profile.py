import math
from pathlib import Path

import numpy as np
import pytest

from trackmania_rl.hybrid_controller import PurePursuitController
from trackmania_rl.hybrid_state import build_centerline_geometry, make_track_relative_state
from trackmania_rl.hybrid_track_profile import TrackProfileConfig, build_track_speed_profile, detect_jumps

ROOT = Path(__file__).resolve().parents[1]


def straight(length=300.0, step=0.5):
    xs = np.arange(0.0, length, step)
    return np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)


def straight_then_turn(radius=40.0):
    line = straight(200.0)
    angles = np.linspace(0.0, math.pi / 2, 300)[1:]
    arc = np.stack([200.0 + radius * np.sin(angles), np.zeros_like(angles), radius * (1 - np.cos(angles))], axis=1)
    return np.vstack([line, arc])


def test_noise_does_not_pin_speed_to_minimum():
    rng = np.random.default_rng(0)
    points = straight(400.0)
    points[:, 1] += rng.normal(0.0, 0.05, len(points))  # vertical jitter like Hockolicious
    points[:, 2] += rng.normal(0.0, 0.02, len(points))
    profile = build_track_speed_profile(build_centerline_geometry(points))
    assert np.median(profile.target_speed_mps) > 45.0  # raw point curvature pins this to the 12 m/s minimum


def test_braking_preview_slows_before_corner():
    profile = build_track_speed_profile(build_centerline_geometry(straight_then_turn()))
    corner_speed = profile.target_at(260.0)
    assert corner_speed < 50.0
    assert profile.target_at(150.0) < 70.0  # already braking 50 m before
    assert profile.target_at(20.0) == pytest.approx(70.0)


def test_detects_ballistic_arc_and_sets_floor():
    g = 15.4
    v_h = 50.0
    ramp = straight(200.0)
    h = np.arange(0.5, 120.0, 0.5)
    flight = np.stack([200.0 + h, 0.3 * h - g * h**2 / (2 * v_h**2), np.zeros_like(h)], axis=1)
    landing = np.stack([np.arange(320.5, 450.0, 0.5), np.full(259, flight[-1, 1]), np.zeros(259)], axis=1)
    geometry = build_centerline_geometry(np.vstack([ramp, flight, landing]))
    jumps = detect_jumps(geometry, TrackProfileConfig())
    assert len(jumps) == 1
    assert jumps[0].reference_horizontal_speed_mps == pytest.approx(v_h, rel=0.05)
    profile = build_track_speed_profile(geometry)
    assert profile.target_at(jumps[0].start_m - 50.0) == pytest.approx(jumps[0].required_speed_mps)
    assert profile.target_at(jumps[0].start_m - 200.0) > jumps[0].required_speed_mps  # free running before the approach


def test_next_jump_approach_does_not_override_current_jump():
    path = ROOT / "maps" / "A05-Race_0.5m_cl2.npy"
    if not path.exists():
        pytest.skip("centerline not available")
    profile = build_track_speed_profile(build_centerline_geometry(np.load(path)))
    first = next(j for j in profile.jumps if j.start_m < 600.0 < j.end_m)
    # The 703 m jump's approach used to start at 583 m and cut the target at this lip.
    assert profile.target_at(first.end_m - 5.0) == pytest.approx(first.required_speed_mps)


def test_flat_track_has_no_jumps():
    assert detect_jumps(build_centerline_geometry(straight_then_turn()), TrackProfileConfig()) == ()


@pytest.mark.parametrize(
    ("track", "fall_m"),
    [("A07-Race", 1577.0), ("ESL-Hockolicious", 967.0)],
)
def test_real_fall_points_are_covered_by_jump_floors(track, fall_m):
    path = ROOT / "maps" / f"{track}_0.5m_cl2.npy"
    if not path.exists():
        pytest.skip("centerline not available")
    profile = build_track_speed_profile(build_centerline_geometry(np.load(path)))
    assert any(j.start_m - 120.0 <= fall_m <= j.end_m for j in profile.jumps)


def test_pure_pursuit_steers_back_from_large_drift():
    geometry = build_centerline_geometry(straight())
    controller = PurePursuitController(build_track_speed_profile(geometry))
    # 6 m to +Z of a +X track, facing along the track: must steer toward -Z (negative).
    state = make_track_relative_state(np.array([50.0, 0.0, 6.0]), np.array([1.0, 0.0, 0.0]), 30.0, geometry)
    command = controller.command(state, geometry)
    assert command.steering < -0.3


def test_recovery_keeps_moving():
    geometry = build_centerline_geometry(straight())
    controller = PurePursuitController(build_track_speed_profile(geometry))
    state = make_track_relative_state(np.array([50.0, 0.0, 12.0]), np.array([0.0, 0.0, 1.0]), 0.5, geometry)
    command = controller.command(state, geometry)
    assert command.recovery
    assert command.throttle == 1.0 and command.brake == 0.0


@pytest.mark.parametrize(("speed", "limit"), [(30.0, 1.0), (70.0, 0.5)])
def test_pure_pursuit_steering_authority_shrinks_with_speed(speed, limit):
    geometry = build_centerline_geometry(straight(1000.0))
    controller = PurePursuitController(build_track_speed_profile(geometry))
    # Facing 40 degrees off the track: an uncapped controller asks for full lock.
    heading = np.array([math.cos(math.radians(40.0)), 0.0, math.sin(math.radians(40.0))])
    state = make_track_relative_state(np.array([100.0, 0.0, 0.0]), heading, speed, geometry)
    command = controller.command(state, geometry)
    assert command.steering == pytest.approx(-limit)


def turn_then_jump(radius=60.0, v_h=90.0):
    """A 90-degree turn, then a short straight and a weak ballistic arc (required speed caps at 70)."""
    g = 15.4
    lead = straight(100.0)
    angles = np.linspace(0.0, math.pi / 2, 200)[1:]
    arc = np.stack([100.0 + radius * np.sin(angles), np.zeros_like(angles), radius * (1 - np.cos(angles))], axis=1)
    end = arc[-1]
    run = np.stack([np.full(80, end[0]), np.zeros(80), end[2] + np.arange(0.5, 40.5, 0.5)], axis=1)
    h = np.arange(0.5, 60.0, 0.5)
    flight = np.stack([np.full_like(h, end[0]), 0.3 * h - g * h**2 / (2 * v_h**2), run[-1, 2] + h], axis=1)
    return np.vstack([lead, arc, run, flight])


def test_jump_approach_never_exceeds_the_corner_limit():
    # A07 v3b (1261 m): a 70 m/s jump approach overrode a 50-80 m radius curve.
    profile = build_track_speed_profile(build_centerline_geometry(turn_then_jump()))
    assert profile.jumps, "fixture must contain a jump"
    assert np.all(profile.target_speed_mps <= profile.corner_limit_mps + 1e-6)


def test_jump_floor_window_is_reported():
    g = 15.4
    v_h = 50.0
    ramp = straight(200.0)
    h = np.arange(0.5, 120.0, 0.5)
    flight = np.stack([200.0 + h, 0.3 * h - g * h**2 / (2 * v_h**2), np.zeros_like(h)], axis=1)
    profile = build_track_speed_profile(build_centerline_geometry(np.vstack([ramp, flight])))
    assert profile.is_jump_locked(profile.jumps[0].start_m - 10.0)
    assert not profile.is_jump_locked(10.0)


def test_pure_pursuit_full_throttle_below_jump_floor():
    # A01 v3b: proportional throttle left the car 3 m/s under the 58.9 m/s floor; it landed short.
    g = 15.4
    v_h = 50.0
    ramp = straight(200.0)
    h = np.arange(0.5, 120.0, 0.5)
    flight = np.stack([200.0 + h, 0.3 * h - g * h**2 / (2 * v_h**2), np.zeros_like(h)], axis=1)
    geometry = build_centerline_geometry(np.vstack([ramp, flight]))
    profile = build_track_speed_profile(geometry)
    controller = PurePursuitController(profile)
    floor = profile.jumps[0].required_speed_mps
    state = make_track_relative_state(np.array([150.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), floor - 1.0, geometry)
    assert controller.command(state, geometry).throttle == 1.0


def test_speed_cap_overrides_jump_floor_and_brakes_ahead():
    # Pace note: A01's 349-506 m "jump" is a hill crest over continuous road (block map).
    g = 15.4
    v_h = 50.0
    ramp = straight(200.0)
    h = np.arange(0.5, 120.0, 0.5)
    flight = np.stack([200.0 + h, 0.3 * h - g * h**2 / (2 * v_h**2), np.zeros_like(h)], axis=1)
    geometry = build_centerline_geometry(np.vstack([ramp, flight]))
    profile = build_track_speed_profile(geometry, TrackProfileConfig(speed_caps=((180.0, 260.0, 30.0),)))
    assert profile.target_at(220.0) == pytest.approx(30.0)
    assert not profile.is_jump_locked(220.0)
    assert profile.target_at(170.0) < 40.0  # braking in time for the cap
    assert profile.target_at(20.0) > 60.0  # far from the cap: unchanged

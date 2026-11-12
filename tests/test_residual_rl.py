"""Step 4 residual RL pieces: observation, residual action, reward, live wiring."""

import csv

import numpy as np
import pytest

from trackmania_rl.hybrid_controller import ConservativeCenterlineController, PurePursuitController
from trackmania_rl.hybrid_live import HybridLivePolicy
from trackmania_rl.hybrid_state import build_centerline_geometry, make_track_relative_state
from trackmania_rl.hybrid_track_profile import build_track_speed_profile
from trackmania_rl.residual_action import (
    ResidualConfig,
    ZeroResidualPolicy,
    apply_steering_residual,
    scale_residual,
)
from trackmania_rl.residual_observation import (
    OBSERVATION_NAMES,
    OBSERVATION_SIZE,
    ResidualObservationBuilder,
    TrackObservationTable,
    _jump_features,
    contact_feature,
    ray_features,
)
from trackmania_rl.residual_reward import RewardConfig, StepRecord, build_transitions, step_rewards

RADIUS_M = 100.0


def arc_centerline(radius=RADIUS_M, length=400.0):
    """Constant right-hand turn: world yaw atan2(z, x) increases along the line."""
    theta = np.arange(0.0, length, 0.5) / radius
    return np.stack([radius * np.sin(theta), np.zeros_like(theta), radius * (1.0 - np.cos(theta))], axis=1)


def orientation_facing(heading):
    forward = np.asarray(heading, dtype=float) / np.linalg.norm(heading)
    up = np.array([0.0, 1.0, 0.0])
    return np.stack([np.cross(up, forward), up, forward])


def pure_pursuit(points):
    return PurePursuitController(build_track_speed_profile(build_centerline_geometry(points)))


def index(name):
    return OBSERVATION_NAMES.index(name)


class ConstantResidual:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def act(self, observation):
        return self.action


def drive(policy, poses):
    """Feed (time, position, velocity) poses through ``decide``; return actions and commands."""
    out = []
    for t, position, velocity in poses:
        out.append(
            policy.decide(race_time_s=t, position=position, velocity=velocity, orientation=orientation_facing(velocity))
        )
    return out


def arc_poses(lateral_m=1.5, speed=30.0, steps=40):
    poses = []
    for k in range(steps):
        theta = (20.0 + k * speed * 0.05) / RADIUS_M
        r = RADIUS_M - lateral_m * np.sin(k / 5.0)
        position = np.array([r * np.sin(theta), 0.0, RADIUS_M - r * np.cos(theta)])
        velocity = speed * np.array([np.cos(theta + 0.05), 0.0, np.sin(theta + 0.05)])
        poses.append((3.0 + k * 0.05, position, velocity))
    return poses


# --- observation -------------------------------------------------------------
def test_observation_has_fixed_size_and_reads_the_turn_direction():
    points = arc_centerline()
    controller = pure_pursuit(points)
    geometry = build_centerline_geometry(points)
    builder = ResidualObservationBuilder(geometry, controller.profile)
    state = make_track_relative_state(points[100], np.array([1.0, 0.0, 0.0]), 30.0, geometry, horizontal_only=True)
    base = controller.command(state, geometry)
    obs = builder.build(state, base, slip_angle_rad=0.1, ranges_m=None, wheel_contact=None)

    assert obs.shape == (OBSERVATION_SIZE,) and obs.dtype == np.float32 and np.all(np.isfinite(obs))
    # Right-hand turn of radius 100 m: curvature +0.01 / scale 0.05 = +0.2 everywhere ahead.
    assert obs[index("curvature_25")] == pytest.approx(0.2, abs=0.02)
    assert obs[index("speed")] == pytest.approx(30.0 / 70.0)
    assert obs[index("slip_angle")] == pytest.approx(0.1)
    assert obs[index("ground_contact")] == 1.0
    assert obs[index("base_steering")] == pytest.approx(base.steering)


def test_lateral_error_is_scaled_by_ten_metres():
    points = np.stack([np.arange(0.0, 300.0, 0.5), np.zeros(600), np.zeros(600)], axis=1)
    controller = pure_pursuit(points)
    geometry = build_centerline_geometry(points)
    builder = ResidualObservationBuilder(geometry, controller.profile)
    state = make_track_relative_state(np.array([50.0, 0.0, 5.0]), np.array([1.0, 0.0, 0.0]), 20.0, geometry, horizontal_only=True)
    obs = builder.build(state, controller.command(state, geometry), slip_angle_rad=np.nan, ranges_m=None, wheel_contact=None)
    assert abs(obs[index("lateral_error")]) == pytest.approx(0.5)
    assert obs[index("slip_angle")] == 0.0  # unknown slip (car at rest) reads as zero


def test_rays_are_fractions_of_max_range():
    rays = ray_features(np.array([0.0, 30.0, 60.0, 90.0] + [60.0] * 9))
    np.testing.assert_allclose(rays[:4], [0.0, 0.5, 1.0, 1.0])
    np.testing.assert_array_equal(ray_features(None), np.ones(13))
    with pytest.raises(ValueError):
        ray_features(np.ones(5))


def test_ground_contact_reads_airborne_as_zero():
    assert contact_feature(np.zeros(4))[0] == 0.0
    assert contact_feature(np.array([1, 1, 0, 0]))[0] == 0.5
    assert contact_feature(None)[0] == 1.0
    with pytest.raises(ValueError):
        contact_feature(np.ones(3))


def test_jump_features_point_at_the_nearest_jump_ahead():
    table = TrackObservationTable(
        progress_m=np.arange(0.0, 1000.0, 0.5),
        signed_curvature=np.zeros(2000),
        slopes=np.zeros((2000, 2)),
        min_vertical_curvature_ahead=np.zeros(2000),
        jump_starts_m=np.array([300.0, 150.0]),
        jump_ends_m=np.array([350.0, 180.0]),
        jump_speeds_mps=np.array([60.0, 35.0]),
    )
    in_jump, distance, speed = _jump_features(table, 100.0)
    assert (in_jump, distance, speed) == (0.0, pytest.approx(50.0 / 200.0), pytest.approx(35.0 / 70.0))
    assert _jump_features(table, 160.0)[0] == 1.0
    assert _jump_features(table, 600.0) == (0.0, 1.0, 0.0)  # nothing within 200 m


# --- residual action -----------------------------------------------------------
def test_residual_is_clipped_and_scaled():
    residual = scale_residual(np.array([2.0, -0.5]), ResidualConfig())
    assert residual.steering_delta == pytest.approx(0.25)
    assert residual.target_speed_delta_mps == pytest.approx(-4.0)
    assert scale_residual(np.zeros(2), ResidualConfig()).is_zero
    with pytest.raises(ValueError):
        scale_residual(np.array([np.nan, 0.0]), ResidualConfig())
    with pytest.raises(ValueError):
        scale_residual(np.zeros(3), ResidualConfig())


def test_steering_residual_respects_the_controller_cap():
    assert apply_steering_residual(0.3, 0.0, 0.5) == 0.3
    assert apply_steering_residual(0.4, 0.25, 0.5) == 0.5
    assert apply_steering_residual(-0.4, -0.25, 0.5) == -0.5


def test_target_speed_offset_moves_the_target_but_not_in_recovery():
    points = np.stack([np.arange(0.0, 300.0, 0.5), np.zeros(600), np.zeros(600)], axis=1)
    controller = pure_pursuit(points)
    geometry = build_centerline_geometry(points)
    on_line = make_track_relative_state(np.array([50.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 20.0, geometry, horizontal_only=True)
    base = controller.command(on_line, geometry)
    assert controller.command(on_line, geometry, target_speed_offset_mps=0.0) == base
    assert controller.command(on_line, geometry, target_speed_offset_mps=-5.0).target_speed_mps == pytest.approx(base.target_speed_mps - 5.0)
    far = make_track_relative_state(np.array([50.0, 0.0, 15.0]), np.array([1.0, 0.0, 0.0]), 20.0, geometry, horizontal_only=True)
    rescue = controller.command(far, geometry)
    assert rescue.recovery
    assert controller.command(far, geometry, target_speed_offset_mps=-5.0) == rescue


# --- live policy wiring ----------------------------------------------------------
def test_zero_residual_reproduces_the_base_stack_exactly(tmp_path):
    points = arc_centerline()
    base_policy = HybridLivePolicy(points, pure_pursuit(points))
    zero_policy = HybridLivePolicy(points, pure_pursuit(points), residual_policy=ZeroResidualPolicy())
    zero_policy.reset(tmp_path / "log.csv")
    poses = arc_poses()
    assert drive(base_policy, poses) == drive(zero_policy, poses)
    zero_policy.close()
    assert len(zero_policy.step_records) == len(poses)
    assert all(not np.any(r.action) for r in zero_policy.step_records)
    rows = list(csv.DictReader((tmp_path / "log.csv").read_text().splitlines()))
    assert {"ground_contact", "residual_active", "residual_steering"} <= set(rows[0])


def test_nonzero_residual_shifts_steering_and_target_speed():
    points = arc_centerline()
    base = drive(HybridLivePolicy(points, pure_pursuit(points)), arc_poses())
    nudged_policy = HybridLivePolicy(points, pure_pursuit(points), residual_policy=ConstantResidual([-1.0, -1.0]))
    nudged = drive(nudged_policy, arc_poses())
    for (_, b), (_, n) in zip(base, nudged):
        assert n.target_speed_mps == pytest.approx(b.target_speed_mps - 8.0)
        assert n.steering == pytest.approx(b.steering - 0.25)
    assert nudged_policy.residual_active_steps == len(base)


def test_residual_is_gated_off_in_recovery():
    points = np.stack([np.arange(0.0, 300.0, 0.5), np.zeros(600), np.zeros(600)], axis=1)
    policy = HybridLivePolicy(points, pure_pursuit(points), residual_policy=ConstantResidual([1.0, 1.0]))
    velocity = np.array([20.0, 0.0, 0.0])
    _, command = policy.decide(race_time_s=3.0, position=np.array([50.0, 0.0, 15.0]), velocity=velocity, orientation=orientation_facing(velocity))
    record = policy.step_records[-1]
    assert command.recovery and not record.residual_active
    assert not np.any(record.action)


def test_residual_needs_pure_pursuit():
    points = arc_centerline()
    with pytest.raises(TypeError):
        HybridLivePolicy(points, ConservativeCenterlineController(), residual_policy=ZeroResidualPolicy())


def test_wheel_contact_reaches_the_observation():
    points = arc_centerline()
    policy = HybridLivePolicy(points, pure_pursuit(points), residual_policy=ZeroResidualPolicy())
    velocity = np.array([30.0, 0.0, 0.0])
    policy.observe_simulation_state(
        race_time_s=3.0, position=points[40], velocity=velocity, orientation=orientation_facing(velocity), wheel_contact=np.zeros(4)
    )
    policy(None, None)
    assert policy.step_records[-1].observation[index("ground_contact")] == 0.0


# --- reward -------------------------------------------------------------------------
def record(progress, impact=False, recovery=False, reversing=False, action=(0.0, 0.0)):
    return StepRecord(np.zeros(OBSERVATION_SIZE, np.float32), np.asarray(action, np.float32), True, progress, impact, recovery, reversing)


def test_step_rewards_pay_progress_and_charge_events():
    cfg = RewardConfig()
    records = [record(0.0, action=(1.0, 0.0)), record(2.0, impact=True), record(4.0, recovery=True), record(50.0, reversing=True)]
    rewards = step_rewards(records, cfg)
    assert rewards[0] == pytest.approx(0.2 - cfg.impact_penalty - cfg.residual_penalty)
    assert rewards[1] == pytest.approx(0.2 - cfg.recovery_penalty)
    # a 46 m map-matching hop is clipped to 6 m
    assert rewards[2] == pytest.approx(0.6 - cfg.reverse_penalty)


@pytest.mark.parametrize(
    ("termination", "length", "last_terminal"),
    [("Finished", 4, True), ("Stuck_Cutoff_2s", 4, True), ("Timeout_90s", 3, False)],
)
def test_transitions_end_by_termination(termination, length, last_terminal):
    records = [record(float(i)) for i in range(4)]
    transitions = build_transitions(records, termination)
    assert len(transitions.rewards) == length
    assert transitions.terminals[-1] == last_terminal
    assert transitions.terminals[:-1].sum() == 0
    assert transitions.observations.shape == transitions.next_observations.shape == (length, OBSERVATION_SIZE)
    if termination.startswith("Stuck"):
        assert transitions.rewards[-1] == pytest.approx(-RewardConfig().stuck_penalty)


def test_disrupted_or_empty_episodes_are_dropped():
    assert build_transitions([record(0.0), record(1.0)], "TMI_Disrupt") is None
    assert build_transitions([record(0.0)], "Finished") is None


# --- physics shield (HANDOVER §82) ----------------------------------------------------
class JumpLockedProfile:
    """Wraps a real profile but reports every point as a jump's required-speed approach."""

    def __init__(self, profile):
        self._profile = profile

    def __getattr__(self, name):
        return getattr(self._profile, name)

    def is_jump_locked(self, progress_m):
        return True


SHIELD_ON = ResidualConfig(jump_floor_shield=True, airborne_shield=True)


def straight_policy(action, residual_config=SHIELD_ON, jump_locked=False):
    points = np.stack([np.arange(0.0, 300.0, 0.5), np.zeros(600), np.zeros(600)], axis=1)
    controller = pure_pursuit(points)
    if jump_locked:
        controller.profile = JumpLockedProfile(controller.profile)
    return HybridLivePolicy(points, controller, residual_policy=ConstantResidual(action), residual_config=residual_config)


def step(policy, wheel_contact=None):
    velocity = np.array([20.0, 0.0, 0.0])
    return policy.decide(
        race_time_s=3.0, position=np.array([50.0, 0.0, 0.5]), velocity=velocity,
        orientation=orientation_facing(velocity), wheel_contact=wheel_contact,
    )[1]


def test_residual_may_not_lower_a_jump_speed_floor():
    base = step(straight_policy([0.0, 0.0], jump_locked=True))
    slower = straight_policy([0.5, -1.0], jump_locked=True)
    command = step(slower)
    assert command.target_speed_mps == base.target_speed_mps  # the floor holds
    assert command.steering == pytest.approx(base.steering + 0.125)  # steering residual still applies
    np.testing.assert_array_equal(slower.step_records[-1].action, [0.5, 0.0])  # the applied action is recorded
    faster = step(straight_policy([0.0, 1.0], jump_locked=True))
    assert faster.target_speed_mps == pytest.approx(base.target_speed_mps + 8.0)  # raising it is allowed


def test_residual_is_off_while_airborne():
    base = step(straight_policy([0.0, 0.0]))
    policy = straight_policy([1.0, -1.0])
    assert step(policy, wheel_contact=np.zeros(4)) == base
    assert not policy.step_records[-1].residual_active
    step(policy, wheel_contact=np.array([1.0, 0.0, 0.0, 0.0]))  # one wheel down = grounded
    assert policy.step_records[-1].residual_active


def test_shield_is_off_by_default():
    config = ResidualConfig()
    base = step(straight_policy([0.0, 0.0], jump_locked=True))
    command = step(straight_policy([0.0, -1.0], config, jump_locked=True), wheel_contact=np.zeros(4))
    assert command.target_speed_mps == pytest.approx(base.target_speed_mps - 8.0)

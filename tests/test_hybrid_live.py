import csv
import math
from pathlib import Path

import numpy as np
import pytest

from config_files.inputs_list import inputs as ACTION_INPUTS
from trackmania_rl.hybrid_controller import ConservativeCenterlineController, ControllerCommand
from trackmania_rl.hybrid_live import DiscreteActionAdapter, HybridLivePolicy, action_index

ROOT = Path(__file__).resolve().parents[1]
A01_CAPTURE = ROOT / "eval_results/hybrid_rollouts/action_transition_a01_20261103_retry3/A01-Race.Challenge_1793720895263_37200.csv"


def cmd(steering=0.0, throttle=0.0, brake=0.0):
    return ControllerCommand(steering, throttle, brake, 30.0, False)


def keys(index):
    return ACTION_INPUTS[index]


def straight_x_centerline(length=200.0):
    xs = np.arange(0.0, length, 0.5)
    return np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)


def orientation_facing(heading):
    """World->car matrix whose row 2 (car forward) is ``heading``."""
    forward = np.asarray(heading, dtype=float) / np.linalg.norm(heading)
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(up, forward)
    return np.stack([side, up, forward])


def test_action_index_round_trips_every_input():
    for index, combo in enumerate(ACTION_INPUTS):
        assert action_index(**combo) == index


def test_full_steering_holds_the_key_every_step():
    adapter = DiscreteActionAdapter()
    assert all(keys(adapter.to_action_index(cmd(steering=1.0, throttle=1.0)))["right"] for _ in range(10))
    adapter.reset()
    assert all(keys(adapter.to_action_index(cmd(steering=-1.0, throttle=1.0)))["left"] for _ in range(10))


def test_partial_commands_are_duty_cycled():
    adapter = DiscreteActionAdapter()
    actions = [keys(adapter.to_action_index(cmd(steering=0.5, throttle=0.25))) for _ in range(40)]
    assert sum(a["right"] for a in actions) == 20
    assert sum(a["accelerate"] for a in actions) == 10
    assert not any(a["left"] or a["brake"] for a in actions)


def test_brake_overrides_throttle():
    adapter = DiscreteActionAdapter()
    combo = keys(adapter.to_action_index(cmd(throttle=1.0, brake=0.8)))
    assert combo["brake"] and not combo["accelerate"]


def test_car_left_of_track_steers_back_with_measured_sign():
    # Track runs along +X.  Car sits at +Z (left_error negative under the state
    # convention).  Measured in-game: LEFT turns heading toward -Z when facing +X,
    # so the car must press LEFT to rejoin.
    policy = HybridLivePolicy(straight_x_centerline(), ConservativeCenterlineController())
    velocity = np.array([20.0, 0.0, 0.0])
    pressed = []
    for step in range(3):
        policy.observe_simulation_state(
            race_time_s=step * 0.05, position=np.array([50.0, 0.0, 4.0]), velocity=velocity, orientation=orientation_facing(velocity)
        )
        action, greedy, _, q_values = policy(None, None)
        pressed.append(keys(action))
    assert any(k["left"] for k in pressed) and not any(k["right"] for k in pressed)
    assert greedy is True and q_values.shape == (len(ACTION_INPUTS),)


def test_policy_requires_observation():
    policy = HybridLivePolicy(straight_x_centerline(), ConservativeCenterlineController())
    with pytest.raises(RuntimeError):
        policy(None, None)


def test_stationary_car_uses_forward_axis():
    policy = HybridLivePolicy(straight_x_centerline(), ConservativeCenterlineController())
    heading, source, _ = policy.heading(np.zeros(3), orientation_facing([1.0, 0.0, 0.0]))
    assert source == "forward_axis"
    np.testing.assert_allclose(heading, [1.0, 0.0, 0.0])


def test_rolling_backward_uses_nose_direction():
    policy = HybridLivePolicy(straight_x_centerline(), ConservativeCenterlineController())
    heading, source, _ = policy.heading(np.array([-10.0, 0.0, 0.0]), orientation_facing([1.0, 0.0, 0.0]))
    assert source == "forward_axis"
    np.testing.assert_allclose(heading, [1.0, 0.0, 0.0])


def test_impact_proxy_flags_sudden_speed_loss(tmp_path):
    policy = HybridLivePolicy(straight_x_centerline(), ConservativeCenterlineController())
    policy.reset(tmp_path / "log.csv")
    for t, speed in [(0.0, 40.0), (0.05, 39.0), (0.10, 20.0)]:
        v = np.array([speed, 0.0, 0.0])
        policy.decide(race_time_s=t, position=np.array([10 + t * 40, 0.0, 0.0]), velocity=v, orientation=orientation_facing(v))
    policy.close()
    assert policy.impact_suspects == 1
    rows = list(csv.DictReader((tmp_path / "log.csv").read_text().splitlines()))
    assert [r["impact_suspect"] for r in rows] == ["False", "False", "True"]


def test_unstick_reverses_when_pinned():
    policy = HybridLivePolicy(straight_x_centerline(), ConservativeCenterlineController())
    stopped = np.zeros(3)
    facing = orientation_facing([1.0, 0.0, 0.0])
    maneuvers = []
    for step in range(80):  # 4 s at full throttle against a "wall"
        action, _ = policy.decide(
            race_time_s=step * 0.05, position=np.array([10.0, 0.0, 0.0]), velocity=stopped, orientation=facing
        )
        maneuvers.append(keys(action)["brake"])
    assert policy.reversals >= 1
    assert not any(maneuvers[: int(2.0 / 0.05)])  # grace period at the start line
    assert any(maneuvers[40:])


@pytest.mark.skipif(not A01_CAPTURE.exists(), reason="real A01 capture not available")
def test_replays_real_a01_capture_without_losing_track():
    centerline = np.load(ROOT / "maps" / "A01-Race_0.5m_cl2.npy")
    policy = HybridLivePolicy(centerline, ConservativeCenterlineController())
    rows = list(csv.DictReader(A01_CAPTURE.read_text().splitlines()))
    progress = []
    for row in rows:
        position = np.array([float(row[k]) for k in ("position_x_m", "position_y_m", "position_z_m")])
        heading = np.array([float(row[k]) for k in ("heading_x", "heading_y", "heading_z")])
        speed = float(row["speed_mps"])
        if math.hypot(heading[0], heading[2]) < 1e-6:
            continue
        velocity = heading / np.linalg.norm(heading) * speed
        action, command = policy.decide(
            race_time_s=float(row["timestamp_s"]), position=position, velocity=velocity, orientation=orientation_facing(heading)
        )
        assert 0 <= action < len(ACTION_INPUTS)
        assert all(math.isfinite(v) for v in (command.steering, command.throttle, command.brake))
        progress.append(policy.max_progress_m)
    # The recorded greedy run reached ~88% of A01 (~1,900 m); temporal map
    # matching must follow it rather than jump to another part of the track.
    assert progress[-1] > 1500.0
    assert np.all(np.diff(progress) < 40.0)


def slipping_orientation(velocity, slip_rad):
    """Car nose rotated ``slip_rad`` away from the velocity in the horizontal plane."""
    yaw = math.atan2(velocity[2], velocity[0]) + slip_rad
    return orientation_facing([math.cos(yaw), 0.0, math.sin(yaw)])


def drive_slipping(policy, slip_rad, steps=3, speed=60.0):
    velocity = np.array([speed, 0.0, 0.0])
    commands = []
    for step in range(steps):
        position = np.array([20.0 + step * speed * 0.05, 0.0, 0.0])
        _, command = policy.decide(
            race_time_s=step * 0.05, position=position, velocity=velocity, orientation=slipping_orientation(velocity, slip_rad)
        )
        commands.append(command)
    return commands


@pytest.mark.parametrize("slip_rad", [0.6, -0.6])
def test_slip_guard_steers_gently_against_the_angle(slip_rad):
    # Live v3 (2026-11-10): full lock with +angle spun A01 and A07 to 3.1 rad;
    # mild steering with -angle held the slide in v1/v2.
    policy = HybridLivePolicy(straight_x_centerline(1000.0), ConservativeCenterlineController())
    commands = drive_slipping(policy, slip_rad)
    assert all(math.copysign(1.0, c.steering) == -math.copysign(1.0, slip_rad) for c in commands)
    assert all(abs(c.steering) <= 0.5 for c in commands)
    assert policy.slip_steps == 3


def test_slip_guard_ignores_small_angles():
    policy = HybridLivePolicy(straight_x_centerline(1000.0), ConservativeCenterlineController())
    drive_slipping(policy, 0.1)
    assert policy.slip_steps == 0


def test_slip_guard_has_hysteresis(tmp_path):
    policy = HybridLivePolicy(straight_x_centerline(1000.0), ConservativeCenterlineController())
    policy.reset(tmp_path / "log.csv")
    velocity = np.array([60.0, 0.0, 0.0])
    for step, slip in enumerate([0.5, 0.2, 0.1]):
        policy.decide(
            race_time_s=step * 0.05,
            position=np.array([20.0 + step * 3.0, 0.0, 0.0]),
            velocity=velocity,
            orientation=slipping_orientation(velocity, slip),
        )
    policy.close()
    rows = list(csv.DictReader((tmp_path / "log.csv").read_text().splitlines()))
    assert [r["maneuver"] for r in rows] == ["slip", "slip", "drive"]

import json
import math
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from unittest.mock import patch

import numpy as np

from trackmania_rl import hybrid_replay, hybrid_simulator
from trackmania_rl.hybrid_controller import ConservativeCenterlineController, ControllerCommand
from trackmania_rl.hybrid_replay import (
    DisturbanceSpec,
    InjectionNotReached,
    MidRolloutInjection,
    disturb_sample,
    replay_midrollout_disturbance,
    summarize_rollout_suffix,
)
from trackmania_rl.hybrid_rollout import RolloutSample
from trackmania_rl.hybrid_simulator import ClosedLoopConfig, simulate_closed_loop, synthetic_centerline_geometry
from trackmania_rl.hybrid_state import build_centerline_geometry


def encoded(value):
    """Compare every scalar, array element, command, and terminal result exactly."""
    return json.dumps(asdict(value), default=lambda array: array.tolist(), allow_nan=False, sort_keys=True)


class RecordingController(ConservativeCenterlineController):
    def __init__(self):
        super().__init__()
        self.states = []

    def command(self, state, geometry):
        self.states.append(state)
        return super().command(state, geometry)


class CoastingController(RecordingController):
    """A stateful controller whose scripted diagnostics expose extra calls."""

    def __init__(self, recovery_steps=()):
        super().__init__()
        self.recovery_steps = set(recovery_steps)

    def command(self, state, geometry):
        index = len(self.states)
        self.states.append(state)
        return ControllerCommand(0.0, 0.0, 0.0, state.speed_mps, index in self.recovery_steps)


class HybridMidRolloutTests(unittest.TestCase):
    def setUp(self):
        self.geometry = build_centerline_geometry(np.asarray([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]))
        self.config = ClosedLoopConfig(dt_s=0.1, max_steps=12, linear_drag_mps2_per_mps=0.0)
        self.initial = {
            "initial_position": np.asarray([0.0, 0.0, 0.0]),
            "initial_heading": np.asarray([1.0, 0.0, 0.0]),
            "initial_speed_mps": 10.0,
        }

    def replay(self, injection, *, controller=None, geometry=None, config=None, **initial):
        return replay_midrollout_disturbance(
            self.geometry if geometry is None else geometry,
            **(self.initial | initial),
            injection=injection,
            controller=CoastingController() if controller is None else controller,
            config=self.config if config is None else config,
        )

    def curved_setup(self):
        geometry = synthetic_centerline_geometry(length_m=300.0, amplitude_m=9.0, wavelength_m=55.0)
        initial = {
            "initial_position": geometry.points[0].copy(),
            "initial_heading": geometry.segment_tangents[0].copy(),
            "initial_speed_mps": 18.0,
        }
        config = ClosedLoopConfig(dt_s=0.07, max_steps=48, search_radius=3)
        return geometry, initial, config

    def test_curved_nonzero_injection_preserves_every_prefix_value(self):
        geometry, initial, config = self.curved_setup()
        baseline = simulate_closed_loop(geometry, **initial, config=config)
        for disturbance in (
            DisturbanceSpec("lateral", lateral_offset_m=5.0),
            DisturbanceSpec("combined", lateral_offset_m=-4.0, heading_offset_rad=0.3, speed_delta_mps=2.0),
        ):
            with self.subTest(disturbance=disturbance.name):
                outcome = replay_midrollout_disturbance(
                    geometry, **initial, config=config, injection=MidRolloutInjection(17, disturbance)
                )
                for before, after in zip(baseline.steps[:17], outcome.result.steps[:17]):
                    self.assertEqual(encoded(before), encoded(after))
                self.assertEqual(encoded(baseline.steps[17].state), encoded(outcome.pre_injection_state))
                self.assertEqual(len(outcome.result.steps), config.max_steps)
                self.assertEqual(outcome.result.elapsed_s, config.max_steps * config.dt_s)

    def test_curved_noop_preserves_entire_trajectory_without_extra_projection(self):
        geometry, initial, config = self.curved_setup()
        baseline = simulate_closed_loop(geometry, **initial, config=config)
        for target in (0, 17, config.max_steps - 1):
            with self.subTest(target=target), patch.object(
                hybrid_simulator, "_validate_pose", wraps=hybrid_simulator._validate_pose
            ) as validate, patch.object(
                hybrid_simulator, "make_track_relative_state", wraps=hybrid_simulator.make_track_relative_state
            ) as project, patch.object(hybrid_replay, "disturb_sample", wraps=disturb_sample) as disturb:
                outcome = replay_midrollout_disturbance(
                    geometry, **initial, config=config, injection=MidRolloutInjection(target, DisturbanceSpec("noop"))
                )
                self.assertEqual(encoded(baseline), encoded(outcome.result))
                self.assertEqual(encoded(outcome.pre_injection_state), encoded(outcome.post_injection_state))
                self.assertEqual(validate.call_count, 1)
                self.assertEqual(project.call_count, 2 * len(baseline.steps))
                disturb.assert_not_called()
                self.assertEqual(outcome.metrics["injection"]["progress_jump_m"], 0.0)

    def test_mutator_receives_independent_arrays_even_when_returning_none(self):
        geometry, initial, config = self.curved_setup()
        baseline = simulate_closed_loop(geometry, **initial, config=config)
        geometry_before = encoded(geometry)
        calls = []

        def observe(index, timestamp, position, heading, speed, state):
            calls.append((index, timestamp))
            position[:] = math.nan
            heading[:] = math.nan
            state.projection[:] = math.nan
            state.tangent[:] = math.nan
            return None

        result = simulate_closed_loop(geometry, **initial, config=config, _pose_mutator=observe)
        self.assertEqual(encoded(baseline), encoded(result))
        self.assertEqual(encoded(geometry), geometry_before)
        self.assertEqual(calls, [(step.step_index, step.timestamp_s) for step in result.steps])

    def test_injection_is_applied_before_command_with_realized_pose_and_snapshots(self):
        controller = RecordingController()
        initial_position = self.initial["initial_position"].copy()
        initial_heading = self.initial["initial_heading"].copy()
        geometry_before = encoded(self.geometry)
        outcome = self.replay(
            MidRolloutInjection(4, DisturbanceSpec("combined", 12.0, 0.25, 3.0)), controller=controller
        )
        pre, post = outcome.pre_injection_state, outcome.post_injection_state
        pose = outcome.disturbed_pose
        self.assertAlmostEqual(post.lateral_error_m - pre.lateral_error_m, 12.0)
        self.assertAlmostEqual(post.heading_error_rad - pre.heading_error_rad, 0.25)
        self.assertEqual(post.speed_mps - pre.speed_mps, 3.0)
        np.testing.assert_array_equal(pose.position, pre.projection + np.asarray([0.0, 0.0, -12.0]))
        self.assertAlmostEqual(math.atan2(pose.heading[2], pose.heading[0]), 0.25)
        self.assertEqual(pose.speed_mps, post.speed_mps)
        self.assertEqual(pose.progress_m, post.progress_m)
        self.assertEqual(pose.source_index, 4)
        self.assertEqual(pose.timestamp_s, 0.4)
        self.assertIs(controller.states[4], post)
        self.assertIs(outcome.result.steps[4].state, post)
        self.assertEqual(len(controller.states), len(outcome.result.steps))
        self.assertIsNone(outcome.metrics["pre_injection"]["recovery"])
        self.assertIsNone(outcome.metrics["pre_injection"]["steering"])
        actual = outcome.result.steps[4].command
        for field in ("recovery", "steering", "throttle", "brake", "target_speed_mps"):
            self.assertEqual(outcome.metrics["post_injection"][field], getattr(actual, field))
        self.assertTrue(actual.recovery)
        self.assertEqual(outcome.metrics["injection"]["requested"], {
            "lateral_offset_m": 12.0, "heading_offset_rad": 0.25, "speed_delta_mps": 3.0
        })
        self.assertEqual(outcome.metrics["injection"]["realized_pose"]["position"], pose.position.tolist())
        np.testing.assert_array_equal(self.initial["initial_position"], initial_position)
        np.testing.assert_array_equal(self.initial["initial_heading"], initial_heading)
        self.assertEqual(encoded(self.geometry), geometry_before)

    def test_disturbance_is_applied_once_and_stateful_controller_is_not_restarted(self):
        controller = CoastingController(recovery_steps=(4, 5))
        with patch.object(hybrid_replay, "disturb_sample", wraps=disturb_sample) as disturb:
            outcome = self.replay(
                MidRolloutInjection(4, DisturbanceSpec("speed", speed_delta_mps=3.0)), controller=controller
            )
        disturb.assert_called_once()
        self.assertEqual(len(controller.states), self.config.max_steps)
        self.assertEqual([step.state.speed_mps for step in outcome.result.steps], [10.0] * 4 + [13.0] * 8)
        self.assertEqual(outcome.result.final_speed_mps, 13.0)
        self.assertEqual([step.step_index for step in outcome.result.steps if step.command.recovery], [4, 5])
        self.assertEqual(outcome.metrics["recovery"]["activation"]["step_index"], 4)
        self.assertEqual(outcome.metrics["recovery"]["completion"]["step_index"], 6)

    def test_repeat_runs_and_metrics_are_exact_and_strict_json_safe(self):
        geometry, initial, config = self.curved_setup()
        injection = MidRolloutInjection(17, DisturbanceSpec("repeat", 10.0, 0.4, -2.0))
        first = replay_midrollout_disturbance(geometry, **initial, config=config, injection=injection)
        second = replay_midrollout_disturbance(geometry, **initial, config=config, injection=injection)
        self.assertEqual(encoded(first), encoded(second))
        self.assertEqual(json.loads(json.dumps(first.metrics, allow_nan=False)), first.metrics)
        self.assertEqual(first.metrics["recovery"], first.metrics["post_rollout"]["recovery"])
        self.assertEqual(first.metrics["total_elapsed_s"], first.result.elapsed_s)
        with self.assertRaises(FrozenInstanceError):
            injection.step_index = 2
        with self.assertRaises(FrozenInstanceError):
            first.result = second.result

    def test_a_falsey_controller_instance_is_retained(self):
        class FalseyController(CoastingController):
            def __bool__(self):
                return False

        controller = FalseyController()
        outcome = self.replay(MidRolloutInjection(4, DisturbanceSpec("noop")), controller=controller)
        self.assertEqual(len(controller.states), len(outcome.result.steps))
        self.assertEqual(outcome.result.final_speed_mps, self.initial["initial_speed_mps"])

    def test_suffix_metrics_exclude_prefix_events_sections_and_duration(self):
        geometry = build_centerline_geometry(np.asarray([[0.0, 0.0, 0.0], [120.0, 0.0, 0.0]]))
        config = replace(self.config, dt_s=1.0, section_bins=6)
        outcome = self.replay(
            MidRolloutInjection(4, DisturbanceSpec("noop")),
            geometry=geometry,
            config=config,
            controller=CoastingController(recovery_steps=(0, 1, 4, 8, 9)),
        )
        metrics = outcome.metrics["post_rollout"]
        self.assertEqual(len(outcome.result.steps), 12)
        self.assertEqual(outcome.result.recovery_samples, 5)
        self.assertEqual(outcome.result.first_recovery_event["step_index"], 2)
        self.assertEqual(metrics["recovery"]["samples"], 3)
        self.assertEqual(metrics["recovery_activation"]["step_index"], 4)
        self.assertEqual(metrics["recovery_activation"]["timestamp_s"], 4.0)
        self.assertEqual(metrics["recovery_completion"]["step_index"], 5)
        self.assertEqual(metrics["stable_rejoin_event"]["step_index"], 7)
        self.assertEqual(metrics["stable_rejoin_event"]["timestamp_s"], 7.0)
        self.assertEqual(metrics["first_redivergence_event"]["step_index"], 8)
        self.assertEqual(metrics["redivergence_count"], 1)
        self.assertEqual(metrics["recovery_latency_s"], 3.0)
        self.assertEqual(metrics["elapsed_s"], 8.0)
        self.assertEqual(outcome.metrics["total_elapsed_s"], 12.0)
        self.assertEqual(metrics["steps"], 8)
        self.assertAlmostEqual(metrics["progress_start_m"], 40.0)
        self.assertAlmostEqual(metrics["progress_gain_m"], 80.0)
        self.assertEqual(sum(section["samples"] for section in metrics["section_metrics"]), 8)
        self.assertEqual(sum(section["recovery_samples"] for section in metrics["section_metrics"]), 3)
        self.assertEqual(metrics["section_metrics"][0]["samples"], 0)
        self.assertIsNone(metrics["section_metrics"][0]["max_abs_lateral_error_m"])
        self.assertEqual(summarize_rollout_suffix(outcome.result, 4, dt_s=1.0), metrics)

        # A suffix of a result already sliced by another consumer still uses
        # global step identities, not offsets into that shorter tuple.
        sliced = replace(outcome.result, steps=outcome.result.steps[2:])
        self.assertEqual(summarize_rollout_suffix(sliced, 4, dt_s=1.0), metrics)

    def test_suffix_error_statistics_omit_prefix_and_include_terminal_state(self):
        outcome = self.replay(
            MidRolloutInjection(4, DisturbanceSpec("rejoin", lateral_offset_m=-12.0)),
            initial_position=np.asarray([0.0, 0.0, -12.0]),
        )
        metrics = outcome.metrics["post_rollout"]
        self.assertEqual(outcome.result.max_abs_lateral_error_m, 12.0)
        self.assertEqual(metrics["max_abs_lateral_error_m"], 0.0)
        self.assertEqual(metrics["mean_horizontal_error_m"], 0.0)
        terminal = replace(outcome.result.final_state, lateral_error_m=9.0, horizontal_distance_to_centerline_m=9.0)
        changed = summarize_rollout_suffix(replace(outcome.result, final_state=terminal), 4, dt_s=0.1)
        self.assertEqual(changed["max_abs_lateral_error_m"], 9.0)
        self.assertEqual(changed["mean_abs_lateral_error_m"], 1.0)
        self.assertEqual(changed["final_horizontal_error_m"], 9.0)

    def test_artificial_backward_injection_jump_is_not_backward_driving(self):
        geometry = build_centerline_geometry(np.asarray([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [50.0, 0.0, 50.0]]))
        outcome = self.replay(
            MidRolloutInjection(3, DisturbanceSpec("jump", lateral_offset_m=-50.0)),
            geometry=geometry,
            initial_position=np.asarray([50.0, 0.0, 10.0]),
            initial_heading=np.asarray([0.0, 0.0, 1.0]),
            initial_speed_mps=0.0,
        )
        self.assertEqual(outcome.pre_injection_state.progress_m, 60.0)
        self.assertEqual(outcome.post_injection_state.progress_m, 0.0)
        self.assertEqual(outcome.metrics["injection"]["progress_jump_m"], -60.0)
        self.assertEqual(outcome.result.nonmonotonic_progress_steps, 1)
        self.assertEqual(outcome.metrics["post_rollout"]["backward_progress_event_count"], 0)
        self.assertEqual(outcome.metrics["post_rollout"]["backward_progress_distance_m"], 0.0)
        self.assertEqual(outcome.metrics["post_rollout"]["progress_start_m"], 0.0)

    def test_elevated_crossing_preserves_actual_tangent_and_prior_search_window(self):
        geometry = build_centerline_geometry(np.asarray([
            [0.0, 20.0, 0.0], [100.0, 20.0, 0.0], [100.0, 0.0, -30.0],
            [50.0, 0.0, -30.0], [50.0, 0.0, 30.0], [150.0, 0.0, 30.0],
        ]))
        config = replace(self.config, search_radius=0)
        disturbance = DisturbanceSpec("crossing", lateral_offset_m=3.0)
        outcome = self.replay(
            MidRolloutInjection(2, disturbance), geometry=geometry, config=config,
            initial_position=np.asarray([48.0, 0.0, 0.0]),
        )
        self.assertEqual(outcome.pre_injection_state.segment_index, 0)
        self.assertEqual(outcome.post_injection_state.segment_index, 0)
        np.testing.assert_array_equal(outcome.pre_injection_state.tangent, [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(outcome.disturbed_pose.position, [50.0, 0.0, -3.0])
        self.assertEqual(outcome.disturbed_pose.progress_m, 50.0)
        self.assertEqual(outcome.post_injection_state.elevation_error_m, -20.0)
        self.assertTrue(all(step.state.segment_index == 0 for step in outcome.result.steps))
        sample = RolloutSample(0.2, np.asarray([50.0, 0.0, 0.0]), self.initial["initial_heading"], 10.0)
        historical = disturb_sample(sample, geometry, disturbance, source_index=2)
        np.testing.assert_array_equal(historical.position, [53.0, 0.0, 0.0])
        referenced = disturb_sample(
            sample, geometry, disturbance, source_index=2, reference_state=outcome.pre_injection_state,
            horizontal_only=True, search_radius=0,
        )
        np.testing.assert_array_equal(referenced.position, outcome.disturbed_pose.position)
        self.assertEqual(referenced.progress_m, outcome.disturbed_pose.progress_m)

    def test_actual_backward_motion_after_injection_is_still_counted(self):
        outcome = self.replay(MidRolloutInjection(3, DisturbanceSpec("reverse", heading_offset_rad=math.pi)))
        self.assertEqual(outcome.metrics["injection"]["progress_jump_m"], 0.0)
        metrics = outcome.metrics["post_rollout"]
        self.assertEqual(metrics["backward_progress_event_count"], 3)
        self.assertAlmostEqual(metrics["backward_progress_distance_m"], 3.0)
        self.assertAlmostEqual(metrics["progress_gain_m"], -3.0)
        self.assertFalse(metrics["progress_continuity"])

    def test_invalid_injection_indices_and_types_are_rejected(self):
        for index in (-1, True, False, 1.5, "1", None):
            with self.subTest(index=index), self.assertRaises(ValueError):
                MidRolloutInjection(index, DisturbanceSpec("invalid"))
        with self.assertRaises(TypeError):
            MidRolloutInjection(1, "invalid")
        with self.assertRaises(TypeError):
            self.replay(1)
        for index in (self.config.max_steps, self.config.max_steps + 10):
            controller = CoastingController()
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "max_steps"):
                self.replay(MidRolloutInjection(index, DisturbanceSpec("late")), controller=controller)
            self.assertEqual(controller.states, [])

    def test_invalid_mutated_speeds_are_rejected_without_clipping(self):
        for speed in (-1.0, 81.0, math.inf, math.nan):
            controller = CoastingController()

            def mutate(index, timestamp, position, heading, current_speed, state):
                return (position, heading, speed) if index == 2 else None

            with self.subTest(speed=speed), self.assertRaisesRegex(ValueError, "speed bounds"):
                simulate_closed_loop(
                    self.geometry, **self.initial, controller=controller, config=self.config, _pose_mutator=mutate
                )
            self.assertEqual(len(controller.states), 2)
        for delta in (-11.0, 71.0):
            controller = CoastingController()
            with self.subTest(delta=delta), self.assertRaisesRegex(ValueError, "disturbed speed"):
                self.replay(
                    MidRolloutInjection(2, DisturbanceSpec("invalid_speed", speed_delta_mps=delta)), controller=controller
                )
            self.assertEqual(len(controller.states), 2)

    def test_speed_bounds_and_last_budget_step_are_allowed(self):
        for delta, expected in ((-10.0, 0.0), (70.0, 80.0)):
            with self.subTest(delta=delta):
                outcome = self.replay(
                    MidRolloutInjection(self.config.max_steps - 1, DisturbanceSpec("boundary", speed_delta_mps=delta))
                )
                self.assertEqual(outcome.post_injection_state.speed_mps, expected)
                self.assertEqual(outcome.result.final_speed_mps, expected)
                self.assertEqual(outcome.metrics["post_rollout"]["steps"], 1)
                self.assertEqual(outcome.metrics["post_rollout"]["elapsed_s"], self.config.dt_s)
                self.assertEqual(len(outcome.result.steps), self.config.max_steps)

    def test_early_termination_raises_with_the_complete_unmodified_result(self):
        geometry = build_centerline_geometry(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        for elevation in (0.0, 6.0):
            initial = self.initial | {"initial_position": np.asarray([0.0, elevation, 0.0])}
            baseline = simulate_closed_loop(geometry, **initial, controller=CoastingController(), config=self.config)
            with self.subTest(elevation=elevation), self.assertRaises(InjectionNotReached) as raised:
                self.replay(
                    MidRolloutInjection(5, DisturbanceSpec("unreached", lateral_offset_m=3.0)),
                    geometry=geometry, **initial,
                )
            self.assertIsInstance(raised.exception, RuntimeError)
            self.assertEqual(raised.exception.step_index, 5)
            self.assertEqual(encoded(raised.exception.result), encoded(baseline))

    def test_suffix_requires_a_valid_recorded_index_and_positive_finite_dt(self):
        result = simulate_closed_loop(self.geometry, **self.initial, config=self.config)
        for index in (-1, True, 0.5, "1", len(result.steps), len(result.steps) + 1):
            with self.subTest(index=index), self.assertRaises(ValueError):
                summarize_rollout_suffix(result, index, dt_s=0.1)
        for dt in (0.0, -0.1, math.nan, math.inf, True, None):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                summarize_rollout_suffix(result, 0, dt_s=dt)
        with self.assertRaises(ValueError):
            summarize_rollout_suffix(replace(result, steps=()), 0, dt_s=0.1)


if __name__ == "__main__":
    unittest.main()

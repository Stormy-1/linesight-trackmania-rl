import unittest

import numpy as np

from trackmania_rl.gen_v3_stage import (
    GEN_V3_GATE,
    STAGE_1,
    StageController,
    derive_episode_metrics,
    derive_wheel_contact_proxies,
    materialize_cycle,
    rollback_required,
    stage_accepted,
    stage_map_cycle,
    summarize_wheel_contact_proxies,
)


class GenV3StageTests(unittest.TestCase):
    def test_stage_one_contains_only_hockolicious_and_a01(self):
        entries = materialize_cycle(stage_map_cycle())

        self.assertEqual({entry[0] for entry in entries}, {"hock", "A01"})
        self.assertEqual(sum(entry[0] == "hock" and entry[3] and entry[4] for entry in entries), 2)
        self.assertEqual(sum(entry[0] == "A01" and entry[3] and entry[4] for entry in entries), 2)
        self.assertEqual(sum(not entry[3] or not entry[4] for entry in entries), 2)
        self.assertEqual(STAGE_1.new_map, "A01")

    def test_metrics_include_finish_completion_timeout_and_contact_failures(self):
        from trackmania_rl.gen_v3_stage import summarize_map_results

        metrics = summarize_map_results(
            [
                {"finished": True, "timed_out": False, "completion_pct": 100, "contact_metric_available": True, "contact_or_barrier_failures": 1},
                {"finished": False, "timed_out": True, "completion_pct": 40, "contact_metric_available": True, "contact_or_barrier_failures": 2},
            ]
        )

        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["finish_rate"], 0.5)
        self.assertEqual(metrics["completion_pct_mean"], 70)
        self.assertEqual(metrics["timeout_rate"], 0.5)
        self.assertEqual(metrics["contact_or_barrier_failures"], 3)

    def test_rollback_and_acceptance_are_behavioral(self):
        healthy_hock = {"finish_rate": 1.0, "timeout_rate": 0.0}
        improving_a01 = {"finish_rate": 0.35, "timeout_rate": 0.1}

        self.assertFalse(rollback_required(healthy_hock, improving_a01))
        self.assertTrue(stage_accepted(0.20, healthy_hock, improving_a01))
        self.assertTrue(rollback_required({"finish_rate": 0.0, "timeout_rate": 1.0}, {"finish_rate": 0.0, "timeout_rate": 1.0}))
        self.assertEqual(GEN_V3_GATE.interval_frames, 250_000)

    def test_episode_metrics_distinguish_timeout_and_infrastructure_failure(self):
        zone_centers = [object()] * 1100
        timeout = derive_episode_metrics(
            zone_centers,
            {"furthest_zone_idx": 55},
            {"race_finished": False},
        )
        infrastructure = derive_episode_metrics(
            zone_centers,
            {"furthest_zone_idx": 55},
            {"race_finished": False, "tmi_protection_cutoff": True},
            rollout_crashed=True,
        )

        self.assertTrue(timeout["timed_out"])
        self.assertFalse(timeout["infrastructure_failure"])
        self.assertFalse(infrastructure["timed_out"])
        self.assertTrue(infrastructure["infrastructure_failure"])
        self.assertFalse(infrastructure["contact_metric_available"])

    def test_controller_records_baseline_then_accepts_only_a_complete_window(self):
        controller = StageController(stage_start_frames=0)
        healthy_hock = {"finished": True, "timed_out": False, "completion_pct": 100}
        baseline_a01 = {"finished": False, "timed_out": True, "completion_pct": 10}
        improved_a01 = {"finished": True, "timed_out": False, "completion_pct": 100}

        for _ in range(4):
            self.assertIsNone(controller.record_episode("hock", healthy_hock, 1, is_evaluation=True))
            self.assertIsNone(controller.record_episode("A01", baseline_a01, 1, is_evaluation=True))
        self.assertIsNone(controller.record_episode("hock", healthy_hock, 1, is_evaluation=True))
        self.assertIsNone(controller.record_episode("A01", baseline_a01, 1, is_evaluation=True))
        self.assertEqual(controller.baseline_new_map_finish_rate, 0.0)

        for _ in range(4):
            self.assertIsNone(controller.record_episode("hock", healthy_hock, 250_001, is_evaluation=True))
            self.assertIsNone(controller.record_episode("A01", improved_a01, 250_001, is_evaluation=True))
        self.assertIsNone(controller.record_episode("hock", healthy_hock, 250_001, is_evaluation=True))
        self.assertEqual(controller.record_episode("A01", improved_a01, 250_001, is_evaluation=True), "accepted")

    def test_controller_does_not_create_baseline_from_infrastructure_failure(self):
        controller = StageController(stage_start_frames=0)
        healthy_hock = {"finished": True, "timed_out": False, "completion_pct": 100}
        failed_a01 = {
            "finished": False,
            "timed_out": False,
            "infrastructure_failure": True,
            "completion_pct": 0,
        }

        for _ in range(5):
            self.assertIsNone(controller.record_episode("hock", healthy_hock, 1, is_evaluation=True))
            decision = controller.record_episode("A01", failed_a01, 1, is_evaluation=True)

        self.assertEqual(decision, "infrastructure_failure")
        self.assertIsNone(controller.baseline_new_map_finish_rate)

    def test_controller_rejects_an_initial_all_timeout_window(self):
        controller = StageController(stage_start_frames=0)
        timed_out = {"finished": False, "timed_out": True, "completion_pct": 0}

        for _ in range(4):
            self.assertIsNone(controller.record_episode("hock", timed_out, 1, is_evaluation=True))
            self.assertIsNone(controller.record_episode("A01", timed_out, 1, is_evaluation=True))
        self.assertIsNone(controller.record_episode("hock", timed_out, 1, is_evaluation=True))
        self.assertEqual(
            controller.record_episode("A01", timed_out, 1, is_evaluation=True),
            "rollback",
        )
        self.assertIsNone(controller.baseline_new_map_finish_rate)
        self.assertIn("initial evaluation window unusable", controller.reason)

    def test_controller_detects_post_baseline_rollback_before_next_acceptance_boundary(self):
        controller = StageController(stage_start_frames=0)
        healthy_hock = {"finished": True, "timed_out": False, "completion_pct": 100}
        baseline_a01 = {"finished": False, "timed_out": True, "completion_pct": 10}
        failed_hock = {"finished": False, "timed_out": True, "completion_pct": 0}

        for _ in range(5):
            controller.record_episode("hock", healthy_hock, 1, is_evaluation=True)
            controller.record_episode("A01", baseline_a01, 1, is_evaluation=True)
        self.assertEqual(controller.baseline_new_map_finish_rate, 0.0)

        for _ in range(4):
            self.assertIsNone(controller.record_episode("hock", failed_hock, 2, is_evaluation=True))
            self.assertIsNone(controller.record_episode("A01", baseline_a01, 2, is_evaluation=True))
        self.assertIsNone(controller.record_episode("hock", failed_hock, 2, is_evaluation=True))
        self.assertEqual(controller.record_episode("A01", baseline_a01, 2, is_evaluation=True), "rollback")
        self.assertEqual(controller.reason, "Hockolicious regression or simultaneous map timeouts")

    def test_wheel_contact_proxies_are_not_barrier_collisions(self):
        sample = np.zeros(32, dtype=np.float32)
        sample[0] = 1
        sample[4:8] = 1
        sample[16] = 1
        sample[20 + 1] = 1
        metrics = derive_wheel_contact_proxies(sample)

        self.assertEqual(metrics["sliding_count"], 1)
        self.assertEqual(metrics["grounded_sliding_count"], 1)
        self.assertEqual(metrics["material_groups"], (0, 1, 4, 4))
        self.assertIsNone(metrics["exact_barrier_collisions"])

    def test_wheel_proxy_summary_ignores_finish_sentinel_and_counts_contact_loss(self):
        previous = np.zeros(32, dtype=np.float32)
        previous[4:8] = 1
        current = previous.copy()
        current[4] = 0

        summary = summarize_wheel_contact_proxies([previous, current, np.asarray(np.nan)])

        self.assertEqual(summary["telemetry_samples"], 2)
        self.assertEqual(summary["ground_contact_loss_count"], 1)
        self.assertEqual(summary["exact_barrier_collisions"], None)


if __name__ == "__main__":
    unittest.main()

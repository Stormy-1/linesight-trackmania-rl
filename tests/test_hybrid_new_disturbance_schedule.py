import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_hybrid_new_disturbance_schedule import (
    DEFAULT_STEPS,
    NEW_HELD_OUT_DISTURBANCES,
    build_new_disturbance_schedule_report,
)


def _step_report(step, family, *, assignments=None):
    return {
        "promotion_allowed": False,
        "independent_confirmation": False,
        "held_out_only_disturbances": family,
        "validation": {"valid": True, "issues": []},
        "simulator": {"injection_step": step, "injection_time_s": step * 0.1},
        "selection": {"eligible_candidates": ["candidate_a"]},
        "candidates": [
            {"configuration_id": "candidate_a", "train_rank": 1},
            {"configuration_id": "candidate_b", "train_rank": 2},
        ],
        "split": {"assignments": assignments or {"case_a": "train", "case_b": "held_out"}},
        "aggregates": {
            "live_defaults_reference": {
                "held_out": {
                    "case_count": 1,
                    "recovery_activated_count": 1,
                    "recovery_success_rate": 0.5,
                    "recovery_failure_count": 1,
                    "progress_continuity_rate": 1.0,
                    "projected_progress_completion_rate": 1.0,
                    "stable_rejoin_rate": 1.0,
                    "mean_recovery_latency_s": 0.5,
                    "total_redivergence_count": 0,
                    "finished_rate": 0.0,
                }
            }
        },
    }


class HybridNewDisturbanceScheduleTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parents[1]))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.directory, ignore_errors=True))
        self.family = [
            {
                "name": item.name,
                "lateral_offset_m": item.lateral_offset_m,
                "heading_offset_rad": item.heading_offset_rad,
                "speed_delta_mps": item.speed_delta_mps,
            }
            for item in NEW_HELD_OUT_DISTURBANCES
        ]
        primary = {
            "schedule": {"injection_steps": [5, 10, 20]},
        }
        heldout = {
            "schedule": {"injection_steps": [35, 50, 80]},
        }
        self.primary = self.directory / "primary.json"
        self.heldout = self.directory / "heldout.json"
        self.primary.write_text(json.dumps(primary), encoding="utf-8")
        self.heldout.write_text(json.dumps(heldout), encoding="utf-8")

    def _write_reports(self, steps, *, family=None):
        paths = []
        for index, step in enumerate(steps):
            path = self.directory / f"report_{index}.json"
            path.write_text(json.dumps(_step_report(step, self.family if family is None else family)), encoding="utf-8")
            paths.append(path)
        return paths

    def test_new_family_and_timing_schedule_are_separated(self):
        paths = self._write_reports(DEFAULT_STEPS)
        result = build_new_disturbance_schedule_report(
            paths,
            existing_schedule_paths=(self.primary, self.heldout),
            expected_steps=DEFAULT_STEPS,
        )
        self.assertTrue(result["validation"]["valid"])
        self.assertTrue(result["validation"]["family_matches_all_reports"])
        self.assertTrue(result["validation"]["disjoint_from_prior_schedules"])
        self.assertFalse(result["promotion_allowed"])
        self.assertEqual(result["schedule"]["stable_eligible_candidates"], ["candidate_a"])

    def test_overlap_or_family_mismatch_blocks_recommendation(self):
        bad_family = [dict(self.family[0], speed_delta_mps=8.0)]
        paths = self._write_reports([15, 20], family=bad_family)
        result = build_new_disturbance_schedule_report(
            paths,
            existing_schedule_paths=(self.primary, self.heldout),
            expected_steps=(15, 20),
        )
        self.assertFalse(result["validation"]["valid"])
        self.assertFalse(result["validation"]["family_matches_all_reports"])
        self.assertFalse(result["validation"]["disjoint_from_prior_schedules"])
        self.assertIsNone(result["schedule"]["diagnostic_recommendation"])
        self.assertEqual(result["schedule"]["stable_eligible_candidates"], [])


if __name__ == "__main__":
    unittest.main()

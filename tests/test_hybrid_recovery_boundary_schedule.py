import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_hybrid_recovery_boundary_schedule import (
    BOUNDARY_HELD_OUT_DISTURBANCES,
    DEFAULT_STEPS,
    build_boundary_schedule_report,
)


def _step_report(step, family):
    return {
        "promotion_allowed": False,
        "independent_confirmation": False,
        "held_out_only_disturbances": family,
        "simulator": {"injection_step": step, "injection_time_s": step * 0.1},
        "selection": {"eligible_candidates": ["candidate_a"]},
        "candidates": [{"configuration_id": "candidate_a", "train_rank": 1}],
        "validation": {"valid": True, "issues": []},
        "split": {"assignments": {"case_a": "train", "case_b": "held_out"}},
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


class HybridRecoveryBoundaryScheduleTests(unittest.TestCase):
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
            for item in BOUNDARY_HELD_OUT_DISTURBANCES
        ]
        schedules = {
            "primary.json": {"schedule": {"injection_steps": [5, 10, 20]}},
            "heldout.json": {"schedule": {"injection_steps": [35, 50, 80]}},
            "new.json": {"schedule": {"injection_steps": [15, 30, 60]}},
        }
        self.schedule_paths = []
        for name, payload in schedules.items():
            path = self.directory / name
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.schedule_paths.append(path)

    def _reports(self, steps=DEFAULT_STEPS, family=None):
        family = self.family if family is None else family
        paths = []
        for index, step in enumerate(steps):
            path = self.directory / f"report_{index}.json"
            path.write_text(json.dumps(_step_report(step, family)), encoding="utf-8")
            paths.append(path)
        return paths

    def test_boundary_reports_require_new_family_and_disjoint_timing(self):
        result = build_boundary_schedule_report(
            self._reports(), existing_schedule_paths=self.schedule_paths, expected_steps=DEFAULT_STEPS
        )
        self.assertTrue(result["validation"]["valid"])
        self.assertTrue(result["validation"]["family_matches_all_reports"])
        self.assertTrue(result["validation"]["disjoint_from_prior_schedules"])
        self.assertFalse(result["promotion_allowed"])

    def test_overlap_or_family_mismatch_blocks_recommendation(self):
        bad_family = [dict(self.family[0], lateral_offset_m=9.0)]
        result = build_boundary_schedule_report(
            self._reports(steps=(25, 30), family=bad_family),
            existing_schedule_paths=self.schedule_paths,
            expected_steps=(25, 30),
        )
        self.assertFalse(result["validation"]["valid"])
        self.assertFalse(result["validation"]["family_matches_all_reports"])
        self.assertFalse(result["validation"]["disjoint_from_prior_schedules"])
        self.assertIsNone(result["schedule"]["diagnostic_recommendation"])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_hybrid_midrollout_heldout_schedule import build_held_out_schedule_report


def _schedule_report(step, *, promotion_allowed=False, assignments=None):
    return {
        "scope": "offline_fixed_midrollout_injection_schedule_stability",
        "promotion_allowed": promotion_allowed,
        "independent_confirmation": False,
        "validation": {"valid": True, "issues": []},
        "schedule": {
            "injection_steps": [step],
            "injection_times_s": [step * 0.1],
            "report_count": 1,
            "same_case_assignments_across_reports": True,
            "candidate_eligibility_rule": "candidate must be eligible at every injection step",
            "diagnostic_recommendation": None,
            "stable_eligible_candidates": [],
        },
        "reports": [],
    }


def _step_report(step, eligible):
    return {
        "promotion_allowed": False,
        "independent_confirmation": False,
        "validation": {"valid": True, "issues": []},
        "simulator": {"injection_step": step, "injection_time_s": step * 0.1},
        "selection": {"eligible_candidates": eligible},
        "candidates": [
            {"configuration_id": "candidate_a", "train_rank": 1},
            {"configuration_id": "candidate_b", "train_rank": 2},
        ],
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


class HybridMidrolloutHeldoutScheduleTests(unittest.TestCase):
    def _paths(self, primary_steps, held_out_reports):
        directory = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parents[1]))
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        primary = _schedule_report(primary_steps[0])
        primary["schedule"]["injection_steps"] = primary_steps
        primary["schedule"]["injection_times_s"] = [step * 0.1 for step in primary_steps]
        primary_path = directory / "primary.json"
        primary_path.write_text(json.dumps(primary), encoding="utf-8")
        report_paths = []
        for index, report in enumerate(held_out_reports):
            path = directory / f"heldout_{index}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            report_paths.append(path)
        return primary_path, report_paths

    def test_requires_disjoint_declared_heldout_schedule(self):
        primary, reports = self._paths([5, 10, 20], [_step_report(35, ["candidate_a"]), _step_report(50, ["candidate_a"]), _step_report(80, ["candidate_a"])])
        result = build_held_out_schedule_report(reports, primary_schedule_path=primary)
        self.assertTrue(result["validation"]["valid"])
        self.assertTrue(result["validation"]["primary_and_held_out_steps_disjoint"])
        self.assertEqual(result["held_out_schedule"]["injection_steps"], [35, 50, 80])
        self.assertFalse(result["promotion_allowed"])

    def test_overlapping_or_unexpected_steps_block_the_report(self):
        primary, reports = self._paths([5, 10, 20], [_step_report(10, ["candidate_a"]), _step_report(50, ["candidate_a"])])
        result = build_held_out_schedule_report(
            reports,
            primary_schedule_path=primary,
            expected_steps=(10, 50),
        )
        self.assertFalse(result["validation"]["valid"])
        self.assertFalse(result["validation"]["primary_and_held_out_steps_disjoint"])
        self.assertIsNone(result["held_out_schedule"]["diagnostic_recommendation"])
        self.assertEqual(result["held_out_schedule"]["stable_eligible_candidates"], [])

    def test_invalid_primary_schedule_is_rejected(self):
        primary, reports = self._paths([5, 10, 20], [_step_report(35, ["candidate_a"]), _step_report(50, ["candidate_a"])])
        data = json.loads(primary.read_text(encoding="utf-8"))
        data["promotion_allowed"] = True
        primary.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            build_held_out_schedule_report(reports, primary_schedule_path=primary, expected_steps=(35, 50))


if __name__ == "__main__":
    unittest.main()

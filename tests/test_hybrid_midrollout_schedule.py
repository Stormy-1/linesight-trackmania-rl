import json
from pathlib import Path
import tempfile
import unittest

from scripts.sweep_hybrid_midrollout_injection_schedule import build_schedule_report


def _report(step, eligible, *, assignments=None, promotion_allowed=False):
    candidates = [
        {"configuration_id": "candidate_a", "train_rank": 1},
        {"configuration_id": "candidate_b", "train_rank": 2},
    ]
    return {
        "promotion_allowed": promotion_allowed,
        "independent_confirmation": False,
        "validation": {"valid": True, "issues": []},
        "simulator": {"injection_step": step, "injection_time_s": step * 0.1},
        "selection": {"eligible_candidates": eligible},
        "candidates": candidates,
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


class HybridMidrolloutScheduleTests(unittest.TestCase):
    def _paths(self, reports):
        directory = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parents[1]))
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        paths = []
        for index, report in enumerate(reports):
            path = directory / f"report_{index}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            paths.append(path)
        return paths

    def test_requires_candidate_eligibility_at_every_schedule_step(self):
        paths = self._paths([
            _report(5, ["candidate_a"]),
            _report(10, ["candidate_a", "candidate_b"]),
            _report(20, ["candidate_b"]),
        ])
        result = build_schedule_report(paths)
        self.assertEqual(result["schedule"]["injection_steps"], [5, 10, 20])
        self.assertEqual(result["schedule"]["stable_eligible_candidates"], [])
        self.assertIsNone(result["schedule"]["diagnostic_recommendation"])
        self.assertTrue(result["validation"]["valid"])

    def test_reports_one_stable_diagnostic_recommendation_without_promotion(self):
        paths = self._paths([
            _report(20, ["candidate_a"]),
            _report(5, ["candidate_a", "candidate_b"]),
        ])
        result = build_schedule_report(paths)
        self.assertEqual(result["schedule"]["injection_steps"], [5, 20])
        self.assertEqual(result["schedule"]["stable_eligible_candidates"], ["candidate_a"])
        self.assertEqual(result["schedule"]["diagnostic_recommendation"], "candidate_a")
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["independent_confirmation"])

    def test_source_assignments_must_match(self):
        paths = self._paths([
            _report(5, ["candidate_a"], assignments={"case_a": "train"}),
            _report(10, ["candidate_a"], assignments={"case_a": "held_out"}),
        ])
        result = build_schedule_report(paths)
        self.assertFalse(result["validation"]["valid"])
        self.assertIsNone(result["schedule"]["diagnostic_recommendation"])

    def test_rejects_duplicate_schedule_steps_and_invalid_reports(self):
        duplicate = self._paths([_report(5, ["candidate_a"]), _report(5, ["candidate_a"])])
        with self.assertRaisesRegex(ValueError, "unique"):
            build_schedule_report(duplicate)

        invalid = self._paths([_report(5, ["candidate_a"], promotion_allowed=True), _report(10, ["candidate_a"])])
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            build_schedule_report(invalid)


if __name__ == "__main__":
    unittest.main()

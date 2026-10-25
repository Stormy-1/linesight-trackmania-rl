import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_hybrid_failure_taxonomy import build_failure_taxonomy, classify_case


def _case(
    *,
    key="case",
    success: bool | None = True,
    activated=True,
    completed=True,
    stable=True,
    continuity=True,
    redivergence=0,
    backward=0.0,
    latency: float | None = 1.0,
):
    return {
        "case_key": key,
        "status": "ok",
        "split": "held_out",
        "disturbance": "disturbance",
        "metrics": {
            "recovery_activated": activated,
            "recovery_completed": completed,
            "recovery_success": success,
            "progress_continuity": continuity,
            "termination_reason": "progress_complete_finish_gate_failed",
            "backward_progress_distance_m": backward,
            "recovery": {
                "stable_rejoin": stable,
                "redivergence_count": redivergence,
                "recovery_latency_s": latency,
            },
        },
    }


class HybridFailureTaxonomyTests(unittest.TestCase):
    def test_classifies_success_failure_and_nonactivation_without_rewriting_metrics(self):
        success = classify_case(_case())
        self.assertEqual(success["primary_outcome"], "recovery_success")
        self.assertIn("recovery_success", success["outcome_labels"])

        failure = classify_case(_case(success=False, continuity=False, backward=1.5))
        self.assertEqual(failure["primary_outcome"], "stable_rejoin_with_backward_progress")
        self.assertEqual(
            failure["outcome_labels"],
            ["backward_progress_after_injection", "recovery_failure"],
        )

        nonactivated = classify_case(
            _case(success=None, activated=False, completed=False, stable=False, latency=None)
        )
        self.assertEqual(nonactivated["primary_outcome"], "not_activated")
        self.assertIn("no_recovery_observation", nonactivated["outcome_labels"])

    def test_prioritizes_redivergence_and_aggregates_deterministically(self):
        failure = classify_case(_case(success=False, continuity=False, redivergence=1, backward=0.5))
        self.assertEqual(failure["primary_outcome"], "stable_rejoin_then_redivergence")
        self.assertIn("backward_progress_after_injection", failure["outcome_labels"])

        report = {
            "simulator": {"injection_step": 10, "injection_time_s": 1.0},
            "maps": {
                "A01-Race": {
                    "cases_by_profile": {
                        "live_defaults_reference": [
                            _case(key="a", success=False, continuity=False, backward=2.0),
                            _case(key="b"),
                        ]
                    }
                }
            },
            "aggregates": {"live_defaults_reference": {"held_out": {"case_count": 2}}},
        }
        directory = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parents[1]))
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        first = directory / "hybrid_midrollout_recovery_comparison.json"
        second = directory / "hybrid_midrollout_recovery_comparison_copy.json"
        first.write_text(json.dumps(report), encoding="utf-8")
        second.write_text(json.dumps(report), encoding="utf-8")

        result = build_failure_taxonomy((first, second))
        self.assertEqual(result["counts"]["held_out_case_records"], 4)
        self.assertEqual(result["counts"]["baseline_case_records"], 4)
        grouped = result["baseline_by_schedule_group_and_disturbance"]["primary_schedule"]["disturbance"]
        self.assertEqual(grouped["primary_outcomes"]["recovery_success"], 2)
        self.assertEqual(grouped["primary_outcomes"]["stable_rejoin_with_backward_progress"], 2)
        self.assertEqual(grouped["backward_progress_case_count"], 2)


if __name__ == "__main__":
    unittest.main()

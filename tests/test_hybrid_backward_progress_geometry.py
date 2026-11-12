import unittest

from scripts.analyze_hybrid_backward_progress_geometry import build_geometry_report


class HybridBackwardProgressGeometryTests(unittest.TestCase):
    def test_classifies_backward_events_from_heading_alignment(self):
        report = build_geometry_report(
            {
                "records": [
                    {
                        "map": "A01-Race",
                        "profile": "default",
                        "disturbance": "severe",
                        "injection_step": 10,
                        "case_key": "case-1",
                        "historical_metrics": {
                            "progress_continuity": False,
                            "backward_progress_distance_m": 0.8,
                        },
                        "trace": {
                            "first_backward_event": {
                                "step_index": 11,
                                "state": {
                                    "speed_mps": 20.0,
                                    "heading_error_rad": 3.141592653589793,
                                },
                            }
                        },
                    },
                    {
                        "historical_metrics": {"progress_continuity": True},
                        "trace": {},
                    },
                ]
            }
        )
        self.assertEqual(report["overall"]["event_count"], 1)
        self.assertEqual(report["overall"]["negative_along_tangent_count"], 1)
        self.assertEqual(report["events"][0]["classification"], "negative_along_tangent")
        self.assertAlmostEqual(report["events"][0]["estimated_along_tangent_speed_mps"], -20.0)

    def test_rejects_non_positive_classification_parameters(self):
        with self.assertRaises(ValueError):
            build_geometry_report({"records": []}, dt_s=0.0)
        with self.assertRaises(ValueError):
            build_geometry_report({"records": []}, negative_threshold_mps=0.0)


if __name__ == "__main__":
    unittest.main()

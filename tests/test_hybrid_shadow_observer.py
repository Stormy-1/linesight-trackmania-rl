import unittest
from pathlib import Path

from scripts.evaluate_hybrid_shadow_observer import build_report, default_observer_configs


class HybridShadowObserverTests(unittest.TestCase):
    def test_default_configs_are_observer_only(self):
        configs = default_observer_configs()
        self.assertEqual(len(configs), 4)
        self.assertEqual(len({config.name for config in configs}), 4)

    def test_saved_cases_produce_event_only_report(self):
        report = build_report()
        self.assertTrue(report["observer_only"])
        self.assertFalse(report["actions_modified"])
        self.assertEqual(len(report["cases"]), 2)
        self.assertTrue(all("observers" in case for case in report["cases"]))
        self.assertFalse(any("steering" in observer for case in report["cases"] for observer in case["observers"]))

    def test_empty_configs_are_rejected(self):
        with self.assertRaises(ValueError):
            build_report(cases=(("A01", Path("missing.npy"), Path("missing.csv")),), configs=())


if __name__ == "__main__":
    unittest.main()

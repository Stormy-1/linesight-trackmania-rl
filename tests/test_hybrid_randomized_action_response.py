import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_hybrid_randomized_action_response import build_report
from trackmania_rl.hybrid_telemetry import HybridTelemetryWriter


def _action_inputs(action: int) -> dict[str, bool]:
    return {
        "accelerate": action < 3,
        "brake": 6 <= action,
        "left": action in {1, 4, 7},
        "right": action in {2, 5, 8},
    }


def _write_capture(path: Path, *, noise: float = 0.0) -> None:
    timestamp = 0.0
    speed = 10.0
    position_x = 0.0
    with HybridTelemetryWriter(path, include_action_columns=True, include_randomization_columns=True) as writer:
        for action in range(9):
            for index in range(51):
                writer.write_sample(
                    timestamp,
                    np.asarray([position_x, 0.0, 0.0]),
                    np.asarray([1.0, 0.0, 0.0]),
                    speed,
                    action_idx=action,
                    action_inputs=_action_inputs(action),
                    action_was_greedy=False,
                    action_is_randomized=True,
                )
                response = float(action * 2.0 + noise * ((index % 3) - 1))
                timestamp += 0.1
                position_x += speed * 0.1
                speed += response * 0.1


class HybridRandomizedActionResponseTests(unittest.TestCase):
    def test_disjoint_quota_and_heldout_baseline_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit = tuple(root / f"fit_{index}.csv" for index in range(4))
            heldout = tuple(root / f"heldout_{index}.csv" for index in range(2))
            for path in (*fit, *heldout):
                _write_capture(path, noise=0.05)
            report = build_report(fit, heldout, minimum_randomized_transitions=50)
            self.assertTrue(report["source_splits_disjoint"])
            self.assertTrue(report["fit_quota"]["all_action_quotas_met"])
            self.assertTrue(report["heldout_quota"]["all_action_quotas_met"])
            self.assertTrue(report["identification_gate_passed"])
            self.assertTrue(report["causal_fitting_allowed"])
            self.assertFalse(report["promotion_allowed"])

    def test_rejects_boundary_crossing_source_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.csv"
            _write_capture(path)
            with self.assertRaises(ValueError):
                build_report((path,), (path,))


if __name__ == "__main__":
    unittest.main()

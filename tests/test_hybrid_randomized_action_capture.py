import unittest

import numpy as np

from trackmania_rl.hybrid_randomized_capture import SeededRandomizedActionPolicy


def _nominal_policy(action: int):
    def policy(_image, _float_inputs):
        q_values = np.arange(12, dtype=np.float32)
        return action, action == int(np.argmax(q_values)), float(q_values.max()), q_values

    return policy


class HybridRandomizedActionCaptureTests(unittest.TestCase):
    def test_seed_reproduces_intervention_and_action_sequence(self):
        first = SeededRandomizedActionPolicy(_nominal_policy(11), seed=42)
        second = SeededRandomizedActionPolicy(_nominal_policy(11), seed=42)
        changed = SeededRandomizedActionPolicy(_nominal_policy(11), seed=43)
        first_sequence = [first(None, None)[:2] + (first.last_action_is_randomized,) for _ in range(40)]
        second_sequence = [second(None, None)[:2] + (second.last_action_is_randomized,) for _ in range(40)]
        changed_sequence = [changed(None, None)[:2] + (changed.last_action_is_randomized,) for _ in range(40)]
        self.assertEqual(first_sequence, second_sequence)
        self.assertNotEqual(first_sequence, changed_sequence)

    def test_intervention_schedule_does_not_depend_on_nominal_action(self):
        first = SeededRandomizedActionPolicy(_nominal_policy(0), seed=7)
        second = SeededRandomizedActionPolicy(_nominal_policy(8), seed=7)
        first_flags = []
        second_flags = []
        first_actions = []
        second_actions = []
        for _ in range(100):
            first_actions.append(first(None, None)[0])
            first_flags.append(first.last_action_is_randomized)
            second_actions.append(second(None, None)[0])
            second_flags.append(second.last_action_is_randomized)
        self.assertEqual(first_flags, second_flags)
        self.assertNotEqual(first_actions, second_actions)
        self.assertTrue(all(action in range(9) for action, randomized in zip(first_actions, first_flags) if randomized))

    def test_rejects_invalid_design(self):
        with self.assertRaises(ValueError):
            SeededRandomizedActionPolicy(_nominal_policy(0), seed=1, intervention_probability=1.1)
        with self.assertRaises(ValueError):
            SeededRandomizedActionPolicy(_nominal_policy(0), seed=1, randomized_action_indices=(0, 0))


if __name__ == "__main__":
    unittest.main()

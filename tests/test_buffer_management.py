import unittest

from trackmania_rl.buffer_management import select_test_transitions


class BufferRoutingTests(unittest.TestCase):
    def test_test_buffer_assignment_is_per_transition(self):
        self.assertEqual(
            select_test_transitions(4, 0.05, random_values=[0.01, 0.04, 0.06, 0.99]),
            [True, True, False, False],
        )

    def test_test_buffer_assignment_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            select_test_transitions(-1, 0.05)
        with self.assertRaises(ValueError):
            select_test_transitions(1, 1.1)


if __name__ == "__main__":
    unittest.main()

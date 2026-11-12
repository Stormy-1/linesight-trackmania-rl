import unittest

import torch

from trackmania_rl.multiprocess.learner_process import _guard_non_finite_optimizer_step, _non_finite_batch_reason


class _IdentityScaler:
    def scale(self, value):
        return value


class LearnerSafetyTests(unittest.TestCase):
    def test_non_finite_gradient_blocks_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scaler = _IdentityScaler()
        parameter.grad = torch.tensor(float("nan"))

        with _guard_non_finite_optimizer_step(optimizer, scaler) as state:
            scaler.scale(torch.tensor(1.0))
            grad_norm = torch.nn.utils.clip_grad_norm_([parameter], max_norm=10.0)
            optimizer.step()

        self.assertTrue(torch.isnan(grad_norm))
        self.assertTrue(state["blocked"])
        self.assertIn("gradient_norm_non_finite", state["reasons"])
        self.assertEqual(
            _non_finite_batch_reason(torch.tensor(1.0), grad_norm, state),
            "gradient_norm_non_finite, gradient_values_non_finite",
        )
        self.assertEqual(parameter.item(), 1.0)

    def test_finite_gradient_allows_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scaler = _IdentityScaler()
        parameter.grad = torch.tensor(1.0)

        with _guard_non_finite_optimizer_step(optimizer, scaler) as state:
            scaler.scale(torch.tensor(1.0))
            grad_norm = torch.nn.utils.clip_grad_norm_([parameter], max_norm=10.0)
            optimizer.step()

        self.assertTrue(torch.isfinite(grad_norm))
        self.assertFalse(state["blocked"])
        self.assertIsNone(_non_finite_batch_reason(torch.tensor(1.0), grad_norm, state))
        self.assertEqual(parameter.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

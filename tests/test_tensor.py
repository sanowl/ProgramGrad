import math
import unittest

from programgrad import Tensor
from programgrad.tensor import hard_data


class TensorTests(unittest.TestCase):
    def test_quadratic_gradient(self):
        x = Tensor(2.0, requires_grad=True, name="x")
        y = x * x + 3.0 * x + 1.0
        y.backward()
        self.assertAlmostEqual(y.data, 11.0)
        self.assertAlmostEqual(x.grad, 7.0)

    def test_exp_log_sigmoid_gradient(self):
        x = Tensor(0.4, requires_grad=True, name="x")
        y = x.exp().log() + x.sigmoid()
        y.backward()
        expected = 1.0 + (1.0 / (1.0 + math.exp(-0.4))) * (1.0 - 1.0 / (1.0 + math.exp(-0.4)))
        self.assertAlmostEqual(x.grad, expected)

    def test_hard_value_propagates_through_arithmetic(self):
        soft = Tensor(0.25, requires_grad=True, name="soft")
        soft.hard_value = 2.0
        out = (soft * 3.0 + 1.0).relu()
        self.assertAlmostEqual(out.data, 1.75)
        self.assertAlmostEqual(out.hard_value, 7.0)
        self.assertAlmostEqual(hard_data(out), 7.0)
        self.assertAlmostEqual(hard_data(out.detach()), 7.0)

    def test_hard_shadow_domain_failure_does_not_abort_soft_forward(self):
        divided = Tensor(1.0, requires_grad=True)
        divided.hard_value = 1.0
        zero = Tensor(1.0)
        zero.hard_value = 0.0
        quotient = divided / zero
        self.assertAlmostEqual(quotient.data, 1.0)
        self.assertIsNone(quotient.hard_value)

        positive = Tensor(2.0, requires_grad=True)
        positive.hard_value = -1.0
        logged = positive.log()
        self.assertAlmostEqual(logged.data, math.log(2.0))
        self.assertIsNone(logged.hard_value)

        base = Tensor(4.0, requires_grad=True)
        base.hard_value = -4.0
        rooted = base**0.5
        self.assertAlmostEqual(rooted.data, 2.0)
        self.assertIsNone(rooted.hard_value)

    def test_zero_to_zero_power_has_finite_base_gradient(self):
        base = Tensor(0.0, requires_grad=True, name="base")
        out = base**0.0
        out.backward()
        self.assertAlmostEqual(out.data, 1.0)
        self.assertAlmostEqual(base.grad, 0.0)

    def test_backward_handles_deep_graph_without_recursion(self):
        x = Tensor(1.0, requires_grad=True)
        out = x
        for _ in range(2_000):
            out = out + 1.0
        out.backward()
        self.assertAlmostEqual(x.grad, 1.0)

    def test_invalid_hard_shadow_is_deferred_but_never_silently_softened(self):
        denominator = Tensor(2.0)
        denominator.hard_value = 0.0
        quotient = Tensor(1.0) / denominator

        self.assertAlmostEqual(quotient.data, 0.5)
        self.assertIsNone(quotient.hard_value)
        self.assertIn("ZeroDivisionError", quotient.hard_error)
        with self.assertRaisesRegex(ValueError, "hard-program value is unavailable"):
            hard_data(quotient)
        with self.assertRaisesRegex(ValueError, "hard-program value is unavailable"):
            hard_data(quotient.detach())

    def test_power_reports_real_domain_errors_clearly(self):
        with self.assertRaisesRegex(ValueError, "real number domain"):
            Tensor(-1.0) ** 0.5
        with self.assertRaisesRegex(ZeroDivisionError, "negative power"):
            Tensor(0.0) ** -1.0


if __name__ == "__main__":
    unittest.main()

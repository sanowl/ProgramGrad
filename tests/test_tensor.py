import math
import unittest

from programgrad import Tensor


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


if __name__ == "__main__":
    unittest.main()


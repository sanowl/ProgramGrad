import unittest

from programgrad import Tensor, gradcheck, soft_if


class GradcheckTests(unittest.TestCase):
    def test_tensor_expression_gradcheck(self):
        result = gradcheck(lambda x, y: x * y + x.sigmoid() + y**2, [0.3, -0.4])
        self.assertTrue(result.passed, result)

    def test_soft_if_surrogate_gradcheck(self):
        def fn(x: Tensor, threshold: Tensor) -> Tensor:
            return soft_if(x - threshold, 2.0 * x, x**2, beta=3.0)

        result = gradcheck(fn, [0.7, 1.1])
        self.assertTrue(result.passed, result)


if __name__ == "__main__":
    unittest.main()


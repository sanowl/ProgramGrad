import math
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

    def test_gradcheck_validates_numeric_configuration(self):
        for kwargs in ({"eps": 0.0}, {"eps": math.inf}, {"atol": -1.0}, {"rtol": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gradcheck(lambda x: x**2, [1.0], **kwargs)
        with self.assertRaisesRegex(ValueError, "inputs must contain only finite"):
            gradcheck(lambda x: x**2, [math.inf])

    def test_gradcheck_checks_perturbed_return_types(self):
        calls = 0

        def inconsistent(x: Tensor):
            nonlocal calls
            calls += 1
            return x if calls == 1 else x.data

        with self.assertRaisesRegex(TypeError, "must return a Tensor"):
            gradcheck(inconsistent, [1.0])

    def test_more_tensor_operations_gradcheck(self):
        cases = [
            (lambda x, y: x / y + x.tanh(), [1.2, 2.3]),
            (lambda x, y: x**y, [1.3, 0.7]),
            (lambda x: x.relu() + (-x).sigmoid(), [0.8]),
        ]
        for fn, inputs in cases:
            with self.subTest(inputs=inputs):
                result = gradcheck(fn, inputs)
                self.assertTrue(result.passed, result)


if __name__ == "__main__":
    unittest.main()

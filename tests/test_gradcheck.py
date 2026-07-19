import math
import unittest

from programgrad import Tensor, bounded_loop, gradcheck, soft_if, soft_select


class GradcheckTests(unittest.TestCase):
    def test_tensor_expression_gradcheck(self):
        result = gradcheck(lambda x, y: x * y + x.sigmoid() + y**2, [0.3, -0.4])
        self.assertTrue(result.passed, result)

    def test_soft_if_surrogate_gradcheck(self):
        def fn(x: Tensor, threshold: Tensor) -> Tensor:
            return soft_if(x - threshold, 2.0 * x, x**2, beta=3.0)

        result = gradcheck(fn, [0.7, 1.1])
        self.assertTrue(result.passed, result)

    def test_soft_select_softmax_gradcheck(self):
        def fn(score_a: Tensor, score_b: Tensor) -> Tensor:
            return soft_select([1.0, 3.0], [score_a, score_b], mode="softmax", tau=1.0)

        result = gradcheck(fn, [0.2, 0.8])
        self.assertTrue(result.passed, result)

    def test_bounded_loop_survival_gradcheck(self):
        def fn(initial: Tensor) -> Tensor:
            return bounded_loop(
                initial,
                3,
                lambda _index, state: state + 0.5,
                lambda index, _state: 1.0 if index < 2 else -1.0,
                beta=5.0,
                mode="survival",
            )

        result = gradcheck(fn, [0.0])
        self.assertTrue(result.passed, result)

    def test_gradcheck_validates_numeric_configuration(self):
        cases = [
            ({"eps": 0.0}, [1.0], ValueError, None),
            ({"eps": math.inf}, [1.0], ValueError, None),
            ({"atol": -1.0}, [1.0], ValueError, None),
            ({"rtol": math.nan}, [1.0], ValueError, None),
            ({}, [math.inf], ValueError, "inputs must contain only finite"),
        ]
        for kwargs, inputs, error, match in cases:
            with self.subTest(kwargs=kwargs, inputs=inputs):
                if match is None:
                    with self.assertRaises(error):
                        gradcheck(lambda x: x**2, inputs, **kwargs)
                else:
                    with self.assertRaisesRegex(error, match):
                        gradcheck(lambda x: x**2, inputs, **kwargs)

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

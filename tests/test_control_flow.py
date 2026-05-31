import unittest

from programgrad import Tensor, diff_if, soft_argmax, soft_if, soft_select, trace


class ControlFlowTests(unittest.TestCase):
    def test_soft_if_gives_threshold_gradient_and_trace(self):
        x = Tensor(0.7, name="x")
        threshold = Tensor(1.2, requires_grad=True, name="threshold")

        with trace(mode="dual", relaxation="soft_gate", fidelity=True) as tr:
            score = x - threshold
            y = soft_if(score, true_value=2.0 * x, false_value=x**2, beta=5.0)
            loss = (y - 1.4) ** 2
            loss.backward()

        self.assertGreater(threshold.grad, 0.0)
        self.assertEqual(len(tr.branches), 1)
        self.assertEqual(tr.branches[0].selected_path, "false")
        self.assertIsNotNone(tr.branches[0].fidelity.output_gap)
        self.assertIn("surrogate", tr.gradient_ledger()[0]["bias_warning"])

    def test_pathwise_diff_if_records_no_soft_gate(self):
        x = Tensor(2.0, requires_grad=True, name="x")
        with trace(mode="pathwise", relaxation="none", fidelity=True) as tr:
            y = diff_if(x - 1.0, lambda: x * 3.0, lambda: x**2, mode="pathwise")
            y.backward()

        self.assertAlmostEqual(x.grad, 3.0)
        self.assertIsNone(tr.branches[0].condition.gate)

    def test_soft_select_records_search_node(self):
        s0 = Tensor(0.2, requires_grad=True, name="s0")
        s1 = Tensor(1.3, requires_grad=True, name="s1")
        with trace(mode="dual", relaxation="softmax_select", fidelity=True) as tr:
            y = soft_select([1.0, 3.0], [s0, s1], tau=0.7, candidate_names=["a", "b"])
            loss = (y - 3.0) ** 2
            loss.backward()

        self.assertEqual(len(tr.searches), 1)
        self.assertEqual(tr.searches[0].selected_index, 1)
        self.assertAlmostEqual(sum(tr.searches[0].soft_weights), 1.0)
        self.assertNotEqual(s0.grad, 0.0)

    def test_soft_argmax_weights_sum_to_one(self):
        weights = soft_argmax([0.0, 1.0, 2.0], tau=1.0)
        self.assertAlmostEqual(sum(weight.data for weight in weights), 1.0)


if __name__ == "__main__":
    unittest.main()


import unittest

from programgrad import (
    Tensor,
    format_hard_soft_table,
    format_temperature_table,
    hard_soft_rows,
    soft_if,
    temperature_sensitivity,
    trace,
)


class EvaluationTests(unittest.TestCase):
    def test_hard_soft_rows_include_losses_and_ledger_metadata(self):
        x = Tensor(0.7, name="x")
        threshold = Tensor(1.1, requires_grad=True, name="threshold")

        with trace(mode="dual", relaxation="soft_gate", fidelity=True) as tr:
            y = soft_if(x - threshold, 2.0 * x, x**2, beta=4.0)
            loss = (y - 1.4) ** 2
            loss.backward()

        rows = hard_soft_rows(tr, target=1.4)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].kind, "branch")
        self.assertIsNotNone(rows[0].hard_loss)
        self.assertIsNotNone(rows[0].soft_loss)

        ledger = tr.gradient_ledger()[0]
        self.assertEqual(ledger["approximates"], "hard branch indicator 1[score > 0]")
        self.assertIn("score", ledger["gradient_flows_to"])
        self.assertIn("temperature sensitivity", ledger["recommended_checks"])

        table = format_hard_soft_table(rows)
        self.assertIn("hard_loss", table)
        self.assertIn("branch#", table)

    def test_temperature_sensitivity_uses_trace_fidelity(self):
        def run_once(beta: float):
            x = Tensor(0.7, name="x")
            threshold = Tensor(1.1, requires_grad=True, name="threshold")
            with trace(mode="dual", relaxation="soft_gate", fidelity=True, record_ops=False) as tr:
                y = soft_if(x - threshold, 2.0 * x, x**2, beta=beta)
            return y, tr

        rows = temperature_sensitivity(run_once, [1.0, 3.0, 6.0], target=1.4)
        self.assertEqual([row.temperature for row in rows], [1.0, 3.0, 6.0])
        self.assertTrue(all(row.output_gap is not None for row in rows))
        self.assertTrue(all(row.entropy is not None for row in rows))

        table = format_temperature_table(rows)
        self.assertIn("entropy", table)
        self.assertIn("loss", table)

    def test_nested_selection_propagates_hard_value(self):
        from programgrad import soft_select

        with trace(mode="dual", relaxation="softmax_select", fidelity=True) as tr:
            inner = soft_select([1.0, 3.0], [0.1, 1.0], tau=1.0)
            outer = soft_select([0.0, inner], [0.1, 1.0], tau=1.0)

        self.assertAlmostEqual(inner.hard_value, 3.0)
        self.assertAlmostEqual(outer.hard_value, 3.0)
        rows = hard_soft_rows(tr, target=3.0)
        self.assertAlmostEqual(rows[-1].hard_value, 3.0)


if __name__ == "__main__":
    unittest.main()

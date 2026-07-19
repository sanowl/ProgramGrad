import unittest

from programgrad import (
    EvaluationRow,
    Tensor,
    format_hard_soft_table,
    format_temperature_table,
    hard_soft_rows,
    soft_if,
    soft_select,
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
        with trace(mode="dual", relaxation="softmax_select", fidelity=True) as tr:
            inner = soft_select([1.0, 3.0], [0.1, 1.0], tau=1.0)
            outer = soft_select([0.0, inner], [0.1, 1.0], tau=1.0)

        self.assertAlmostEqual(inner.hard_value, 3.0)
        self.assertAlmostEqual(outer.hard_value, 3.0)
        rows = hard_soft_rows(tr, target=3.0)
        self.assertAlmostEqual(rows[-1].hard_value, 3.0)

    def test_mixed_events_are_reported_in_execution_order(self):
        def run_once(temperature: float):
            with trace(mode="dual", fidelity=True, record_ops=False) as tr:
                soft_select([1.0, 3.0], [0.0, 1.0], tau=temperature)
                result = soft_if(-0.5, 10.0, 2.0, beta=temperature)
            return result, tr

        result, tr = run_once(2.0)
        rows = hard_soft_rows(tr)
        self.assertEqual([row.kind for row in rows], ["search", "branch"])

        sensitivity = temperature_sensitivity(run_once, [2.0])
        self.assertAlmostEqual(sensitivity[0].hard_value, 2.0)
        self.assertAlmostEqual(sensitivity[0].output_gap, abs(2.0 - result.data))

    def test_temperature_sensitivity_validates_temperature(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            temperature_sensitivity(lambda _temperature: Tensor(1.0), [0.0])

    def test_hard_soft_rows_work_without_fidelity_metrics(self):
        with trace(mode="dual", fidelity=False, record_ops=False) as tr:
            soft_if(-0.5, 10.0, 2.0, beta=4.0)
            soft_select([1.0, 3.0], [0.0, 1.0], tau=1.0)

        rows = hard_soft_rows(tr, target=2.0)
        self.assertEqual([row.kind for row in rows], ["branch", "search"])
        self.assertAlmostEqual(rows[0].hard_value, 2.0)
        self.assertAlmostEqual(rows[1].hard_value, 3.0)
        self.assertIsNone(rows[0].output_gap)
        self.assertIsNone(rows[0].entropy)
        self.assertIsNone(rows[0].path_agreement)
        self.assertIsNone(rows[0].temperature)
        self.assertIsNotNone(rows[0].hard_loss)

    def test_hard_soft_rows_include_final_loop_state(self):
        from programgrad import bounded_loop

        with trace(mode="dual", fidelity=True, record_ops=False) as tr:
            bounded_loop(
                0.0,
                3,
                lambda _index, state: state + 1.0,
                lambda index, _state: 1.0 if index < 1 else -1.0,
                beta=20.0,
            )

        rows = hard_soft_rows(tr, target=1.0)
        self.assertEqual([row.kind for row in rows], ["loop"])
        self.assertAlmostEqual(rows[0].hard_value, 1.0)
        self.assertIsNotNone(rows[0].output_gap)
        self.assertIsNotNone(rows[0].entropy)

    def test_temperature_sensitivity_matches_returned_tensor(self):
        def run_once(temperature: float):
            with trace(mode="dual", fidelity=True, record_ops=False) as tr:
                first = soft_if(-0.5, 10.0, 2.0, beta=temperature)
                soft_if(1.0, 100.0, 0.0, beta=temperature)
            return first, tr

        rows = temperature_sensitivity(run_once, [2.0])
        self.assertAlmostEqual(rows[0].hard_value, 2.0)
        self.assertAlmostEqual(rows[0].soft_value, run_once(2.0)[0].data)
        self.assertAlmostEqual(rows[0].output_gap, abs(2.0 - rows[0].soft_value))
        self.assertNotEqual(rows[0].hard_value, 100.0)

    def test_markdown_tables_escape_cell_separators_and_newlines(self):
        row = EvaluationRow(
            name="branch|name\nnext",
            kind="branch",
            hard_value=1.0,
            soft_value=1.0,
            output_gap=0.0,
        )
        table = format_hard_soft_table([row])
        self.assertIn("branch\\|name<br>next", table)


if __name__ == "__main__":
    unittest.main()

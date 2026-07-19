import math
import unittest

from programgrad import (
    Tensor,
    bounded_loop,
    diff_if,
    hard_data,
    soft_argmax,
    soft_if,
    soft_select,
    trace,
)
from programgrad.relaxations import sigmoid_gate, softmax, straight_through_gate


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
        self.assertEqual(tr.gradient_ledger()[0]["estimator"], "surrogate")
        self.assertIn("hard objective after training", tr.gradient_ledger()[0]["recommended_checks"])

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
        self.assertEqual(tr.gradient_ledger()[0]["approximates"], "hard argmax over candidate scores")

    def test_soft_argmax_weights_sum_to_one(self):
        weights = soft_argmax([0.0, 1.0, 2.0], tau=1.0)
        self.assertAlmostEqual(sum(weight.data for weight in weights), 1.0)

    def test_hard_score_metadata_drives_branch_selection(self):
        score = Tensor(-0.5, requires_grad=True, name="score")
        score.hard_value = 1.0

        with trace(mode="dual", relaxation="straight_through", fidelity=True) as tr:
            y = diff_if(score, lambda: Tensor(4.0), lambda: Tensor(9.0), mode="straight_through")

        self.assertAlmostEqual(y.data, 4.0)
        self.assertAlmostEqual(y.hard_value, 4.0)
        self.assertEqual(tr.branches[0].selected_path, "true")
        self.assertTrue(tr.branches[0].condition.hard_value)

    def test_nested_soft_select_gap_uses_hard_value(self):
        with trace(mode="dual", relaxation="softmax_select", fidelity=True) as tr:
            inner = soft_select([1.0, 3.0], [0.1, 1.0], tau=1.0)
            outer = soft_select([0.0, inner], [0.1, 1.0], tau=1.0)

        self.assertAlmostEqual(outer.hard_value, 3.0)
        self.assertAlmostEqual(tr.searches[-1].fidelity.hard_value, 3.0)
        self.assertAlmostEqual(tr.searches[-1].fidelity.output_gap, abs(3.0 - outer.data))

    def test_temperature_validation_rejects_non_positive_values(self):
        with self.assertRaisesRegex(ValueError, "beta must be a finite positive number"):
            sigmoid_gate(0.0, beta=0.0)
        with self.assertRaisesRegex(ValueError, "tau must be a finite positive number"):
            softmax([1.0, 2.0], tau=math.nan)
        with self.assertRaisesRegex(ValueError, "beta must be a finite positive number"):
            straight_through_gate(1.0, beta=-3.0)
        with self.assertRaisesRegex(ValueError, "beta must be a finite positive number"):
            sigmoid_gate(0.0, beta=None)
        with self.assertRaisesRegex(ValueError, "not bool"):
            sigmoid_gate(0.0, beta=True)
        with self.assertRaisesRegex(ValueError, "not bool"):
            softmax([1.0, 2.0], tau=True)

    def test_nested_soft_branch_uses_original_hard_path(self):
        with trace(mode="dual", fidelity=True, record_ops=False) as tr:
            inner = soft_if(-0.1, 10.0, 1.0, beta=1.0)
            outer = soft_if(inner - 3.0, 20.0, 0.0, beta=1.0)

        self.assertGreater(inner.data, 3.0)
        self.assertAlmostEqual(inner.hard_value, 1.0)
        self.assertAlmostEqual(outer.hard_value, 0.0)
        self.assertEqual(tr.branches[-1].selected_path, "false")
        self.assertAlmostEqual(tr.branches[-1].condition.score, -2.0)
        self.assertFalse(tr.branches[-1].fidelity.path_agreement)

    def test_nested_soft_select_uses_hard_scores(self):
        with trace(mode="dual", fidelity=True, record_ops=False) as tr:
            nested_score = soft_if(-0.1, 10.0, 1.0, beta=1.0)
            selected = soft_select([10.0, 20.0], [nested_score, 3.0], tau=1.0)

        self.assertGreater(nested_score.data, 3.0)
        self.assertEqual(tr.searches[0].scores, [1.0, 3.0])
        self.assertGreater(tr.searches[0].soft_scores[0], 3.0)
        self.assertEqual(tr.searches[0].selected_index, 1)
        self.assertAlmostEqual(selected.hard_value, 20.0)
        self.assertFalse(tr.searches[0].fidelity.path_agreement)

    def test_soft_select_validates_candidate_names(self):
        with self.assertRaisesRegex(ValueError, "candidate_names"):
            soft_select([1.0, 2.0], [0.0, 1.0], candidate_names=["only-one"])
        with self.assertRaisesRegex(TypeError, "only strings"):
            soft_select([1.0], [0.0], candidate_names=[1])

    def test_fidelity_flag_disables_metrics_but_keeps_trace(self):
        with trace(mode="dual", fidelity=False, record_ops=False) as tr:
            soft_if(1.0, 2.0, 0.0)
            soft_select([1.0, 2.0], [0.0, 1.0])

        self.assertEqual(len(tr.branches), 1)
        self.assertEqual(len(tr.searches), 1)
        self.assertIsNone(tr.branches[0].fidelity)
        self.assertIsNone(tr.searches[0].fidelity)
        self.assertEqual(tr.fidelity_report(), {"branches": [], "searches": [], "loops": []})
        self.assertTrue(all(entry["fidelity_metrics"] is None for entry in tr.gradient_ledger()))
        self.assertIn("gap=none", tr.show())

    def test_bounded_loop_tracks_hard_stop_and_validates_configuration(self):
        initial = Tensor(0.0, requires_grad=True)
        with trace(mode="dual", fidelity=True, record_ops=False) as tr:
            result = bounded_loop(
                initial,
                4,
                lambda _index, state: state + 1.0,
                lambda index, _state: 1.0 if index < 2 else -1.0,
                beta=50.0,
            )
        result.backward()

        self.assertAlmostEqual(result.hard_value, 2.0)
        self.assertEqual(len(tr.loops), 4)
        self.assertTrue(tr.loops[0].hard_alive)
        self.assertAlmostEqual(tr.loops[0].hard_carried_state, 1.0)
        self.assertAlmostEqual(tr.loops[1].hard_carried_state, 2.0)
        self.assertFalse(tr.loops[2].hard_alive)
        self.assertAlmostEqual(tr.loops[2].hard_carried_state, 2.0)
        self.assertLess(tr.loops[2].hard_continue_score, 0.0)
        self.assertIsNone(tr.loops[3].hard_continue_score)
        self.assertAlmostEqual(tr.loops[3].hard_carried_state, 2.0)
        self.assertIn("hard_state=2", tr.show())
        hard_path = tr.hard_path()
        self.assertTrue(any(":stop" in event for event in hard_path))
        self.assertEqual(sum(":stop" in event for event in hard_path), 1)
        self.assertEqual(sum("soft-unroll" in event for event in hard_path), 0)
        self.assertGreater(initial.grad, 0.0)
        self.assertIn("loops", tr.fidelity_report())
        self.assertAlmostEqual(tr.fidelity_report()["loops"][0]["hard_value"], 2.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            bounded_loop(0.0, -1, lambda _i, state: state, lambda _i, _state: 1.0)
        with self.assertRaisesRegex(TypeError, "not bool"):
            bounded_loop(0.0, True, lambda _i, state: state, lambda _i, _state: 1.0)
        with self.assertRaisesRegex(ValueError, "beta"):
            bounded_loop(0.0, 0, lambda _i, state: state, lambda _i, _state: 1.0, beta=0.0)

    def test_bounded_loop_rejects_body_that_drops_hard_metadata(self):
        initial = Tensor(0.0)
        initial.hard_value = 0.0

        def escaping_body(_index: int, state: Tensor) -> float:
            return state.data + 1.0

        with self.assertRaisesRegex(ValueError, "dropped hard_value"):
            bounded_loop(
                initial,
                2,
                escaping_body,
                lambda _index, _state: 1.0,
                beta=10.0,
            )

    def test_hard_data_is_exported(self):
        value = Tensor(0.5)
        value.hard_value = 3.0
        self.assertAlmostEqual(hard_data(value), 3.0)

    def test_invalid_diff_if_mode_does_not_execute_callbacks(self):
        calls: list[str] = []
        with self.assertRaisesRegex(ValueError, "mode must be"):
            diff_if(
                1.0,
                lambda: calls.append("true") or 1.0,
                lambda: calls.append("false") or 0.0,
                mode="unknown",
            )
        self.assertEqual(calls, [])

    def test_softmax_stabilizes_hard_shadow_scores(self):
        high = Tensor(0.0)
        high.hard_value = 1_000.0
        low = Tensor(0.0)
        low.hard_value = 0.0

        weights = softmax([high, low])
        self.assertAlmostEqual(weights[0].data, 0.5)
        self.assertAlmostEqual(weights[0].hard_value, 1.0)
        self.assertAlmostEqual(weights[1].hard_value, 0.0)

    def test_softmax_rejects_non_finite_scores(self):
        with self.assertRaisesRegex(ValueError, "scores must be finite"):
            softmax([0.0, math.nan])
        hard_infinite = Tensor(0.0)
        hard_infinite.hard_value = math.inf
        with self.assertRaisesRegex(ValueError, "hard scores must be finite"):
            softmax([hard_infinite, 0.0])

    def test_unselected_invalid_hard_branch_does_not_poison_output(self):
        denominator = Tensor(2.0)
        denominator.hard_value = 0.0
        invalid_branch = Tensor(1.0) / denominator

        result = soft_if(1.0, 4.0, invalid_branch)
        self.assertAlmostEqual(result.hard_value, 4.0)
        self.assertIsNone(result.hard_error)

    def test_selected_invalid_hard_branch_fails_explicitly(self):
        denominator = Tensor(2.0)
        denominator.hard_value = 0.0
        invalid_branch = Tensor(1.0) / denominator

        with self.assertRaisesRegex(ValueError, "hard-program value is unavailable"):
            soft_if(-1.0, 4.0, invalid_branch)

    def test_branch_rejects_non_finite_scores(self):
        with self.assertRaisesRegex(ValueError, "score must be finite"):
            soft_if(math.nan, 1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "score must be finite"):
            diff_if(math.inf, lambda: 1.0, lambda: 0.0, mode="pathwise")

    def test_bounded_loop_ignores_hard_errors_after_hard_stop(self):
        def body(_index: int, state: Tensor) -> Tensor:
            return state + 1.0

        def continue_score(index: int, _state: Tensor) -> Tensor:
            if index == 0:
                return Tensor(-1.0)
            poisoned = Tensor(1.0)
            poisoned.hard_error = "boom"
            return poisoned

        with trace(mode="dual", fidelity=False, record_ops=False) as tr:
            result = bounded_loop(0.0, 3, body, continue_score, beta=10.0)

        self.assertAlmostEqual(result.hard_value, 0.0)
        self.assertFalse(tr.loops[0].hard_alive)
        self.assertTrue(tr.loops[0].on_hard_path)
        self.assertIsNone(tr.loops[1].hard_continue_score)
        self.assertFalse(tr.loops[1].on_hard_path)
        self.assertIsNone(tr.loops[2].hard_continue_score)
        loop_events = [event for event in tr.hard_path() if event.startswith("loop#")]
        self.assertEqual(len(loop_events), 1)
        self.assertIn(":stop(score=-1)", loop_events[0])

    def test_unselected_nested_branch_is_not_on_hard_path(self):
        with trace(mode="dual", fidelity=True, record_ops=False) as tr:
            soft_if(
                1.0,
                lambda: soft_if(1.0, 4.0, 5.0, beta=5.0),
                lambda: soft_if(-1.0, 8.0, 9.0, beta=5.0),
                beta=5.0,
            )

        self.assertEqual(len(tr.branches), 3)
        self.assertTrue(tr.branches[0].on_hard_path)
        nested = [branch for branch in tr.branches[1:] ]
        self.assertEqual(sum(branch.on_hard_path for branch in nested), 1)
        self.assertEqual(sum(not branch.on_hard_path for branch in nested), 1)
        hard_path = tr.hard_path()
        self.assertEqual(len(hard_path), 2)
        self.assertTrue(all("branch#" in event for event in hard_path))

    def test_bounded_loop_zero_steps_keeps_hard_value(self):
        result = bounded_loop(3.0, 0, lambda _i, state: state, lambda _i, _state: 1.0)
        self.assertAlmostEqual(result.data, 3.0)
        self.assertAlmostEqual(result.hard_value, 3.0)

    def test_bounded_loop_rejects_non_finite_continue_score(self):
        with self.assertRaisesRegex(ValueError, "continue_score must be finite"):
            bounded_loop(
                0.0,
                1,
                lambda _i, state: state,
                lambda _i, _state: math.nan,
            )


if __name__ == "__main__":
    unittest.main()

import unittest

from examples.learnable_threshold import X_VALUE, train as train_threshold
from examples.tiny_tree_search import TREE, score, train as train_tree


class ExampleTests(unittest.TestCase):
    def test_threshold_trace_matches_returned_parameter(self):
        threshold, final_trace = train_threshold(steps=2)
        self.assertAlmostEqual(
            final_trace.branches[-1].condition.score,
            X_VALUE - threshold.data,
        )

    def test_tree_trace_matches_returned_parameters(self):
        theta, final_trace = train_tree(steps=2)
        expected_root_scores = [score(child, theta).data for child in TREE.children]
        self.assertEqual(len(final_trace.searches), 3)
        for actual, expected in zip(final_trace.searches[-1].soft_scores, expected_root_scores):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

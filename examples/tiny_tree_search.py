"""Tiny differentiable tree search demo.

The hard program chooses the highest-scoring child. The soft shadow trace uses
softmax selection so heuristic weights can receive gradients.

Run with:
    python examples/tiny_tree_search.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from programgrad import (
    Tensor,
    format_hard_soft_table,
    format_temperature_table,
    hard_soft_rows,
    soft_select,
    temperature_sensitivity,
    trace,
)

TARGET = 3.2


@dataclass(frozen=True)
class Node:
    name: str
    feature_a: float = 0.0
    feature_b: float = 0.0
    cost: float = 0.0
    value: float | None = None
    children: list["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


TREE = Node(
    "root",
    children=[
        Node(
            "left",
            feature_a=0.7,
            feature_b=0.2,
            cost=0.1,
            children=[
                Node("left.left", feature_a=0.2, feature_b=0.6, cost=0.2, value=1.0),
                Node("left.right", feature_a=0.9, feature_b=0.3, cost=0.4, value=2.5),
            ],
        ),
        Node(
            "right",
            feature_a=0.3,
            feature_b=0.9,
            cost=0.2,
            children=[
                Node("right.left", feature_a=0.4, feature_b=0.7, cost=0.1, value=3.2),
                Node("right.right", feature_a=0.1, feature_b=0.2, cost=0.1, value=0.5),
            ],
        ),
    ],
)


def score(node: Node, theta: list[Tensor]) -> Tensor:
    return theta[0] * node.feature_a + theta[1] * node.feature_b - theta[2] * node.cost


def soft_tree_value(node: Node, theta: list[Tensor], *, tau: float = 0.65) -> Tensor:
    if node.is_leaf:
        assert node.value is not None
        return Tensor(node.value, name=node.name)

    values = [soft_tree_value(child, theta, tau=tau) for child in node.children]
    scores = [score(child, theta) for child in node.children]
    return soft_select(
        values,
        scores,
        tau=tau,
        candidate_names=[child.name for child in node.children],
    )


def hard_tree_value(node: Node, theta: list[Tensor]) -> float:
    if node.is_leaf:
        assert node.value is not None
        return node.value
    selected = max(node.children, key=lambda child: score(child, theta).data)
    return hard_tree_value(selected, theta)


def train(steps: int = 80, lr: float = 0.05) -> tuple[list[Tensor], object]:
    theta = [
        Tensor(0.1, requires_grad=True, name="theta_a"),
        Tensor(0.1, requires_grad=True, name="theta_b"),
        Tensor(0.1, requires_grad=True, name="theta_cost"),
    ]
    target = Tensor(TARGET, name="target")
    final_trace = None

    for step in range(steps):
        ctx = trace(mode="dual", relaxation="softmax_select", fidelity=True) if step == steps - 1 else None
        if ctx is None:
            y = soft_tree_value(TREE, theta)
            loss = (y - target) ** 2
            loss.backward()
        else:
            with ctx as tr:
                y = soft_tree_value(TREE, theta)
                loss = (y - target) ** 2
                loss.backward()
            final_trace = tr

        for param in theta:
            param.data -= lr * param.grad

    assert final_trace is not None
    return theta, final_trace


def run_tree_once(theta_values: list[float], tau: float) -> tuple[Tensor, object]:
    theta = [
        Tensor(theta_values[0], requires_grad=True, name="theta_a"),
        Tensor(theta_values[1], requires_grad=True, name="theta_b"),
        Tensor(theta_values[2], requires_grad=True, name="theta_cost"),
    ]
    with trace(mode="dual", relaxation="softmax_select", fidelity=True, record_ops=False) as tr:
        y = soft_tree_value(TREE, theta, tau=tau)
    return y, tr


if __name__ == "__main__":
    theta, tr = train()
    out = Path("tree_search_trace.svg")
    tr.export_svg(out)
    theta_values = [param.data for param in theta]
    sensitivity = temperature_sensitivity(
        lambda tau: run_tree_once(theta_values, tau),
        [0.25, 0.5, 0.65, 1.0],
        target=TARGET,
    )
    print("learned theta:", ", ".join(f"{param.name}={param.data:.4f}" for param in theta))
    print(f"hard tree value after training: {hard_tree_value(TREE, theta):.4f}")
    print(tr.show())
    print("\nhard-vs-soft evaluation:")
    print(format_hard_soft_table(hard_soft_rows(tr, target=TARGET)))
    print("\ntemperature sensitivity:")
    print(format_temperature_table(sensitivity))
    print(f"wrote {out}")

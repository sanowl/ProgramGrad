"""Learn a threshold inside a relaxed branch.

Run with:
    python examples/learnable_threshold.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from programgrad import (  # noqa: E402
    Tensor,
    TraceContext,
    format_hard_soft_table,
    format_temperature_table,
    hard_soft_rows,
    soft_if,
    temperature_sensitivity,
    trace,
    training_mode,
)

TARGET = 1.4
X_VALUE = 0.7


def train(steps: int = 60, lr: float = 0.08, beta: float = 6.0) -> tuple[Tensor, TraceContext]:
    x = Tensor(X_VALUE, name="x")
    threshold = Tensor(1.35, requires_grad=True, name="threshold")
    target = Tensor(TARGET, name="target")

    with training_mode(hard_shadow=False):
        for _ in range(steps):
            score = x - threshold
            y = soft_if(score, true_value=2.0 * x, false_value=x**2, beta=beta)
            loss = (y - target) ** 2
            loss.backward()
            threshold.data -= lr * threshold.grad

    # Re-evaluate after the final update so the returned trace and parameters
    # describe the same program state.
    with trace(mode="dual", relaxation="soft_gate", fidelity=True) as final_trace:
        score = x - threshold
        y = soft_if(score, true_value=2.0 * x, false_value=x**2, beta=beta)
        loss = (y - target) ** 2
        loss.backward()
    return threshold, final_trace


def run_threshold_once(threshold_value: float, beta: float) -> tuple[Tensor, TraceContext]:
    x = Tensor(X_VALUE, name="x")
    threshold = Tensor(threshold_value, requires_grad=True, name="threshold")
    with trace(mode="dual", relaxation="soft_gate", fidelity=True, record_ops=False) as tr:
        y = soft_if(x - threshold, true_value=2.0 * x, false_value=x**2, beta=beta)
    return y, tr


if __name__ == "__main__":
    learned_threshold, tr = train()
    out = Path("threshold_trace.svg")
    tr.export_svg(out)
    hard_output = 2.0 * X_VALUE if X_VALUE > learned_threshold.data else X_VALUE**2
    sensitivity = temperature_sensitivity(
        lambda beta: run_threshold_once(learned_threshold.data, beta),
        [1.0, 3.0, 6.0, 12.0],
        target=TARGET,
    )
    print(f"learned threshold: {learned_threshold.data:.4f}")
    print(f"hard program output after training: {hard_output:.4f}")
    print(tr.show())
    print("\nhard-vs-soft evaluation:")
    print(format_hard_soft_table(hard_soft_rows(tr, target=TARGET)))
    print("\ntemperature sensitivity:")
    print(format_temperature_table(sensitivity))
    print(f"wrote {out}")

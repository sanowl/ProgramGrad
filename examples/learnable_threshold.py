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

from programgrad import Tensor, soft_if, trace


def train(steps: int = 60, lr: float = 0.08, beta: float = 6.0) -> tuple[Tensor, object]:
    x = Tensor(0.7, name="x")
    threshold = Tensor(1.35, requires_grad=True, name="threshold")
    target = Tensor(1.4, name="target")
    final_trace = None

    for step in range(steps):
        ctx = trace(mode="dual", relaxation="soft_gate", fidelity=True) if step == steps - 1 else None
        if ctx is None:
            score = x - threshold
            y = soft_if(score, true_value=2.0 * x, false_value=x**2, beta=beta)
            loss = (y - target) ** 2
            loss.backward()
        else:
            with ctx as tr:
                score = x - threshold
                y = soft_if(score, true_value=2.0 * x, false_value=x**2, beta=beta)
                loss = (y - target) ** 2
                loss.backward()
            final_trace = tr

        threshold.data -= lr * threshold.grad

    assert final_trace is not None
    return threshold, final_trace


if __name__ == "__main__":
    learned_threshold, tr = train()
    out = Path("threshold_trace.svg")
    tr.export_svg(out)
    hard_output = 1.4 if 0.7 > learned_threshold.data else 0.49
    print(f"learned threshold: {learned_threshold.data:.4f}")
    print(f"hard program output after training: {hard_output:.4f}")
    print(tr.show())
    print(f"wrote {out}")

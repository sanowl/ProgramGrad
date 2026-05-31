"""Smooth relaxations used by control-flow primitives."""

from __future__ import annotations

from .ops import tensor_sum
from .tensor import Tensor, ensure_tensor


def sigmoid_gate(score: Tensor | float, beta: float = 10.0) -> Tensor:
    """Return ``sigmoid(beta * score)`` as a branch gate."""

    return (ensure_tensor(score) * float(beta)).sigmoid()


def straight_through_gate(score: Tensor | float, beta: float = 10.0) -> Tensor:
    """Hard forward gate with soft surrogate gradient."""

    score_t = ensure_tensor(score)
    hard = Tensor(1.0 if score_t.data > 0.0 else 0.0)
    soft = sigmoid_gate(score_t, beta=beta)
    return hard + (soft - soft.detach())


def softmax(scores: list[Tensor | float], tau: float = 1.0) -> list[Tensor]:
    """Stable scalar softmax over a short candidate list."""

    if tau <= 0.0:
        raise ValueError("tau must be positive")
    score_tensors = [ensure_tensor(score) for score in scores]
    if not score_tensors:
        raise ValueError("softmax requires at least one score")
    offset = max(score.data for score in score_tensors)
    exps = [((score - offset) / tau).exp() for score in score_tensors]
    denom = tensor_sum(exps)
    return [value / denom for value in exps]


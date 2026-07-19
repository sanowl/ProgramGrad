"""Smooth relaxations used by control-flow primitives."""

from __future__ import annotations

import math

from .ops import tensor_sum
from .tensor import Tensor, ensure_tensor, hard_data


def validate_temperature(value: float, *, name: str) -> float:
    """Normalize and validate a relaxation temperature."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number, not bool")
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return temperature


def sigmoid_gate(score: Tensor | float, beta: float = 10.0) -> Tensor:
    """Return ``sigmoid(beta * score)`` as a branch gate."""

    beta = validate_temperature(beta, name="beta")
    return (ensure_tensor(score) * beta).sigmoid()


def straight_through_gates(
    score: Tensor | float,
    beta: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Return ``(ste_gate, soft_gate)`` from one shared sigmoid evaluation."""

    score_t = ensure_tensor(score)
    soft = sigmoid_gate(score_t, beta=beta)
    hard = Tensor(1.0 if hard_data(score_t) > 0.0 else 0.0)
    ste = hard + (soft - soft.detach())
    return ste, soft


def straight_through_gate(score: Tensor | float, beta: float = 10.0) -> Tensor:
    """Hard forward gate with soft surrogate gradient."""

    ste, _soft = straight_through_gates(score, beta=beta)
    return ste


def softmax(scores: list[Tensor | float], tau: float = 1.0) -> list[Tensor]:
    """Stable scalar softmax over a short candidate list."""

    tau = validate_temperature(tau, name="tau")
    score_tensors = [ensure_tensor(score) for score in scores]
    if not score_tensors:
        raise ValueError("softmax requires at least one score")

    hard_scores: list[float] = []
    has_hard_shadow = False
    for score in score_tensors:
        if not math.isfinite(score.data):
            raise ValueError("softmax scores must be finite")
        hard_score = hard_data(score)
        if not math.isfinite(hard_score):
            raise ValueError("softmax hard scores must be finite")
        hard_scores.append(hard_score)
        if score.hard_value is not None:
            has_hard_shadow = True

    offset = Tensor(max(score.data for score in score_tensors))
    if has_hard_shadow:
        offset.hard_value = max(hard_scores)
    exps = [((score - offset) / tau).exp() for score in score_tensors]
    denom = tensor_sum(exps)
    return [value / denom for value in exps]

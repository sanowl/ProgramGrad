"""Smooth relaxations used by control-flow primitives."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

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


def _require_nonempty_scores(scores: Sequence[Tensor | float], *, name: str) -> list[Tensor]:
    score_tensors = [ensure_tensor(score) for score in scores]
    if not score_tensors:
        raise ValueError(f"{name} requires at least one score")
    return score_tensors


def _straight_through(hard: Tensor, soft: Tensor) -> Tensor:
    """Hard forward value with soft surrogate gradient."""

    return hard + (soft - soft.detach())


def sigmoid_gate(score: Tensor | float, beta: float = 10.0) -> Tensor:
    """Return ``sigmoid(beta * score)`` as a branch gate."""

    beta = validate_temperature(beta, name="beta")
    return (ensure_tensor(score) * beta).sigmoid()


def straight_through_gates(
    score: Tensor | float,
    beta: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Return ``(ste_gate, soft_gate)`` from one shared sigmoid evaluation."""

    from .trace import in_soft_only

    score_t = ensure_tensor(score)
    soft = sigmoid_gate(score_t, beta=beta)
    decision = score_t.data if in_soft_only() else hard_data(score_t)
    hard = Tensor(1.0 if decision > 0.0 else 0.0)
    return _straight_through(hard, soft), soft


def straight_through_gate(score: Tensor | float, beta: float = 10.0) -> Tensor:
    """Hard forward gate with soft surrogate gradient."""

    ste, _soft = straight_through_gates(score, beta=beta)
    return ste


def softmax(scores: list[Tensor | float], tau: float = 1.0) -> list[Tensor]:
    """Stable scalar softmax over a short candidate list."""

    from .trace import in_soft_only

    tau = validate_temperature(tau, name="tau")
    score_tensors = _require_nonempty_scores(scores, name="softmax")
    use_hard = not in_soft_only()

    hard_scores: list[float] = []
    has_hard_shadow = False
    for score in score_tensors:
        if not math.isfinite(score.data):
            raise ValueError("softmax scores must be finite")
        hard_score = hard_data(score) if use_hard else score.data
        if use_hard and not math.isfinite(hard_score):
            raise ValueError("softmax hard scores must be finite")
        hard_scores.append(hard_score)
        if use_hard and score.hard_value is not None:
            has_hard_shadow = True

    offset = Tensor(max(score.data for score in score_tensors))
    if has_hard_shadow:
        offset.hard_value = max(hard_scores)
    exps = [((score - offset) / tau).exp() for score in score_tensors]
    denom = tensor_sum(exps)
    return [value / denom for value in exps]


def _gumbel_noise(count: int, *, seed: int | None) -> list[float]:
    rng = random.Random(seed)
    noises: list[float] = []
    for _ in range(count):
        # Inverse-CDF Gumbel(0, 1): -log(-log(U)), with U in (0, 1).
        u = rng.random()
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        noises.append(-math.log(-math.log(u)))
    return noises


def gumbel_softmax(
    scores: Sequence[Tensor | float],
    *,
    tau: float = 1.0,
    seed: int | None = None,
) -> list[Tensor]:
    """Concrete / Gumbel-Softmax relaxation over candidate scores."""

    score_tensors = _require_nonempty_scores(scores, name="gumbel_softmax")
    noises = _gumbel_noise(len(score_tensors), seed=seed)
    perturbed = [score + noise for score, noise in zip(score_tensors, noises)]
    return softmax(perturbed, tau=tau)


def gumbel_softmax_straight_through(
    scores: Sequence[Tensor | float],
    *,
    tau: float = 1.0,
    seed: int | None = None,
) -> tuple[list[Tensor], list[Tensor], int]:
    """Hard Gumbel sample forward with soft Concrete backward.

    Returns ``(ste_weights, soft_weights, sampled_index)``.
    """

    soft_weights = gumbel_softmax(scores, tau=tau, seed=seed)
    sampled_index = max(range(len(soft_weights)), key=lambda idx: soft_weights[idx].data)
    ste_weights = [
        _straight_through(Tensor(1.0 if index == sampled_index else 0.0), soft)
        for index, soft in enumerate(soft_weights)
    ]
    return ste_weights, soft_weights, sampled_index

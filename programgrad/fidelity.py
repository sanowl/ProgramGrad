"""Hard-soft fidelity metrics for surrogate program traces."""

from __future__ import annotations

import math
from typing import Iterable


def scalar_gap(hard_value: float, soft_value: float | None) -> float | None:
    if soft_value is None:
        return None
    return abs(float(hard_value) - float(soft_value))


def binary_entropy(prob: float | None) -> float | None:
    if prob is None:
        return None
    p = min(max(float(prob), 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def categorical_entropy(weights: Iterable[float]) -> float:
    total = 0.0
    for weight in weights:
        p = min(max(float(weight), 1e-12), 1.0)
        total -= p * math.log(p)
    return total


def path_agrees(hard_index: int, weights: list[float]) -> bool:
    if not weights:
        return False
    return max(range(len(weights)), key=lambda i: weights[i]) == hard_index


"""Hard-soft fidelity metrics for surrogate program traces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .ir import LoopFrame


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


@dataclass(frozen=True)
class FinalLoopSummary:
    """Hard vs soft comparison for the final bounded-loop state."""

    event_id: int
    hard_value: float
    soft_value: float
    output_gap: float | None
    gate_entropy: float | None


def summarize_final_loop(
    loops: Sequence["LoopFrame"],
    *,
    include_metrics: bool,
) -> FinalLoopSummary | None:
    """Compare frozen hard state against the final soft carried state."""

    hard_loops = [frame for frame in loops if frame.on_hard_path]
    if not hard_loops or not loops:
        return None
    hard_frame = hard_loops[-1]
    soft_frame = loops[-1]
    if hard_frame.hard_carried_state is None:
        return None
    hard_value = hard_frame.hard_carried_state
    soft_value = soft_frame.carried_state
    return FinalLoopSummary(
        event_id=soft_frame.id,
        hard_value=hard_value,
        soft_value=soft_value,
        output_gap=scalar_gap(hard_value, soft_value) if include_metrics else None,
        gate_entropy=binary_entropy(soft_frame.continue_gate) if include_metrics else None,
    )

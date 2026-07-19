"""Internal convenience wrappers around scalar Tensor operations.

Not part of the public ``programgrad`` API surface. Prefer ``Tensor`` methods
in user code; ``tensor_sum`` is used by relaxations for a clean sum graph.
"""

from __future__ import annotations

from typing import Iterable

from .tensor import Tensor, ensure_tensor


def exp(x: Tensor | float) -> Tensor:
    return ensure_tensor(x).exp()


def log(x: Tensor | float) -> Tensor:
    return ensure_tensor(x).log()


def sigmoid(x: Tensor | float) -> Tensor:
    return ensure_tensor(x).sigmoid()


def tanh(x: Tensor | float) -> Tensor:
    return ensure_tensor(x).tanh()


def relu(x: Tensor | float) -> Tensor:
    return ensure_tensor(x).relu()


def tensor_sum(values: Iterable[Tensor | float]) -> Tensor:
    """Sum tensors without a useless ``0 + x`` leaf in the graph."""

    iterator = iter(values)
    try:
        total = ensure_tensor(next(iterator))
    except StopIteration:
        return Tensor(0.0)
    for value in iterator:
        total = total + ensure_tensor(value)
    return total


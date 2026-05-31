"""Convenience wrappers around scalar Tensor operations."""

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
    total = Tensor(0.0)
    for value in values:
        total = total + ensure_tensor(value)
    return total


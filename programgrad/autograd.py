"""Reverse-mode autodiff traversal for scalar tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .tensor import Tensor


def topo_sort(output: "Tensor") -> list["Tensor"]:
    """Return parents before children without recursion depth limits."""

    topo: list["Tensor"] = []
    visited: set[int] = set()
    stack: list[tuple["Tensor", bool]] = [(output, False)]
    while stack:
        node, expanded = stack.pop()
        if node.id in visited:
            continue
        if expanded:
            visited.add(node.id)
            topo.append(node)
            continue
        stack.append((node, True))
        stack.extend((parent, False) for parent in reversed(node._prev))
    return topo


def backward(output: "Tensor", grad: float = 1.0) -> None:
    """Backpropagate from ``output`` through its reachable graph."""

    topo = topo_sort(output)
    for node in topo:
        node.grad = 0.0

    output.grad = float(grad)
    for node in reversed(topo):
        node._backward()

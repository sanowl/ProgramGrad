"""Reverse-mode autodiff traversal for scalar tensors."""

from __future__ import annotations

from typing import List, Set


def topo_sort(output: "Tensor") -> List["Tensor"]:
    """Return parents before children for a scalar computation graph."""

    topo: List["Tensor"] = []
    visited: Set[int] = set()

    def visit(node: "Tensor") -> None:
        if node.id in visited:
            return
        visited.add(node.id)
        for parent in node._prev:
            visit(parent)
        topo.append(node)

    visit(output)
    return topo


def backward(output: "Tensor", grad: float = 1.0) -> None:
    """Backpropagate from ``output`` through its reachable graph."""

    topo = topo_sort(output)
    for node in topo:
        node.grad = 0.0

    output.grad = float(grad)
    for node in reversed(topo):
        node._backward()


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .tensor import Tensor


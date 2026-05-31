"""Finite-difference validation for scalar ProgramGrad functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .tensor import Tensor


@dataclass(frozen=True)
class GradcheckResult:
    passed: bool
    analytical: list[float]
    numerical: list[float]
    max_abs_error: float
    max_rel_error: float


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[float],
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> GradcheckResult:
    """Compare reverse-mode gradients with central finite differences."""

    values = [float(value) for value in inputs]
    tensors = [
        Tensor(value, requires_grad=True, name=f"x{idx}")
        for idx, value in enumerate(values)
    ]
    output = fn(*tensors)
    if not isinstance(output, Tensor):
        raise TypeError("gradcheck function must return a Tensor")
    output.backward()
    analytical = [tensor.grad for tensor in tensors]

    numerical: list[float] = []
    for idx in range(len(values)):
        plus = list(values)
        minus = list(values)
        plus[idx] += eps
        minus[idx] -= eps
        y_plus = fn(*[Tensor(value, requires_grad=True) for value in plus]).data
        y_minus = fn(*[Tensor(value, requires_grad=True) for value in minus]).data
        numerical.append((y_plus - y_minus) / (2.0 * eps))

    abs_errors = [abs(a - n) for a, n in zip(analytical, numerical)]
    rel_errors = [
        abs(a - n) / max(1.0, abs(a), abs(n))
        for a, n in zip(analytical, numerical)
    ]
    max_abs = max(abs_errors) if abs_errors else 0.0
    max_rel = max(rel_errors) if rel_errors else 0.0
    return GradcheckResult(
        passed=max_abs <= atol or max_rel <= rtol,
        analytical=analytical,
        numerical=numerical,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
    )


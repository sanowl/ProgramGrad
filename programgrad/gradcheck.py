"""Finite-difference validation for scalar ProgramGrad functions."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

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

    eps = float(eps)
    atol = float(atol)
    rtol = float(rtol)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a finite positive number")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be a finite non-negative number")
    if not math.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be a finite non-negative number")

    values = [float(value) for value in inputs]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("inputs must contain only finite values")
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
        plus_output = fn(*[Tensor(value, requires_grad=True) for value in plus])
        minus_output = fn(*[Tensor(value, requires_grad=True) for value in minus])
        if not isinstance(plus_output, Tensor) or not isinstance(minus_output, Tensor):
            raise TypeError("gradcheck function must return a Tensor")
        y_plus = plus_output.data
        y_minus = minus_output.data
        numerical.append((y_plus - y_minus) / (2.0 * eps))

    abs_errors = [abs(a - n) for a, n in zip(analytical, numerical)]
    rel_errors = [
        abs(a - n) / max(1e-12, abs(a), abs(n))
        for a, n in zip(analytical, numerical)
    ]
    max_abs = max(abs_errors) if abs_errors else 0.0
    max_rel = max(rel_errors) if rel_errors else 0.0
    passed = all(
        math.isfinite(error)
        and error <= atol + rtol * max(abs(analytical_value), abs(numerical_value))
        for error, analytical_value, numerical_value in zip(
            abs_errors,
            analytical,
            numerical,
        )
    )
    return GradcheckResult(
        passed=passed,
        analytical=analytical,
        numerical=numerical,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
    )

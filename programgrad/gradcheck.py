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


def _require_positive(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _require_nonnegative(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _as_tensor_output(value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("gradcheck function must return a Tensor")
    return value


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[float],
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> GradcheckResult:
    """Compare reverse-mode gradients with central finite differences."""

    eps = _require_positive(eps, name="eps")
    atol = _require_nonnegative(atol, name="atol")
    rtol = _require_nonnegative(rtol, name="rtol")

    values = [float(value) for value in inputs]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("inputs must contain only finite values")
    tensors = [
        Tensor(value, requires_grad=True, name=f"x{idx}")
        for idx, value in enumerate(values)
    ]
    output = _as_tensor_output(fn(*tensors))
    output.backward()
    analytical = [tensor.grad for tensor in tensors]

    numerical: list[float] = []
    for idx in range(len(values)):
        plus = list(values)
        minus = list(values)
        plus[idx] += eps
        minus[idx] -= eps
        plus_output = _as_tensor_output(
            fn(*[Tensor(value, requires_grad=True) for value in plus])
        )
        minus_output = _as_tensor_output(
            fn(*[Tensor(value, requires_grad=True) for value in minus])
        )
        numerical.append((plus_output.data - minus_output.data) / (2.0 * eps))

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

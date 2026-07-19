"""A tiny scalar Tensor with reverse-mode autodiff."""

from __future__ import annotations

import math
from itertools import count
from typing import Any, Callable, Iterable

_TENSOR_IDS = count()


def ensure_tensor(value: Any) -> "Tensor":
    if isinstance(value, Tensor):
        return value
    return Tensor(float(value))


def hard_data(value: "Tensor") -> float:
    """Return the hard-program value when one is attached to a tensor."""

    from .trace import hard_shadow_enabled

    if not hard_shadow_enabled():
        return value.data
    if value.hard_error is not None:
        raise ValueError(f"hard-program value is unavailable: {value.hard_error}")
    return value.hard_value if value.hard_value is not None else value.data


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _propagate_hard(
    output: "Tensor",
    inputs: Iterable["Tensor"],
    fn: Callable[..., float],
) -> None:
    """Propagate hard-program metadata through a deterministic operation.

    Soft forward already succeeded on ``.data``. If the hard shadow leaves the
    real domain (for example ``log`` of a non-positive hard value), leave
    ``hard_value`` unset instead of aborting the surrogate.
    """

    from .trace import hard_shadow_enabled

    if not hard_shadow_enabled():
        return

    input_values = tuple(inputs)
    has_hard = False
    for value in input_values:
        if value.hard_error is not None:
            output.hard_error = value.hard_error
            return
        if value.hard_value is not None:
            has_hard = True
    if not has_hard:
        return
    try:
        result = fn(*(hard_data(value) for value in input_values))
    except (ValueError, ArithmeticError, TypeError) as exc:
        output.hard_error = f"{type(exc).__name__}: {exc}"
        return
    if isinstance(result, complex):
        output.hard_error = "operation left the real number domain"
        return
    try:
        output.hard_value = float(result)
    except (TypeError, ValueError, OverflowError) as exc:
        output.hard_error = f"{type(exc).__name__}: {exc}"


def _record_op(
    op_name: str,
    inputs: Iterable["Tensor"],
    output: "Tensor",
    local_derivative: dict[str, float | str],
) -> None:
    from .trace import current_trace, ops_recording_enabled

    if not ops_recording_enabled():
        return
    tr = current_trace()
    if tr is not None:
        tr.record_op(op_name, list(inputs), output, local_derivative)


def _binary_op(
    lhs: "Tensor",
    rhs: "Tensor",
    *,
    op_name: str,
    soft_value: float,
    hard_fn: Callable[[float, float], float],
    local_derivative: dict[str, float | str],
    accumulate: Callable[["Tensor"], None],
) -> "Tensor":
    out = Tensor(
        soft_value,
        requires_grad=lhs.requires_grad or rhs.requires_grad,
        _children=(lhs, rhs),
        _op=op_name,
    )
    _propagate_hard(out, (lhs, rhs), hard_fn)
    out._backward = lambda: accumulate(out)
    _record_op(op_name, (lhs, rhs), out, local_derivative)
    return out


def _unary_op(
    value: "Tensor",
    *,
    op_name: str,
    soft_value: float,
    hard_fn: Callable[[float], float],
    local_derivative: dict[str, float | str],
    accumulate: Callable[["Tensor"], None],
) -> "Tensor":
    out = Tensor(
        soft_value,
        requires_grad=value.requires_grad,
        _children=(value,),
        _op=op_name,
    )
    _propagate_hard(out, (value,), hard_fn)
    out._backward = lambda: accumulate(out)
    _record_op(op_name, (value,), out, local_derivative)
    return out


class Tensor:
    """Scalar reverse-mode autodiff value.

    ``Tensor`` is deliberately scalar-only for the first release. That keeps
    the gradient semantics readable while the trace and control-flow layers
    are still being stabilized.
    """

    def __init__(
        self,
        data: float,
        *,
        requires_grad: bool = False,
        name: str | None = None,
        _children: tuple["Tensor", ...] = (),
        _op: str = "leaf",
    ) -> None:
        self.data = float(data)
        self.requires_grad = bool(requires_grad)
        self.grad = 0.0
        self.name = name
        self.id = next(_TENSOR_IDS)
        self._prev = tuple(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None
        self.hard_value: float | None = None
        self.hard_error: str | None = None

    def __repr__(self) -> str:
        label = f", name={self.name!r}" if self.name else ""
        hard = f", hard_value={self.hard_value:.6g}" if self.hard_value is not None else ""
        hard_error = f", hard_error={self.hard_error!r}" if self.hard_error is not None else ""
        return (
            f"Tensor(data={self.data:.6g}, grad={self.grad:.6g}, "
            f"requires_grad={self.requires_grad}{label}{hard}{hard_error})"
        )

    def item(self) -> float:
        return self.data

    def detach(self) -> "Tensor":
        out = Tensor(self.data, name=self.name)
        out.hard_value = self.hard_value
        out.hard_error = self.hard_error
        return out

    def zero_grad(self) -> None:
        self.grad = 0.0

    def backward(self, grad: float = 1.0) -> None:
        from .autograd import backward

        backward(self, grad)

    def __add__(self, other: Any) -> "Tensor":
        other = ensure_tensor(other)

        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        return _binary_op(
            self,
            other,
            op_name="add",
            soft_value=self.data + other.data,
            hard_fn=lambda lhs, rhs: lhs + rhs,
            local_derivative={"dout/dlhs": 1.0, "dout/drhs": 1.0},
            accumulate=accumulate,
        )

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __sub__(self, other: Any) -> "Tensor":
        other = ensure_tensor(other)

        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

        return _binary_op(
            self,
            other,
            op_name="sub",
            soft_value=self.data - other.data,
            hard_fn=lambda lhs, rhs: lhs - rhs,
            local_derivative={"dout/dlhs": 1.0, "dout/drhs": -1.0},
            accumulate=accumulate,
        )

    def __rsub__(self, other: Any) -> "Tensor":
        return ensure_tensor(other) - self

    def __neg__(self) -> "Tensor":
        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                self.grad -= out.grad

        return _unary_op(
            self,
            op_name="neg",
            soft_value=-self.data,
            hard_fn=lambda value: -value,
            local_derivative={"dout/dx": -1.0},
            accumulate=accumulate,
        )

    def __mul__(self, other: Any) -> "Tensor":
        other = ensure_tensor(other)

        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        return _binary_op(
            self,
            other,
            op_name="mul",
            soft_value=self.data * other.data,
            hard_fn=lambda lhs, rhs: lhs * rhs,
            local_derivative={"dout/dlhs": other.data, "dout/drhs": self.data},
            accumulate=accumulate,
        )

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __truediv__(self, other: Any) -> "Tensor":
        other = ensure_tensor(other)
        if other.data == 0.0:
            raise ZeroDivisionError("division by zero in Tensor")

        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                self.grad += (1.0 / other.data) * out.grad
            if other.requires_grad:
                other.grad -= (self.data / (other.data * other.data)) * out.grad

        return _binary_op(
            self,
            other,
            op_name="div",
            soft_value=self.data / other.data,
            hard_fn=lambda lhs, rhs: lhs / rhs,
            local_derivative={
                "dout/dlhs": 1.0 / other.data,
                "dout/drhs": -self.data / (other.data**2),
            },
            accumulate=accumulate,
        )

    def __rtruediv__(self, other: Any) -> "Tensor":
        return ensure_tensor(other) / self

    def __pow__(self, power: Any) -> "Tensor":
        power_t = ensure_tensor(power)
        if self.data <= 0.0 and power_t.requires_grad:
            raise ValueError("gradient with respect to exponent requires positive base")

        try:
            value = self.data**power_t.data
        except ZeroDivisionError as exc:
            raise ZeroDivisionError("zero cannot be raised to a negative power") from exc
        if isinstance(value, complex):
            raise ValueError("Tensor power must stay in the real number domain")

        if self.data == 0.0 and power_t.data == 0.0:
            base_derivative: float | str = 0.0
        elif self.data == 0.0 and power_t.data < 1.0:
            base_derivative = "undefined"
        else:
            base_derivative = power_t.data * (self.data ** (power_t.data - 1.0))
        exponent_derivative: float | str = (
            value * math.log(self.data) if self.data > 0.0 else "undefined"
        )

        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                if self.data == 0.0 and power_t.data < 1.0 and power_t.data != 0.0:
                    raise ValueError("undefined derivative for zero base and power < 1")
                derivative = 0.0 if base_derivative == "undefined" else float(base_derivative)
                self.grad += derivative * out.grad
            if power_t.requires_grad:
                power_t.grad += float(exponent_derivative) * out.grad

        return _binary_op(
            self,
            power_t,
            op_name="pow",
            soft_value=value,
            hard_fn=lambda base, exponent: base**exponent,
            local_derivative={
                "dout/dbase": base_derivative,
                "dout/dexponent": exponent_derivative,
            },
            accumulate=accumulate,
        )

    def __rpow__(self, other: Any) -> "Tensor":
        return ensure_tensor(other) ** self

    def exp(self) -> "Tensor":
        value = math.exp(self.data)

        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                self.grad += value * out.grad

        return _unary_op(
            self,
            op_name="exp",
            soft_value=value,
            hard_fn=math.exp,
            local_derivative={"dout/dx": value},
            accumulate=accumulate,
        )

    def log(self) -> "Tensor":
        if self.data <= 0.0:
            raise ValueError("log is only defined for positive Tensor values")

        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        return _unary_op(
            self,
            op_name="log",
            soft_value=math.log(self.data),
            hard_fn=math.log,
            local_derivative={"dout/dx": 1.0 / self.data},
            accumulate=accumulate,
        )

    def tanh(self) -> "Tensor":
        value = math.tanh(self.data)

        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                self.grad += (1.0 - value * value) * out.grad

        return _unary_op(
            self,
            op_name="tanh",
            soft_value=value,
            hard_fn=math.tanh,
            local_derivative={"dout/dx": 1.0 - value * value},
            accumulate=accumulate,
        )

    def sigmoid(self) -> "Tensor":
        value = _stable_sigmoid(self.data)

        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                self.grad += value * (1.0 - value) * out.grad

        return _unary_op(
            self,
            op_name="sigmoid",
            soft_value=value,
            hard_fn=_stable_sigmoid,
            local_derivative={"dout/dx": value * (1.0 - value)},
            accumulate=accumulate,
        )

    def relu(self) -> "Tensor":
        value = self.data if self.data > 0.0 else 0.0
        local = 1.0 if self.data > 0.0 else 0.0

        def accumulate(out: Tensor) -> None:
            if self.requires_grad:
                self.grad += local * out.grad

        return _unary_op(
            self,
            op_name="relu",
            soft_value=value,
            hard_fn=lambda hard_value: max(0.0, hard_value),
            local_derivative={"dout/dx": local},
            accumulate=accumulate,
        )

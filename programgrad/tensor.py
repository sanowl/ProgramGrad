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
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )
        _propagate_hard(out, (self, other), lambda lhs, rhs: lhs + rhs)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        _record_op("add", (self, other), out, {"dout/dlhs": 1.0, "dout/drhs": 1.0})
        return out

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __sub__(self, other: Any) -> "Tensor":
        other = ensure_tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="sub",
        )
        _propagate_hard(out, (self, other), lambda lhs, rhs: lhs - rhs)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

        out._backward = _backward
        _record_op("sub", (self, other), out, {"dout/dlhs": 1.0, "dout/drhs": -1.0})
        return out

    def __rsub__(self, other: Any) -> "Tensor":
        return ensure_tensor(other) - self

    def __neg__(self) -> "Tensor":
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="neg",
        )
        _propagate_hard(out, (self,), lambda value: -value)

        def _backward() -> None:
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        _record_op("neg", (self,), out, {"dout/dx": -1.0})
        return out

    def __mul__(self, other: Any) -> "Tensor":
        other = ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )
        _propagate_hard(out, (self, other), lambda lhs, rhs: lhs * rhs)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        _record_op(
            "mul",
            (self, other),
            out,
            {"dout/dlhs": other.data, "dout/drhs": self.data},
        )
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __truediv__(self, other: Any) -> "Tensor":
        other = ensure_tensor(other)
        if other.data == 0.0:
            raise ZeroDivisionError("division by zero in Tensor")
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="div",
        )
        _propagate_hard(out, (self, other), lambda lhs, rhs: lhs / rhs)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 / other.data) * out.grad
            if other.requires_grad:
                other.grad -= (self.data / (other.data * other.data)) * out.grad

        out._backward = _backward
        _record_op(
            "div",
            (self, other),
            out,
            {"dout/dlhs": 1.0 / other.data, "dout/drhs": -self.data / (other.data**2)},
        )
        return out

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

        out = Tensor(
            value,
            requires_grad=self.requires_grad or power_t.requires_grad,
            _children=(self, power_t),
            _op="pow",
        )
        _propagate_hard(out, (self, power_t), lambda base, exponent: base**exponent)

        def _backward() -> None:
            if self.requires_grad:
                if self.data == 0.0 and power_t.data < 1.0 and power_t.data != 0.0:
                    raise ValueError("undefined derivative for zero base and power < 1")
                derivative = (
                    0.0
                    if self.data == 0.0 and power_t.data == 0.0
                    else power_t.data * (self.data ** (power_t.data - 1.0))
                )
                self.grad += derivative * out.grad
            if power_t.requires_grad:
                power_t.grad += out.data * math.log(self.data) * out.grad

        out._backward = _backward
        _record_op(
            "pow",
            (self, power_t),
            out,
            {
                "dout/dbase": (
                    0.0
                    if self.data == 0.0 and power_t.data == 0.0
                    else power_t.data * (self.data ** (power_t.data - 1.0))
                    if not (self.data == 0.0 and power_t.data < 1.0)
                    else "undefined"
                ),
                "dout/dexponent": out.data * math.log(self.data) if self.data > 0.0 else "undefined",
            },
        )
        return out

    def __rpow__(self, other: Any) -> "Tensor":
        return ensure_tensor(other) ** self

    def exp(self) -> "Tensor":
        value = math.exp(self.data)
        out = Tensor(value, requires_grad=self.requires_grad, _children=(self,), _op="exp")
        _propagate_hard(out, (self,), math.exp)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += value * out.grad

        out._backward = _backward
        _record_op("exp", (self,), out, {"dout/dx": value})
        return out

    def log(self) -> "Tensor":
        if self.data <= 0.0:
            raise ValueError("log is only defined for positive Tensor values")
        out = Tensor(
            math.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )
        _propagate_hard(out, (self,), math.log)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        _record_op("log", (self,), out, {"dout/dx": 1.0 / self.data})
        return out

    def tanh(self) -> "Tensor":
        value = math.tanh(self.data)
        out = Tensor(value, requires_grad=self.requires_grad, _children=(self,), _op="tanh")
        _propagate_hard(out, (self,), math.tanh)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 - value * value) * out.grad

        out._backward = _backward
        _record_op("tanh", (self,), out, {"dout/dx": 1.0 - value * value})
        return out

    def sigmoid(self) -> "Tensor":
        value = _stable_sigmoid(self.data)
        out = Tensor(value, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")
        _propagate_hard(out, (self,), _stable_sigmoid)

        def _backward() -> None:
            if self.requires_grad:
                self.grad += value * (1.0 - value) * out.grad

        out._backward = _backward
        _record_op("sigmoid", (self,), out, {"dout/dx": value * (1.0 - value)})
        return out

    def relu(self) -> "Tensor":
        value = self.data if self.data > 0.0 else 0.0
        out = Tensor(value, requires_grad=self.requires_grad, _children=(self,), _op="relu")
        _propagate_hard(out, (self,), lambda hard_value: max(0.0, hard_value))

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        _record_op("relu", (self,), out, {"dout/dx": 1.0 if self.data > 0.0 else 0.0})
        return out

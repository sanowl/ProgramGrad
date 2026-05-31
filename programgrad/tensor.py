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


def _record_op(
    op_name: str,
    inputs: Iterable["Tensor"],
    output: "Tensor",
    local_derivative: dict[str, float | str],
) -> None:
    try:
        from .trace import current_trace
    except Exception:  # pragma: no cover - defensive import guard
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

    def __repr__(self) -> str:
        label = f", name={self.name!r}" if self.name else ""
        return (
            f"Tensor(data={self.data:.6g}, grad={self.grad:.6g}, "
            f"requires_grad={self.requires_grad}{label})"
        )

    def item(self) -> float:
        return self.data

    def detach(self) -> "Tensor":
        return Tensor(self.data, name=self.name)

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

        out = Tensor(
            self.data**power_t.data,
            requires_grad=self.requires_grad or power_t.requires_grad,
            _children=(self, power_t),
            _op="pow",
        )

        def _backward() -> None:
            if self.requires_grad:
                if self.data == 0.0 and power_t.data < 1.0:
                    raise ValueError("undefined derivative for zero base and power < 1")
                self.grad += power_t.data * (self.data ** (power_t.data - 1.0)) * out.grad
            if power_t.requires_grad:
                power_t.grad += out.data * math.log(self.data) * out.grad

        out._backward = _backward
        _record_op(
            "pow",
            (self, power_t),
            out,
            {
                "dout/dbase": power_t.data * (self.data ** (power_t.data - 1.0))
                if not (self.data == 0.0 and power_t.data < 1.0)
                else "undefined",
                "dout/dexponent": out.data * math.log(self.data) if self.data > 0.0 else "undefined",
            },
        )
        return out

    def __rpow__(self, other: Any) -> "Tensor":
        return ensure_tensor(other) ** self

    def exp(self) -> "Tensor":
        value = math.exp(self.data)
        out = Tensor(value, requires_grad=self.requires_grad, _children=(self,), _op="exp")

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

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        _record_op("log", (self,), out, {"dout/dx": 1.0 / self.data})
        return out

    def tanh(self) -> "Tensor":
        value = math.tanh(self.data)
        out = Tensor(value, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 - value * value) * out.grad

        out._backward = _backward
        _record_op("tanh", (self,), out, {"dout/dx": 1.0 - value * value})
        return out

    def sigmoid(self) -> "Tensor":
        if self.data >= 0.0:
            z = math.exp(-self.data)
            value = 1.0 / (1.0 + z)
        else:
            z = math.exp(self.data)
            value = z / (1.0 + z)
        out = Tensor(value, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += value * (1.0 - value) * out.grad

        out._backward = _backward
        _record_op("sigmoid", (self,), out, {"dout/dx": value * (1.0 - value)})
        return out

    def relu(self) -> "Tensor":
        value = self.data if self.data > 0.0 else 0.0
        out = Tensor(value, requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        _record_op("relu", (self,), out, {"dout/dx": 1.0 if self.data > 0.0 else 0.0})
        return out


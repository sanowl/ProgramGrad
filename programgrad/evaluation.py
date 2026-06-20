"""Research evaluation helpers for ProgramGrad traces."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .tensor import Tensor


@dataclass(frozen=True)
class EvaluationRow:
    """One hard-vs-soft comparison row."""

    name: str
    kind: str
    hard_value: float
    soft_value: float | None
    output_gap: float | None
    target: float | None = None
    hard_loss: float | None = None
    soft_loss: float | None = None
    path_agreement: bool | None = None
    entropy: float | None = None
    temperature: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemperatureSensitivityRow:
    """One temperature point for a branch or selection relaxation."""

    temperature: float
    soft_value: float
    hard_value: float | None
    output_gap: float | None
    entropy: float | None
    path_agreement: bool | None
    loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def squared_loss(value: float, target: float) -> float:
    return (float(value) - float(target)) ** 2


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(headers: list[str], rows: Iterable[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def hard_soft_rows(trace: "TraceContext", *, target: float | None = None) -> list[EvaluationRow]:
    """Build rows comparing the hard program value with the soft surrogate."""

    rows: list[EvaluationRow] = []
    for branch in trace.branches:
        metrics = branch.fidelity
        if metrics is None:
            continue
        hard_loss = squared_loss(metrics.hard_value, target) if target is not None else None
        soft_loss = (
            squared_loss(metrics.soft_value, target)
            if target is not None and metrics.soft_value is not None
            else None
        )
        rows.append(
            EvaluationRow(
                name=f"branch#{branch.id}:{branch.selected_path}",
                kind="branch",
                hard_value=metrics.hard_value,
                soft_value=metrics.soft_value,
                output_gap=metrics.output_gap,
                target=target,
                hard_loss=hard_loss,
                soft_loss=soft_loss,
                path_agreement=metrics.path_agreement,
                entropy=metrics.gate_entropy,
                temperature=metrics.temperature,
            )
        )
    for search in trace.searches:
        metrics = search.fidelity
        hard_loss = squared_loss(metrics.hard_value, target) if target is not None else None
        soft_loss = (
            squared_loss(metrics.soft_value, target)
            if target is not None and metrics.soft_value is not None
            else None
        )
        rows.append(
            EvaluationRow(
                name=f"search#{search.id}:candidate[{search.selected_index}]",
                kind="search",
                hard_value=metrics.hard_value,
                soft_value=metrics.soft_value,
                output_gap=metrics.output_gap,
                target=target,
                hard_loss=hard_loss,
                soft_loss=soft_loss,
                path_agreement=metrics.path_agreement,
                entropy=metrics.gate_entropy,
                temperature=metrics.temperature,
            )
        )
    return rows


def format_hard_soft_table(rows: Iterable[EvaluationRow]) -> str:
    """Render a compact markdown table for README/demo output."""

    return _markdown_table(
        [
            "event",
            "kind",
            "temp",
            "hard",
            "soft",
            "gap",
            "agree",
            "entropy",
            "hard_loss",
            "soft_loss",
        ],
        [
            [
                row.name,
                row.kind,
                row.temperature,
                row.hard_value,
                row.soft_value,
                row.output_gap,
                row.path_agreement,
                row.entropy,
                row.hard_loss,
                row.soft_loss,
            ]
            for row in rows
        ],
    )


def _extract_primary_result(result: Any) -> tuple[Tensor, "TraceContext | None"]:
    if isinstance(result, Tensor):
        return result, None
    if isinstance(result, tuple):
        tensor = next((item for item in result if isinstance(item, Tensor)), None)
        trace_obj = next((item for item in result if _looks_like_trace(item)), None)
        if tensor is not None:
            return tensor, trace_obj
    raise TypeError("temperature run function must return a Tensor or a tuple containing a Tensor")


def _looks_like_trace(value: object) -> bool:
    return hasattr(value, "branches") and hasattr(value, "searches") and hasattr(value, "fidelity_report")


def _primary_fidelity(trace_obj: "TraceContext | None") -> tuple[float | None, float | None, bool | None]:
    if trace_obj is None:
        return None, None, None
    events = []
    events.extend(branch.fidelity for branch in trace_obj.branches if branch.fidelity is not None)
    events.extend(search.fidelity for search in trace_obj.searches)
    if not events:
        return None, None, None
    latest = events[-1]
    return latest.output_gap, latest.gate_entropy, latest.path_agreement


def temperature_sensitivity(
    run_fn: Callable[[float], Tensor | tuple[Any, ...]],
    temperatures: Iterable[float],
    *,
    target: float | None = None,
) -> list[TemperatureSensitivityRow]:
    """Evaluate a relaxation across beta/tau values.

    ``run_fn`` receives a temperature and returns either a ``Tensor`` or a tuple
    containing a ``Tensor`` and optionally a ``TraceContext``. The helper records
    the soft value and, when available, the hard value/fidelity from the trace.
    """

    rows: list[TemperatureSensitivityRow] = []
    for temperature in temperatures:
        result = run_fn(float(temperature))
        tensor, trace_obj = _extract_primary_result(result)
        gap, entropy, agreement = _primary_fidelity(trace_obj)
        hard_value = tensor.hard_value
        if trace_obj is not None:
            events = []
            events.extend(branch.fidelity for branch in trace_obj.branches if branch.fidelity is not None)
            events.extend(search.fidelity for search in trace_obj.searches)
            if events:
                hard_value = events[-1].hard_value
        loss = squared_loss(tensor.data, target) if target is not None else None
        rows.append(
            TemperatureSensitivityRow(
                temperature=float(temperature),
                soft_value=tensor.data,
                hard_value=hard_value,
                output_gap=gap,
                entropy=entropy,
                path_agreement=agreement,
                loss=loss,
            )
        )
    return rows


def format_temperature_table(rows: Iterable[TemperatureSensitivityRow]) -> str:
    return _markdown_table(
        ["temp", "hard", "soft", "gap", "agree", "entropy", "loss"],
        [
            [
                row.temperature,
                row.hard_value,
                row.soft_value,
                row.output_gap,
                row.path_agreement,
                row.entropy,
                row.loss,
            ]
            for row in rows
        ],
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .trace import TraceContext


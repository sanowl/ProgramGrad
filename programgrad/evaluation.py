"""Research evaluation helpers for ProgramGrad traces."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from .fidelity import scalar_gap, summarize_final_loop
from .relaxations import validate_temperature
from .tensor import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from .ir import FidelityMetrics
    from .trace import TraceContext


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
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(headers: list[str], rows: Iterable[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _event_row(
    *,
    event_id: int,
    name: str,
    kind: str,
    hard_value: float,
    soft_value: float | None,
    metrics: "FidelityMetrics | None",
    target: float | None,
) -> tuple[int, EvaluationRow]:
    hard_loss = squared_loss(hard_value, target) if target is not None else None
    soft_loss = (
        squared_loss(soft_value, target)
        if target is not None and soft_value is not None
        else None
    )
    return (
        event_id,
        EvaluationRow(
            name=name,
            kind=kind,
            hard_value=hard_value,
            soft_value=soft_value,
            output_gap=None if metrics is None else metrics.output_gap,
            target=target,
            hard_loss=hard_loss,
            soft_loss=soft_loss,
            path_agreement=None if metrics is None else metrics.path_agreement,
            entropy=None if metrics is None else metrics.gate_entropy,
            temperature=None if metrics is None else metrics.temperature,
        ),
    )


def hard_soft_rows(trace: "TraceContext", *, target: float | None = None) -> list[EvaluationRow]:
    """Build rows comparing the hard program value with the soft surrogate.

    When a trace was recorded with ``fidelity=False``, rows still come from the
    structural branch/search nodes; gap/entropy/agreement fields stay ``None``.
    """

    indexed_rows: list[tuple[int, EvaluationRow]] = []
    for branch in trace.branches:
        if not branch.on_hard_path:
            continue
        metrics = branch.fidelity
        indexed_rows.append(
            _event_row(
                event_id=branch.id,
                name=f"branch#{branch.id}:{branch.selected_path}",
                kind="branch",
                hard_value=metrics.hard_value if metrics is not None else branch.output_hard,
                soft_value=metrics.soft_value if metrics is not None else branch.output_soft,
                metrics=metrics,
                target=target,
            )
        )
    for search in trace.searches:
        if not search.on_hard_path:
            continue
        metrics = search.fidelity
        indexed_rows.append(
            _event_row(
                event_id=search.id,
                name=f"search#{search.id}:candidate[{search.selected_index}]",
                kind="search",
                hard_value=metrics.hard_value if metrics is not None else search.output_hard,
                soft_value=metrics.soft_value if metrics is not None else search.output_soft,
                metrics=metrics,
                target=target,
            )
        )
    loop_summary = summarize_final_loop(trace.loops, include_metrics=trace.fidelity)
    if loop_summary is not None:
        hard_loss = (
            squared_loss(loop_summary.hard_value, target) if target is not None else None
        )
        soft_loss = (
            squared_loss(loop_summary.soft_value, target) if target is not None else None
        )
        indexed_rows.append(
            (
                loop_summary.event_id,
                EvaluationRow(
                    name=f"loop#{loop_summary.event_id}:final",
                    kind="loop",
                    hard_value=loop_summary.hard_value,
                    soft_value=loop_summary.soft_value,
                    output_gap=loop_summary.output_gap,
                    target=target,
                    hard_loss=hard_loss,
                    soft_loss=soft_loss,
                    path_agreement=None,
                    entropy=loop_summary.gate_entropy,
                    temperature=None,
                ),
            )
        )
    indexed_rows.sort(key=lambda item: item[0])
    return [row for _, row in indexed_rows]


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


def _fidelity_events(trace_obj: "TraceContext") -> list["FidelityMetrics"]:
    events = [
        (branch.id, branch.fidelity)
        for branch in trace_obj.branches
        if branch.fidelity is not None
    ]
    events.extend(
        (search.id, search.fidelity)
        for search in trace_obj.searches
        if search.fidelity is not None
    )
    events.sort(key=lambda item: item[0])
    return [metrics for _, metrics in events]


def _match_fidelity(
    trace_obj: "TraceContext | None",
    *,
    soft_value: float,
    hard_value: float | None,
) -> "FidelityMetrics | None":
    if trace_obj is None:
        return None
    matches = [
        metrics
        for metrics in _fidelity_events(trace_obj)
        if metrics.soft_value is not None
        and math.isclose(metrics.soft_value, soft_value, rel_tol=0.0, abs_tol=1e-9)
    ]
    if hard_value is not None:
        hard_matches = [
            metrics
            for metrics in matches
            if math.isclose(metrics.hard_value, hard_value, rel_tol=0.0, abs_tol=1e-9)
        ]
        if hard_matches:
            matches = hard_matches
    return matches[-1] if matches else None


def temperature_sensitivity(
    run_fn: Callable[[float], Tensor | tuple[Any, ...]],
    temperatures: Iterable[float],
    *,
    target: float | None = None,
) -> list[TemperatureSensitivityRow]:
    """Evaluate a relaxation across beta/tau values.

    ``run_fn`` receives a temperature and returns either a ``Tensor`` or a tuple
    containing a ``Tensor`` and optionally a ``TraceContext``. Soft/hard values
    come from the returned tensor; fidelity entropy/agreement are taken only from
    a matching trace event when one exists.
    """

    rows: list[TemperatureSensitivityRow] = []
    for temperature in temperatures:
        temperature = validate_temperature(temperature, name="temperature")
        result = run_fn(temperature)
        tensor, trace_obj = _extract_primary_result(result)
        hard_value = tensor.hard_value
        matched = _match_fidelity(
            trace_obj,
            soft_value=tensor.data,
            hard_value=hard_value,
        )
        if hard_value is None and matched is not None:
            hard_value = matched.hard_value
        gap = scalar_gap(hard_value, tensor.data) if hard_value is not None else None
        if matched is not None and matched.output_gap is not None:
            gap = matched.output_gap
        loss = squared_loss(tensor.data, target) if target is not None else None
        rows.append(
            TemperatureSensitivityRow(
                temperature=temperature,
                soft_value=tensor.data,
                hard_value=hard_value,
                output_gap=gap,
                entropy=None if matched is None else matched.gate_entropy,
                path_agreement=None if matched is None else matched.path_agreement,
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

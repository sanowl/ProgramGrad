"""Trace context for hard-soft program execution."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar, Token
from itertools import count
from pathlib import Path
from typing import Any, Iterator

from .ir import BranchNode, LedgerEntry, LoopFrame, OpNode, SearchNode

_TRACE_STACK: ContextVar[tuple["TraceContext", ...]] = ContextVar(
    "programgrad_trace_stack",
    default=(),
)
_SOFT_ONLY_DEPTH: ContextVar[int] = ContextVar("programgrad_soft_only_depth", default=0)
# Hot-path flags: checked on every tensor op without walking the trace stack.
_RECORD_OPS_ACTIVE: ContextVar[bool] = ContextVar("programgrad_record_ops", default=False)
_HARD_SHADOW_ACTIVE: ContextVar[bool] = ContextVar("programgrad_hard_shadow", default=True)


def current_trace() -> "TraceContext | None":
    stack = _TRACE_STACK.get()
    if not stack:
        return None
    return stack[-1]


def ops_recording_enabled() -> bool:
    """True only inside a trace that requested per-op recording."""

    return _RECORD_OPS_ACTIVE.get()


def hard_shadow_enabled() -> bool:
    """True while hard-program shadow values are being maintained."""

    return _HARD_SHADOW_ACTIVE.get()


def in_soft_only() -> bool:
    """True while evaluating a branch/candidate off the original hard path."""

    return _SOFT_ONLY_DEPTH.get() > 0


@contextmanager
def soft_only_region() -> Iterator[None]:
    """Mark nested control-flow events as soft-only (not on the hard path)."""

    token = _SOFT_ONLY_DEPTH.set(_SOFT_ONLY_DEPTH.get() + 1)
    try:
        yield
    finally:
        _SOFT_ONLY_DEPTH.reset(token)


def _require_nonempty_str(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


@contextmanager
def training_mode(*, hard_shadow: bool = False) -> Iterator[None]:
    """Cheap context for optimization loops.

    Disables op recording and, by default, hard-shadow propagation. Use a full
    ``trace(...)`` or ``training_trace()`` when you need decision fidelity.
    """

    hard_shadow = _require_bool(hard_shadow, name="hard_shadow")
    ops_token = _RECORD_OPS_ACTIVE.set(False)
    hard_token = _HARD_SHADOW_ACTIVE.set(hard_shadow)
    try:
        yield
    finally:
        _HARD_SHADOW_ACTIVE.reset(hard_token)
        _RECORD_OPS_ACTIVE.reset(ops_token)


def trace(
    *,
    mode: str = "dual",
    relaxation: str = "soft_gate",
    fidelity: bool = True,
    record_ops: bool = False,
    hard_shadow: bool = True,
) -> "TraceContext":
    return TraceContext(
        mode=mode,
        relaxation=relaxation,
        fidelity=fidelity,
        record_ops=record_ops,
        hard_shadow=hard_shadow,
    )


def training_trace(
    *,
    mode: str = "dual",
    relaxation: str = "soft_gate",
    fidelity: bool = False,
    record_ops: bool = False,
    hard_shadow: bool = True,
) -> "TraceContext":
    """Trace preset for training: no op log, fidelity off unless requested."""

    return TraceContext(
        mode=mode,
        relaxation=relaxation,
        fidelity=fidelity,
        record_ops=record_ops,
        hard_shadow=hard_shadow,
    )


class TraceContext:
    """Collect operation, branch, search, ledger, and fidelity events."""

    def __init__(
        self,
        *,
        mode: str = "dual",
        relaxation: str = "soft_gate",
        fidelity: bool = True,
        record_ops: bool = False,
        hard_shadow: bool = True,
    ) -> None:
        self.mode = _require_nonempty_str(mode, name="mode")
        self.relaxation = _require_nonempty_str(relaxation, name="relaxation")
        self.fidelity = _require_bool(fidelity, name="fidelity")
        self.record_ops_enabled = _require_bool(record_ops, name="record_ops")
        self.hard_shadow = _require_bool(hard_shadow, name="hard_shadow")
        if self.fidelity and not self.hard_shadow:
            raise ValueError(
                "fidelity=True requires hard_shadow=True; "
                "hard-vs-soft metrics are meaningless without hard shadows"
            )
        self.ops: list[OpNode] = []
        self.branches: list[BranchNode] = []
        self.searches: list[SearchNode] = []
        self.loops: list[LoopFrame] = []
        self.ledger: list[LedgerEntry] = []
        self._ids = count()
        self._ops_token: Token[bool] | None = None
        self._hard_token: Token[bool] | None = None

    def __enter__(self) -> "TraceContext":
        self._ops_token = _RECORD_OPS_ACTIVE.set(self.record_ops_enabled)
        self._hard_token = _HARD_SHADOW_ACTIVE.set(self.hard_shadow)
        _TRACE_STACK.set((*_TRACE_STACK.get(), self))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        stack = _TRACE_STACK.get()
        if not stack or stack[-1] is not self:  # pragma: no cover - misuse guard
            raise RuntimeError("trace context stack corruption")
        _TRACE_STACK.set(stack[:-1])
        if self._hard_token is not None:
            _HARD_SHADOW_ACTIVE.reset(self._hard_token)
            self._hard_token = None
        if self._ops_token is not None:
            _RECORD_OPS_ACTIVE.reset(self._ops_token)
            self._ops_token = None

    def next_id(self) -> int:
        return next(self._ids)

    def _tensor_label(self, value: Any) -> str:
        name = getattr(value, "name", None)
        data = getattr(value, "data", None)
        ident = getattr(value, "id", None)
        if name:
            return f"{name}={data:.6g}"
        if ident is not None and data is not None:
            return f"t{ident}={data:.6g}"
        return str(value)

    def record_op(
        self,
        op_name: str,
        inputs: list[Any],
        output: Any,
        local_derivative: dict[str, float | str],
    ) -> None:
        if not self.record_ops_enabled:
            return
        self.ops.append(
            OpNode(
                id=self.next_id(),
                op_name=op_name,
                inputs=[self._tensor_label(value) for value in inputs],
                output=self._tensor_label(output),
                local_derivative=local_derivative,
            )
        )

    def record_branch(self, node: BranchNode, ledger_entry: LedgerEntry) -> None:
        self.branches.append(node)
        self.ledger.append(ledger_entry)

    def record_search(self, node: SearchNode, ledger_entry: LedgerEntry) -> None:
        self.searches.append(node)
        self.ledger.append(ledger_entry)

    def record_loop(self, frame: LoopFrame) -> None:
        self.loops.append(frame)

    def hard_path(self) -> list[str]:
        from .fidelity import loop_decision_label

        events: list[tuple[int, str]] = []
        for branch in self.branches:
            if not branch.on_hard_path:
                continue
            events.append(
                (
                    branch.id,
                    f"branch#{branch.id}:{branch.selected_path}"
                    f"(score={branch.condition.score:.6g})",
                )
            )
        for search in self.searches:
            if not search.on_hard_path:
                continue
            events.append(
                (
                    search.id,
                    f"search#{search.id}:candidate[{search.selected_index}]"
                    f"(score={search.scores[search.selected_index]:.6g})",
                )
            )
        for loop in self.loops:
            # Soft-only unroll frames after hard exit are omitted from the hard path.
            if not loop.on_hard_path or loop.hard_continue_score is None:
                continue
            decision = loop_decision_label(loop)
            events.append(
                (
                    loop.id,
                    f"loop#{loop.id}:iteration[{loop.iteration_index}]:{decision}"
                    f"(score={loop.hard_continue_score:.6g})",
                )
            )
        events.sort(key=lambda event: event[0])
        return [description for _, description in events]

    def soft_gates(self) -> list[float]:
        gates = [
            branch.condition.gate
            for branch in self.branches
            if branch.condition.gate is not None
        ]
        return [float(gate) for gate in gates]

    def gradient_ledger(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.ledger]

    def fidelity_report(self) -> dict[str, list[dict[str, Any]]]:
        branch_rows = [
            branch.fidelity.to_dict()
            for branch in self.branches
            if branch.fidelity is not None
        ]
        search_rows = [
            search.fidelity.to_dict()
            for search in self.searches
            if search.fidelity is not None
        ]
        from .fidelity import summarize_final_loop

        loop_rows: list[dict[str, Any]] = []
        summary = summarize_final_loop(self.loops, include_metrics=self.fidelity)
        if summary is not None and self.fidelity:
            loop_rows.append(
                {
                    "hard_value": summary.hard_value,
                    "soft_value": summary.soft_value,
                    "output_gap": summary.output_gap,
                    "path_agreement": None,
                    "gate_entropy": summary.gate_entropy,
                    "temperature": None,
                }
            )
        return {"branches": branch_rows, "searches": search_rows, "loops": loop_rows}

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "relaxation": self.relaxation,
            "fidelity": self.fidelity,
            "record_ops": self.record_ops_enabled,
            "hard_shadow": self.hard_shadow,
            "ops": [node.to_dict() for node in self.ops],
            "branches": [node.to_dict() for node in self.branches],
            "searches": [node.to_dict() for node in self.searches],
            "loops": [node.to_dict() for node in self.loops],
            "ledger": [entry.to_dict() for entry in self.ledger],
        }

    def export_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    def export_svg(self, path: str | Path) -> Path:
        from .visualize import export_svg

        return export_svg(self, path)

    def show(self) -> str:
        from .fidelity import loop_decision_label

        lines = ["ProgramGrad trace"]
        lines.append(
            f"mode={self.mode}, relaxation={self.relaxation}, "
            f"fidelity={self.fidelity}, record_ops={self.record_ops_enabled}, "
            f"hard_shadow={self.hard_shadow}"
        )
        if self.branches:
            lines.append("branches:")
            for branch in self.branches:
                gate = branch.condition.gate
                gate_text = "none" if gate is None else f"{gate:.6g}"
                gap = None if branch.fidelity is None else branch.fidelity.output_gap
                gap_text = "none" if gap is None else f"{gap:.6g}"
                soft_text = "none" if branch.output_soft is None else f"{branch.output_soft:.6g}"
                lines.append(
                    f"  #{branch.id} hard={branch.selected_path} gate={gate_text} "
                    f"hard={branch.output_hard:.6g} soft={soft_text} gap={gap_text}"
                )
        if self.searches:
            lines.append("searches:")
            for search in self.searches:
                weights = ", ".join(f"{w:.3g}" for w in search.soft_weights)
                gap = None if search.fidelity is None else search.fidelity.output_gap
                gap_text = "none" if gap is None else f"{gap:.6g}"
                lines.append(
                    f"  #{search.id} selected={search.selected_index} "
                    f"weights=[{weights}] gap={gap_text}"
                )
        if self.loops:
            lines.append("loops:")
            for loop in self.loops:
                gate = "none" if loop.continue_gate is None else f"{loop.continue_gate:.6g}"
                hard_state = (
                    "none"
                    if loop.hard_carried_state is None
                    else f"{loop.hard_carried_state:.6g}"
                )
                decision = loop_decision_label(loop)
                lines.append(
                    f"  #{loop.id} iteration={loop.iteration_index} hard={decision} "
                    f"hard_state={hard_state} soft_state={loop.carried_state:.6g} gate={gate}"
                )
        if self.ledger:
            lines.append("gradient ledger:")
            for entry in self.ledger:
                lines.append(f"  #{entry.id} {entry.surrogate_type}: {entry.gradient_contract}")
                if entry.approximates:
                    lines.append(f"     approximates: {entry.approximates}")
                if entry.gradient_flows_to:
                    lines.append(f"     gradients: {', '.join(entry.gradient_flows_to)}")
                lines.append(f"     warning: {entry.bias_warning}")
                if entry.recommended_checks:
                    lines.append(f"     checks: {', '.join(entry.recommended_checks)}")
        return "\n".join(lines)

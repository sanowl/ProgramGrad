"""Trace context for hard-soft program execution."""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path
from typing import Any

from .ir import BranchNode, LedgerEntry, LoopFrame, OpNode, SearchNode

_TRACE_STACK: list["TraceContext"] = []


def current_trace() -> "TraceContext | None":
    if not _TRACE_STACK:
        return None
    return _TRACE_STACK[-1]


def trace(
    *,
    mode: str = "dual",
    relaxation: str = "soft_gate",
    fidelity: bool = True,
    record_ops: bool = True,
) -> "TraceContext":
    return TraceContext(
        mode=mode,
        relaxation=relaxation,
        fidelity=fidelity,
        record_ops=record_ops,
    )


class TraceContext:
    """Collect operation, branch, search, ledger, and fidelity events."""

    def __init__(
        self,
        *,
        mode: str = "dual",
        relaxation: str = "soft_gate",
        fidelity: bool = True,
        record_ops: bool = True,
    ) -> None:
        self.mode = mode
        self.relaxation = relaxation
        self.fidelity = fidelity
        self.record_ops_enabled = record_ops
        self.ops: list[OpNode] = []
        self.branches: list[BranchNode] = []
        self.searches: list[SearchNode] = []
        self.loops: list[LoopFrame] = []
        self.ledger: list[LedgerEntry] = []
        self._ids = count()

    def __enter__(self) -> "TraceContext":
        _TRACE_STACK.append(self)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        popped = _TRACE_STACK.pop()
        if popped is not self:  # pragma: no cover - stack corruption guard
            raise RuntimeError("trace context stack corruption")

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
        path: list[str] = []
        for branch in self.branches:
            path.append(
                f"branch#{branch.id}:{branch.selected_path}"
                f"(score={branch.condition.score:.6g})"
            )
        for search in self.searches:
            path.append(
                f"search#{search.id}:candidate[{search.selected_index}]"
                f"(score={search.scores[search.selected_index]:.6g})"
            )
        return path

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
        search_rows = [search.fidelity.to_dict() for search in self.searches]
        return {"branches": branch_rows, "searches": search_rows}

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "relaxation": self.relaxation,
            "ops": [node.to_dict() for node in self.ops],
            "branches": [node.to_dict() for node in self.branches],
            "searches": [node.to_dict() for node in self.searches],
            "loops": [node.to_dict() for node in self.loops],
            "ledger": [entry.to_dict() for entry in self.ledger],
        }

    def export_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    def export_svg(self, path: str | Path) -> Path:
        from .visualize import export_svg

        return export_svg(self, path)

    def show(self) -> str:
        lines = ["ProgramGrad trace"]
        lines.append(f"mode={self.mode}, relaxation={self.relaxation}")
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
                lines.append(
                    f"  #{search.id} selected={search.selected_index} "
                    f"weights=[{weights}] gap={search.fidelity.output_gap:.6g}"
                )
        if self.ledger:
            lines.append("gradient ledger:")
            for entry in self.ledger:
                lines.append(f"  #{entry.id} {entry.surrogate_type}: {entry.gradient_contract}")
                lines.append(f"     warning: {entry.bias_warning}")
        return "\n".join(lines)

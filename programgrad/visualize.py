"""Minimal SVG trace export without external graph dependencies."""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .trace import TraceContext


def _row(y: int, title: str, lines: list[str], color: str) -> str:
    escaped_title = html.escape(title)
    text = [
        f'<text x="34" y="{y + 26}" font-family="Arial" font-size="15" '
        f'font-weight="700" fill="#172033">{escaped_title}</text>'
    ]
    for idx, line in enumerate(lines[:3]):
        text.append(
            f'<text x="34" y="{y + 48 + idx * 18}" font-family="Arial" '
            f'font-size="13" fill="#344054">{html.escape(line)}</text>'
        )
    return (
        f'<rect x="20" y="{y}" width="860" height="92" rx="6" '
        f'fill="{color}" stroke="#c8d1dc"/>' + "".join(text)
    )


def _fmt_optional(value: float | None, *, digits: str = ".4g") -> str:
    if value is None:
        return "none"
    return format(value, digits)


def export_svg(trace: "TraceContext", path: str | Path) -> Path:
    rows: list[tuple[str, list[str], str]] = []
    for branch in trace.branches:
        gate = branch.condition.gate
        gate_text = _fmt_optional(gate)
        gap = None if branch.fidelity is None else branch.fidelity.output_gap
        soft_score = _fmt_optional(branch.condition.soft_score)
        rows.append(
            (
                f"Branch #{branch.id}: hard path = {branch.selected_path}",
                [
                    f"hard score {branch.condition.comparator}: {branch.condition.score:.4g}; "
                    f"soft score={soft_score}; gate={gate_text}",
                    f"hard={branch.output_hard:.4g}; soft={_fmt_optional(branch.output_soft)}; "
                    f"gap={_fmt_optional(gap)}",
                    branch.condition.gradient_contract,
                ],
                "#f2f7ff",
            )
        )
    for search in trace.searches:
        weights = ", ".join(f"{w:.3f}" for w in search.soft_weights)
        soft_scores = search.soft_scores if search.soft_scores is not None else search.scores
        gap = None if search.fidelity is None else search.fidelity.output_gap
        rows.append(
            (
                f"Search #{search.id}: selected candidate {search.selected_index}",
                [
                    f"hard scores={', '.join(f'{s:.3g}' for s in search.scores)}",
                    f"soft scores={', '.join(f'{s:.3g}' for s in soft_scores)}; weights=[{weights}]",
                    f"hard={search.output_hard:.4g}; soft={search.output_soft:.4g}; "
                    f"gap={_fmt_optional(gap)}",
                ],
                "#f6fff4",
            )
        )
    for loop in trace.loops:
        if loop.hard_continue_score is None:
            decision = "soft-unroll"
        else:
            decision = "continue" if loop.hard_alive else "stop"
        rows.append(
            (
                f"Loop #{loop.id}: iteration {loop.iteration_index}",
                [
                    f"hard decision={decision}; "
                    f"hard continue score={_fmt_optional(loop.hard_continue_score)}",
                    f"hard state={_fmt_optional(loop.hard_carried_state)}; "
                    f"soft state={loop.carried_state:.4g}",
                    f"soft continue score={_fmt_optional(loop.continue_score)}; "
                    f"continue gate={_fmt_optional(loop.continue_gate)}",
                ],
                "#f8f5ff",
            )
        )
    for op in trace.ops[:20]:
        rows.append(
            (
                f"Op #{op.id}: {op.op_name}",
                [
                    f"inputs: {', '.join(op.inputs)}",
                    f"output: {op.output}",
                    f"local derivative: {op.local_derivative}",
                ],
                "#fffaf0",
            )
        )

    if not rows:
        rows.append(("Empty trace", ["No trace events were recorded."], "#f8fafc"))

    height = 70 + len(rows) * 108
    content = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" '
        f'height="{height}" viewBox="0 0 900 {height}">',
        '<rect width="900" height="100%" fill="#ffffff"/>',
        '<text x="20" y="38" font-family="Arial" font-size="24" '
        'font-weight="700" fill="#101828">ProgramGrad Trace</text>',
        f'<text x="20" y="60" font-family="Arial" font-size="13" fill="#667085">'
        f'mode={html.escape(trace.mode)}, relaxation={html.escape(trace.relaxation)}, '
        f'fidelity={str(trace.fidelity).lower()}, '
        f'record_ops={str(trace.record_ops_enabled).lower()}, '
        f'hard_shadow={str(trace.hard_shadow).lower()}</text>',
    ]
    for idx, (title, lines, color) in enumerate(rows):
        content.append(_row(82 + idx * 108, title, lines, color))
    content.append("</svg>")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(content), encoding="utf-8")
    return output

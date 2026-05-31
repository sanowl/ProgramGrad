"""Minimal SVG trace export without external graph dependencies."""

from __future__ import annotations

import html
from pathlib import Path


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


def export_svg(trace: "TraceContext", path: str | Path) -> Path:
    rows: list[tuple[str, list[str], str]] = []
    for branch in trace.branches:
        gate = branch.condition.gate
        gate_text = "none" if gate is None else f"{gate:.4f}"
        gap = None if branch.fidelity is None else branch.fidelity.output_gap
        gap_text = "none" if gap is None else f"{gap:.4g}"
        rows.append(
            (
                f"Branch #{branch.id}: hard path = {branch.selected_path}",
                [
                    f"score {branch.condition.comparator}: {branch.condition.score:.4g}; gate={gate_text}",
                    f"hard={branch.output_hard:.4g}; soft={branch.output_soft}; gap={gap_text}",
                    branch.condition.gradient_contract,
                ],
                "#f2f7ff",
            )
        )
    for search in trace.searches:
        weights = ", ".join(f"{w:.3f}" for w in search.soft_weights)
        rows.append(
            (
                f"Search #{search.id}: selected candidate {search.selected_index}",
                [
                    f"scores={', '.join(f'{s:.3g}' for s in search.scores)}",
                    f"soft weights=[{weights}]",
                    f"hard={search.output_hard:.4g}; soft={search.output_soft:.4g}; gap={search.fidelity.output_gap:.4g}",
                ],
                "#f6fff4",
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
        f'mode={html.escape(trace.mode)}, relaxation={html.escape(trace.relaxation)}</text>',
    ]
    for idx, (title, lines, color) in enumerate(rows):
        content.append(_row(82 + idx * 108, title, lines, color))
    content.append("</svg>")

    output = Path(path)
    output.write_text("\n".join(content), encoding="utf-8")
    return output


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .trace import TraceContext


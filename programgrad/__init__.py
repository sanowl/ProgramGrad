"""ProgramGrad public API.

ProgramGrad is intentionally small in v0.1: scalar reverse-mode autodiff,
explicit control-flow primitives, trace metadata, and fidelity reports.
"""

from .control_flow import bounded_loop, diff_if, soft_argmax, soft_if, soft_select
from .evaluation import (
    EvaluationRow,
    TemperatureSensitivityRow,
    format_hard_soft_table,
    format_temperature_table,
    hard_soft_rows,
    temperature_sensitivity,
)
from .gradcheck import GradcheckResult, gradcheck
from .tensor import Tensor, hard_data
from .trace import TraceContext, soft_only_region, trace, training_mode, training_trace

__all__ = [
    "GradcheckResult",
    "Tensor",
    "TraceContext",
    "bounded_loop",
    "diff_if",
    "EvaluationRow",
    "format_hard_soft_table",
    "format_temperature_table",
    "gradcheck",
    "hard_data",
    "hard_soft_rows",
    "soft_argmax",
    "soft_if",
    "soft_only_region",
    "soft_select",
    "temperature_sensitivity",
    "TemperatureSensitivityRow",
    "trace",
    "training_mode",
    "training_trace",
]

"""ProgramGrad public API.

ProgramGrad is intentionally small in v0.1: scalar reverse-mode autodiff,
explicit control-flow primitives, trace metadata, and fidelity reports.
"""

from .control_flow import bounded_loop, diff_if, soft_argmax, soft_if, soft_select
from .gradcheck import GradcheckResult, gradcheck
from .tensor import Tensor
from .trace import TraceContext, trace

__all__ = [
    "GradcheckResult",
    "Tensor",
    "TraceContext",
    "bounded_loop",
    "diff_if",
    "gradcheck",
    "soft_argmax",
    "soft_if",
    "soft_select",
    "trace",
]


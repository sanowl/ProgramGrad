"""ProgramGrad trace IR.

The IR keeps the hard program behavior separate from the differentiable
surrogate. That separation is the main research object of the project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class RelaxationSpec:
    surrogate_type: str
    temperature: float | None = None
    estimator: str = "surrogate"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FidelityMetrics:
    hard_value: float
    soft_value: float | None
    output_gap: float | None
    path_agreement: bool | None
    gate_entropy: float | None
    temperature: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpNode:
    id: int
    op_name: str
    inputs: list[str]
    output: str
    local_derivative: dict[str, float | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConditionNode:
    score: float
    comparator: str
    hard_value: bool
    gate: float | None
    relaxation: RelaxationSpec
    temperature: float | None
    gradient_contract: str
    soft_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["relaxation"] = self.relaxation.to_dict()
        return data


@dataclass(frozen=True)
class BranchNode:
    id: int
    condition: ConditionNode
    selected_path: Literal["true", "false"]
    output_hard: float
    output_soft: float | None
    fidelity: FidelityMetrics | None = None
    on_hard_path: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["condition"] = self.condition.to_dict()
        if self.fidelity is not None:
            data["fidelity"] = self.fidelity.to_dict()
        return data


@dataclass(frozen=True)
class LoopFrame:
    id: int
    iteration_index: int
    carried_state: float
    continue_score: float | None = None
    continue_gate: float | None = None
    hard_carried_state: float | None = None
    hard_continue_score: float | None = None
    hard_alive: bool | None = None
    on_hard_path: bool = True
    # Soft value returned by the loop surrogate so far. For survival mode this
    # matches carried_state; for exit_distribution it is the exit-mass mixture.
    output_soft: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchNode:
    id: int
    candidates: list[str]
    scores: list[float]
    selected_index: int
    soft_weights: list[float]
    output_hard: float
    output_soft: float
    fidelity: FidelityMetrics | None
    relaxation: RelaxationSpec
    soft_scores: list[float] | None = None
    on_hard_path: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.fidelity is not None:
            data["fidelity"] = self.fidelity.to_dict()
        data["relaxation"] = self.relaxation.to_dict()
        return data


@dataclass(frozen=True)
class LedgerEntry:
    id: int
    surrogate_type: str
    gradient_contract: str
    bias_warning: str
    approximates: str = ""
    gradient_flows_to: list[str] = field(default_factory=list)
    estimator: str = "surrogate"
    recommended_checks: list[str] = field(default_factory=list)
    fidelity_metrics: FidelityMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.fidelity_metrics is not None:
            data["fidelity_metrics"] = self.fidelity_metrics.to_dict()
        return data

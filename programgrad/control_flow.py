"""Explicit differentiable control-flow primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from .fidelity import binary_entropy, categorical_entropy, path_agrees, scalar_gap
from .ir import BranchNode, ConditionNode, FidelityMetrics, LedgerEntry, LoopFrame, RelaxationSpec, SearchNode
from .relaxations import sigmoid_gate, softmax, straight_through_gate
from .tensor import Tensor, ensure_tensor
from .trace import current_trace

T = TypeVar("T")


def _value(value: Tensor | float | Callable[[], Tensor | float]) -> Tensor:
    if callable(value):
        value = value()
    return ensure_tensor(value)


def _record_branch(
    *,
    score: Tensor,
    selected_path: str,
    output_hard: float,
    output_soft: float | None,
    gate: float | None,
    beta: float | None,
    surrogate_type: str,
    estimator: str,
    gradient_contract: str,
    bias_warning: str,
    approximates: str,
    gradient_flows_to: list[str],
    recommended_checks: list[str],
) -> None:
    tr = current_trace()
    if tr is None:
        return

    path_agreement = None
    if gate is not None:
        path_agreement = (gate >= 0.5) == (selected_path == "true")
    metrics = FidelityMetrics(
        hard_value=output_hard,
        soft_value=output_soft,
        output_gap=scalar_gap(output_hard, output_soft),
        path_agreement=path_agreement,
        gate_entropy=binary_entropy(gate),
        temperature=beta,
    )
    relaxation = RelaxationSpec(
        surrogate_type=surrogate_type,
        temperature=beta,
        estimator=estimator,
        notes=bias_warning,
    )
    condition = ConditionNode(
        score=score.data,
        comparator="> 0",
        hard_value=score.data > 0.0,
        gate=gate,
        relaxation=relaxation,
        temperature=beta,
        gradient_contract=gradient_contract,
    )
    event_id = tr.next_id()
    branch = BranchNode(
        id=event_id,
        condition=condition,
        selected_path="true" if selected_path == "true" else "false",
        output_hard=output_hard,
        output_soft=output_soft,
        fidelity=metrics,
    )
    ledger = LedgerEntry(
        id=event_id,
        surrogate_type=surrogate_type,
        gradient_contract=gradient_contract,
        bias_warning=bias_warning,
        approximates=approximates,
        gradient_flows_to=gradient_flows_to,
        estimator=estimator,
        recommended_checks=recommended_checks,
        fidelity_metrics=metrics,
    )
    tr.record_branch(branch, ledger)


def soft_if(
    score: Tensor | float,
    true_value: Tensor | float | Callable[[], Tensor | float],
    false_value: Tensor | float | Callable[[], Tensor | float],
    *,
    beta: float = 10.0,
) -> Tensor:
    """Relax a hard branch with ``sigmoid(beta * score)``.

    The returned tensor is the soft surrogate value. The trace records the hard
    branch result, the soft value, entropy, path agreement, and a warning that
    gradients are gradients of the surrogate rather than the hard step.
    """

    score_t = ensure_tensor(score)
    true_t = _value(true_value)
    false_t = _value(false_value)
    gate = sigmoid_gate(score_t, beta=beta)
    out = gate * true_t + (1.0 - gate) * false_t

    hard_true = score_t.data > 0.0
    hard_out = true_t if hard_true else false_t
    hard_out_value = hard_out.hard_value if hard_out.hard_value is not None else hard_out.data
    out.hard_value = hard_out_value
    _record_branch(
        score=score_t,
        selected_path="true" if hard_true else "false",
        output_hard=hard_out_value,
        output_soft=out.data,
        gate=gate.data,
        beta=beta,
        surrogate_type="soft_gate",
        estimator="surrogate",
        gradient_contract=(
            "Forward value is a sigmoid-gated mixture; gradients flow through "
            "the branch values and through the gate score."
        ),
        bias_warning=(
            "This is not the exact derivative of the discontinuous hard branch; "
            "it is the derivative of the relaxed surrogate."
        ),
        approximates="hard branch indicator 1[score > 0]",
        gradient_flows_to=["true_value", "false_value", "score", "score parents"],
        recommended_checks=[
            "finite-difference gradcheck on the soft surrogate",
            "hard-vs-soft output gap",
            "temperature sensitivity",
            "hard objective after training",
        ],
    )
    return out


def diff_if(
    score: Tensor | float,
    true_fn: Callable[[], Tensor | float],
    false_fn: Callable[[], Tensor | float],
    *,
    mode: str = "pathwise",
    beta: float = 10.0,
) -> Tensor:
    """Controlled branch primitive.

    Modes:
      - ``pathwise`` evaluates only the hard branch and does not create a gate.
      - ``soft`` evaluates both branches and returns ``soft_if``.
      - ``straight_through`` uses hard forward behavior with a soft gate gradient.
    """

    score_t = ensure_tensor(score)
    hard_true = score_t.data > 0.0

    if mode == "pathwise":
        out = _value(true_fn if hard_true else false_fn)
        hard_out_value = out.hard_value if out.hard_value is not None else out.data
        out.hard_value = hard_out_value
        _record_branch(
            score=score_t,
            selected_path="true" if hard_true else "false",
            output_hard=hard_out_value,
            output_soft=None,
            gate=None,
            beta=None,
            surrogate_type="pathwise",
            estimator="exact-selected-path",
            gradient_contract=(
                "Gradients flow only through the branch that actually executed; "
                "the hard comparison contributes no useful branch-boundary gradient."
            ),
            bias_warning="Pathwise mode is honest but cannot train a hard threshold directly.",
            approximates="selected hard branch only",
            gradient_flows_to=["selected branch value"],
            recommended_checks=[
                "confirm the hard path is expected",
                "use soft mode if branch-boundary learning is required",
            ],
        )
        return out

    if mode == "soft":
        return soft_if(score_t, true_fn, false_fn, beta=beta)

    if mode != "straight_through":
        raise ValueError("mode must be one of: pathwise, soft, straight_through")

    true_t = _value(true_fn)
    false_t = _value(false_fn)
    gate = straight_through_gate(score_t, beta=beta)
    soft_gate = sigmoid_gate(score_t, beta=beta)
    out = gate * true_t + (1.0 - gate) * false_t
    soft_out = soft_gate * true_t + (1.0 - soft_gate) * false_t
    hard_out = true_t if hard_true else false_t
    hard_out_value = hard_out.hard_value if hard_out.hard_value is not None else hard_out.data
    out.hard_value = hard_out_value
    _record_branch(
        score=score_t,
        selected_path="true" if hard_true else "false",
        output_hard=hard_out_value,
        output_soft=soft_out.data,
        gate=soft_gate.data,
        beta=beta,
        surrogate_type="straight_through",
        estimator="hard-forward-soft-backward",
        gradient_contract=(
            "Forward pass uses the hard branch; backward pass uses the sigmoid "
            "surrogate gate gradient."
        ),
        bias_warning="Straight-through gradients are biased estimators, not true hard-branch derivatives.",
        approximates="hard branch forward pass with sigmoid surrogate backward pass",
        gradient_flows_to=["selected branch value", "score via surrogate gradient"],
        recommended_checks=[
            "compare against soft mode",
            "hard objective after training",
            "temperature sensitivity",
        ],
    )
    return out


def soft_argmax(scores: Sequence[Tensor | float], *, tau: float = 1.0) -> list[Tensor]:
    """Return softmax weights for differentiable candidate selection."""

    return softmax(list(scores), tau=tau)


def soft_select(
    values: Sequence[Tensor | float],
    scores: Sequence[Tensor | float],
    *,
    tau: float = 1.0,
    candidate_names: Sequence[str] | None = None,
) -> Tensor:
    """Select among values with a hard argmax trace and softmax surrogate."""

    if len(values) != len(scores):
        raise ValueError("values and scores must have the same length")
    if not values:
        raise ValueError("soft_select requires at least one candidate")

    value_tensors = [ensure_tensor(value) for value in values]
    score_tensors = [ensure_tensor(score) for score in scores]
    weights = softmax(score_tensors, tau=tau)
    out = Tensor(0.0)
    for weight, value in zip(weights, value_tensors):
        out = out + weight * value

    hard_index = max(range(len(score_tensors)), key=lambda i: score_tensors[i].data)
    hard_out = value_tensors[hard_index]
    hard_out_value = hard_out.hard_value if hard_out.hard_value is not None else hard_out.data
    out.hard_value = hard_out_value

    tr = current_trace()
    if tr is not None:
        weight_values = [weight.data for weight in weights]
        metrics = FidelityMetrics(
            hard_value=hard_out_value,
            soft_value=out.data,
            output_gap=scalar_gap(hard_out.data, out.data),
            path_agreement=path_agrees(hard_index, weight_values),
            gate_entropy=categorical_entropy(weight_values),
            temperature=tau,
        )
        relaxation = RelaxationSpec(
            surrogate_type="softmax_select",
            temperature=tau,
            estimator="surrogate",
            notes="Hard argmax is paired with softmax-weighted candidate values.",
        )
        event_id = tr.next_id()
        names = list(candidate_names) if candidate_names is not None else [
            f"candidate_{idx}" for idx in range(len(values))
        ]
        node = SearchNode(
            id=event_id,
            candidates=names,
            scores=[score.data for score in score_tensors],
            selected_index=hard_index,
            soft_weights=weight_values,
            output_hard=hard_out_value,
            output_soft=out.data,
            fidelity=metrics,
            relaxation=relaxation,
        )
        ledger = LedgerEntry(
            id=event_id,
            surrogate_type="softmax_select",
            gradient_contract=(
                "Forward surrogate is the softmax-weighted expected value; "
                "gradients flow into candidate values and selection scores."
            ),
            bias_warning=(
                "The gradient optimizes the soft expected value, so the learned "
                "scores must still be evaluated with the original hard argmax."
            ),
            approximates="hard argmax over candidate scores",
            gradient_flows_to=["candidate values", "candidate scores", "score parents"],
            estimator="surrogate",
            recommended_checks=[
                "hard argmax objective after training",
                "path agreement",
                "candidate entropy",
                "temperature sensitivity",
            ],
            fidelity_metrics=metrics,
        )
        tr.record_search(node, ledger)
    return out


def bounded_loop(
    init_state: Tensor | float,
    steps: int,
    body_fn: Callable[[int, Tensor], Tensor | float],
    continue_score_fn: Callable[[int, Tensor], Tensor | float],
    *,
    beta: float = 10.0,
) -> Tensor:
    """Bounded early-exit relaxation using survival gates.

    This is intentionally limited: it relaxes a fixed maximum number of steps
    rather than claiming support for arbitrary data-dependent Python loops.
    """

    state = ensure_tensor(init_state)
    alive = Tensor(1.0)
    tr = current_trace()
    for index in range(steps):
        candidate = ensure_tensor(body_fn(index, state))
        continue_score = ensure_tensor(continue_score_fn(index, state))
        gate = sigmoid_gate(continue_score, beta=beta)
        alive = alive * gate
        state = alive * candidate + (1.0 - alive) * state
        if tr is not None:
            tr.record_loop(
                LoopFrame(
                    id=tr.next_id(),
                    iteration_index=index,
                    carried_state=state.data,
                    stop_score=continue_score.data,
                    continue_gate=gate.data,
                )
            )
    return state

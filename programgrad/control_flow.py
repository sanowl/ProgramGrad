"""Explicit differentiable control-flow primitives."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from operator import index as integer_index

from typing import Literal

from .fidelity import binary_entropy, categorical_entropy, path_agrees, scalar_gap
from .ir import BranchNode, ConditionNode, FidelityMetrics, LedgerEntry, LoopFrame, RelaxationSpec, SearchNode
from .relaxations import (
    gumbel_softmax,
    gumbel_softmax_straight_through,
    sigmoid_gate,
    softmax,
    straight_through_gates,
    validate_temperature,
)
from .tensor import Tensor, ensure_tensor, hard_data
from .trace import current_trace, in_soft_only, soft_only_region

BranchValue = Tensor | float | Callable[[], Tensor | float]
BranchPath = Literal["true", "false"]


@dataclass(frozen=True)
class _SelectModeSpec:
    surrogate_type: str
    estimator: str
    notes: str
    gradient_contract: str
    bias_warning: str


_SELECT_MODES: dict[str, _SelectModeSpec] = {
    "softmax": _SelectModeSpec(
        surrogate_type="softmax_select",
        estimator="surrogate",
        notes="Hard argmax is paired with softmax-weighted candidate values.",
        gradient_contract=(
            "Forward surrogate is the softmax-weighted expected value; "
            "gradients flow into candidate values and selection scores."
        ),
        bias_warning=(
            "The gradient optimizes the soft expected value, so the learned "
            "scores must still be evaluated with the original hard argmax."
        ),
    ),
    "gumbel": _SelectModeSpec(
        surrogate_type="gumbel_softmax_select",
        estimator="surrogate",
        notes="Hard argmax is paired with a Concrete / Gumbel-Softmax mixture.",
        gradient_contract=(
            "Forward surrogate is the Gumbel-Softmax expected value; "
            "gradients flow into candidate values and selection scores."
        ),
        bias_warning=(
            "Gumbel-Softmax optimizes a noisy Concrete relaxation; evaluate the "
            "learned scores with the original hard argmax."
        ),
    ),
    "gumbel_st": _SelectModeSpec(
        surrogate_type="gumbel_straight_through_select",
        estimator="hard-forward-soft-backward",
        notes=(
            "Hard program argmax is traced separately from the Gumbel sample used "
            "in the straight-through forward pass."
        ),
        gradient_contract=(
            "Forward pass uses a hard Gumbel sample; backward pass uses the "
            "Concrete / Gumbel-Softmax surrogate gradient."
        ),
        bias_warning=(
            "Straight-through Gumbel gradients are biased estimators; always "
            "check the hard argmax objective after training."
        ),
    ),
}


def _resolve_branch_value(value: BranchValue) -> Tensor:
    if callable(value):
        value = value()
    return ensure_tensor(value)


def _evaluate_both_branches(
    hard_true: bool,
    true_value: BranchValue,
    false_value: BranchValue,
) -> tuple[Tensor, Tensor]:
    """Evaluate both sides; mark the unselected side as soft-only."""

    if hard_true:
        true_t = _resolve_branch_value(true_value)
        with soft_only_region():
            false_t = _resolve_branch_value(false_value)
            # Soft-only results must not poison the hard-path mixture metadata.
            _clear_hard(false_t)
    else:
        with soft_only_region():
            true_t = _resolve_branch_value(true_value)
            _clear_hard(true_t)
        false_t = _resolve_branch_value(false_value)
    return true_t, false_t


def _attach_hard(output: Tensor, hard_value: float) -> float:
    """Attach a concrete hard-program value, clearing any deferred hard error."""

    from .trace import hard_shadow_enabled

    if not hard_shadow_enabled():
        return hard_value
    output.hard_value = hard_value
    output.hard_error = None
    return hard_value


def _clear_hard(output: Tensor) -> None:
    output.hard_value = None
    output.hard_error = None


def _decision_score(score: Tensor, *, require_hard: bool) -> float:
    """Score used for hard-path decisions; soft-only paths use ``.data``."""

    return hard_data(score) if require_hard else score.data


def _gated_mixture(gate: Tensor, true_value: Tensor, false_value: Tensor) -> Tensor:
    return gate * true_value + (1.0 - gate) * false_value


def _require_finite_score(
    score: Tensor,
    *,
    name: str = "score",
    require_hard: bool = True,
) -> float:
    """Validate score finiteness and return the hard score used for decisions."""

    if not math.isfinite(score.data):
        raise ValueError(f"{name} must be finite")
    decision = _decision_score(score, require_hard=require_hard)
    if require_hard and not math.isfinite(decision):
        raise ValueError(f"{name} hard value must be finite")
    return decision


def _record_branch(
    *,
    score: Tensor,
    hard_score: float,
    selected_path: BranchPath,
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

    metrics = None
    if tr.fidelity:
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
        score=hard_score,
        comparator="> 0",
        hard_value=hard_score > 0.0,
        gate=gate,
        relaxation=relaxation,
        temperature=beta,
        gradient_contract=gradient_contract,
        soft_score=score.data,
    )
    event_id = tr.next_id()
    branch = BranchNode(
        id=event_id,
        condition=condition,
        selected_path=selected_path,
        output_hard=output_hard,
        output_soft=output_soft,
        fidelity=metrics,
        on_hard_path=not in_soft_only(),
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


def _record_search(
    *,
    names: list[str],
    hard_scores: list[float],
    soft_scores: list[float],
    hard_index: int,
    soft_weights: list[float],
    output_hard: float,
    output_soft: float,
    tau: float,
    surrogate_type: str = "softmax_select",
    estimator: str = "surrogate",
    notes: str = "Hard argmax is paired with softmax-weighted candidate values.",
    gradient_contract: str = (
        "Forward surrogate is the softmax-weighted expected value; "
        "gradients flow into candidate values and selection scores."
    ),
    bias_warning: str = (
        "The gradient optimizes the soft expected value, so the learned "
        "scores must still be evaluated with the original hard argmax."
    ),
) -> None:
    tr = current_trace()
    if tr is None:
        return

    metrics = None
    if tr.fidelity:
        metrics = FidelityMetrics(
            hard_value=output_hard,
            soft_value=output_soft,
            output_gap=scalar_gap(output_hard, output_soft),
            path_agreement=path_agrees(hard_index, soft_weights),
            gate_entropy=categorical_entropy(soft_weights),
            temperature=tau,
        )
    event_id = tr.next_id()
    node = SearchNode(
        id=event_id,
        candidates=names,
        scores=hard_scores,
        selected_index=hard_index,
        soft_weights=soft_weights,
        output_hard=output_hard,
        output_soft=output_soft,
        fidelity=metrics,
        relaxation=RelaxationSpec(
            surrogate_type=surrogate_type,
            temperature=tau,
            estimator=estimator,
            notes=notes,
        ),
        soft_scores=soft_scores,
        on_hard_path=not in_soft_only(),
    )
    ledger = LedgerEntry(
        id=event_id,
        surrogate_type=surrogate_type,
        gradient_contract=gradient_contract,
        bias_warning=bias_warning,
        approximates="hard argmax over candidate scores",
        gradient_flows_to=["candidate values", "candidate scores", "score parents"],
        estimator=estimator,
        recommended_checks=[
            "hard argmax objective after training",
            "path agreement",
            "candidate entropy",
            "temperature sensitivity",
        ],
        fidelity_metrics=metrics,
    )
    tr.record_search(node, ledger)


def soft_if(
    score: Tensor | float,
    true_value: BranchValue,
    false_value: BranchValue,
    *,
    beta: float = 10.0,
) -> Tensor:
    """Relax a hard branch with ``sigmoid(beta * score)``.

    The returned tensor is the soft surrogate value. The trace records the hard
    branch result, the soft value, entropy, path agreement, and a warning that
    gradients are gradients of the surrogate rather than the hard step.
    """

    score_t = ensure_tensor(score)
    beta = validate_temperature(beta, name="beta")
    hard_score = _require_finite_score(score_t, require_hard=not in_soft_only())
    hard_true = hard_score > 0.0
    true_t, false_t = _evaluate_both_branches(hard_true, true_value, false_value)
    gate = sigmoid_gate(score_t, beta=beta)
    out = _gated_mixture(gate, true_t, false_t)

    hard_out = true_t if hard_true else false_t
    if in_soft_only():
        hard_out_value = hard_out.data
    else:
        hard_out_value = _attach_hard(out, hard_data(hard_out))
    _record_branch(
        score=score_t,
        hard_score=hard_score,
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

    if mode not in {"pathwise", "soft", "straight_through"}:
        raise ValueError("mode must be one of: pathwise, soft, straight_through")

    score_t = ensure_tensor(score)
    on_hard_path = not in_soft_only()
    hard_score = _require_finite_score(score_t, require_hard=on_hard_path)
    hard_true = hard_score > 0.0

    if mode == "pathwise":
        out = _resolve_branch_value(true_fn if hard_true else false_fn)
        if on_hard_path:
            hard_out_value = _attach_hard(out, hard_data(out))
        else:
            hard_out_value = out.data
        _record_branch(
            score=score_t,
            hard_score=hard_score,
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

    beta = validate_temperature(beta, name="beta")
    true_t, false_t = _evaluate_both_branches(hard_true, true_fn, false_fn)
    gate, soft_gate = straight_through_gates(score_t, beta=beta)
    out = _gated_mixture(gate, true_t, false_t)
    soft_out = _gated_mixture(soft_gate, true_t, false_t)
    hard_out = true_t if hard_true else false_t
    if on_hard_path:
        hard_out_value = _attach_hard(out, hard_data(hard_out))
    else:
        hard_out_value = hard_out.data
    _record_branch(
        score=score_t,
        hard_score=hard_score,
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
    """Return softmax weights for differentiable candidate selection.

    This is an untraced building block. Prefer ``soft_select`` when you need a
    hard argmax paired with ledger / fidelity metadata.
    """

    return softmax(list(scores), tau=tau)


def soft_select(
    values: Sequence[Tensor | float],
    scores: Sequence[Tensor | float],
    *,
    tau: float = 1.0,
    mode: str = "softmax",
    seed: int | None = None,
    candidate_names: Sequence[str] | None = None,
) -> Tensor:
    """Select among values with a hard argmax trace and a soft surrogate.

    Modes:
      - ``softmax``: expected value under softmax weights
      - ``gumbel``: Concrete / Gumbel-Softmax expected value
      - ``gumbel_st``: hard Gumbel sample forward, soft Concrete backward
    """

    spec = _SELECT_MODES.get(mode)
    if spec is None:
        raise ValueError("mode must be one of: softmax, gumbel, gumbel_st")
    if len(values) != len(scores):
        raise ValueError("values and scores must have the same length")
    if not values:
        raise ValueError("soft_select requires at least one candidate")
    tau = validate_temperature(tau, name="tau")

    if candidate_names is None:
        names = [f"candidate_{idx}" for idx in range(len(values))]
    else:
        names = list(candidate_names)
        if len(names) != len(values):
            raise ValueError("candidate_names must have the same length as values")
        if any(not isinstance(name, str) for name in names):
            raise TypeError("candidate_names must contain only strings")

    value_tensors = [ensure_tensor(value) for value in values]
    score_tensors = [ensure_tensor(score) for score in scores]
    if mode == "softmax":
        mix_weights = softmax(score_tensors, tau=tau)
        report_weights = mix_weights
    elif mode == "gumbel":
        mix_weights = gumbel_softmax(score_tensors, tau=tau, seed=seed)
        report_weights = mix_weights
    else:
        mix_weights, report_weights, _sampled = gumbel_softmax_straight_through(
            score_tensors,
            tau=tau,
            seed=seed,
        )

    out = Tensor(0.0)
    for weight, value in zip(mix_weights, value_tensors):
        out = out + weight * value

    soft_scores = [score.data for score in score_tensors]
    if in_soft_only():
        for score in soft_scores:
            if not math.isfinite(score):
                raise ValueError("soft_select scores must be finite")
        hard_scores = soft_scores
        hard_index = max(range(len(score_tensors)), key=soft_scores.__getitem__)
        hard_out_value = value_tensors[hard_index].data
        _clear_hard(out)
    else:
        hard_scores = [hard_data(score) for score in score_tensors]
        hard_index = max(range(len(score_tensors)), key=hard_scores.__getitem__)
        hard_out_value = _attach_hard(out, hard_data(value_tensors[hard_index]))
    _record_search(
        names=names,
        hard_scores=hard_scores,
        soft_scores=soft_scores,
        hard_index=hard_index,
        soft_weights=[weight.data for weight in report_weights],
        output_hard=hard_out_value,
        output_soft=out.data,
        tau=tau,
        surrogate_type=spec.surrogate_type,
        estimator=spec.estimator,
        notes=spec.notes,
        gradient_contract=spec.gradient_contract,
        bias_warning=spec.bias_warning,
    )
    return out


def _loop_body_hard_value(candidate: Tensor, *, state: Tensor) -> float:
    """Read the hard body update, refusing silent soft fallback."""

    if candidate.hard_error is not None:
        return hard_data(candidate)
    if candidate.hard_value is not None:
        return candidate.hard_value
    if state.hard_value is None:
        return candidate.data
    raise ValueError(
        "bounded_loop body dropped hard_value metadata while the hard "
        "program was still alive; return Tensor results from Tensor ops "
        "(avoid reading .data inside the body)."
    )


def _validate_bounded_loop_args(
    steps: int,
    *,
    mode: str,
    beta: float,
) -> tuple[int, float]:
    if mode not in {"survival", "exit_distribution"}:
        raise ValueError("mode must be one of: survival, exit_distribution")
    if isinstance(steps, bool):
        raise TypeError("steps must be an integer, not bool")
    try:
        step_count = integer_index(steps)
    except TypeError:
        raise TypeError("steps must be an integer") from None
    if step_count < 0:
        raise ValueError("steps must be non-negative")
    return step_count, validate_temperature(beta, name="beta")


def bounded_loop(
    init_state: Tensor | float,
    steps: int,
    body_fn: Callable[[int, Tensor], Tensor | float],
    continue_score_fn: Callable[[int, Tensor], Tensor | float],
    *,
    beta: float = 10.0,
    mode: str = "survival",
) -> Tensor:
    """Bounded early-exit relaxation.

    Modes:
      - ``survival``: soft state is carried by a running survival gate
      - ``exit_distribution``: soft output is the expectation under the discrete
        exit-mass distribution over step candidates, while bodies still see a
        survival-carried soft state

    This is intentionally limited: it relaxes a fixed maximum number of steps
    rather than claiming support for arbitrary data-dependent Python loops.
    Once the carried state has a hard shadow, the body must return a ``Tensor``
    that preserves ``hard_value`` (use Tensor ops; do not escape via ``.data``).
    """

    step_count, beta = _validate_bounded_loop_args(steps, mode=mode, beta=beta)

    state = ensure_tensor(init_state)
    alive = Tensor(1.0)
    survival = Tensor(1.0)
    soft_out = Tensor(0.0)
    hard_state = hard_data(state)
    hard_alive = True
    tr = current_trace()
    for index in range(step_count):
        decision_on_hard_path = hard_alive
        if hard_alive:
            candidate = ensure_tensor(body_fn(index, state))
            continue_score = ensure_tensor(continue_score_fn(index, state))
        else:
            with soft_only_region():
                candidate = ensure_tensor(body_fn(index, state))
                continue_score = ensure_tensor(continue_score_fn(index, state))
        _require_finite_score(
            continue_score,
            name="continue_score",
            require_hard=decision_on_hard_path,
        )
        gate = sigmoid_gate(continue_score, beta=beta)

        # After the hard program stops, keep unrolling the soft surrogate only.
        # Do not consult hard continue scores again — that can abort a valid
        # soft forward and pollute hard-path metadata with soft values.
        hard_continue_score: float | None = None
        if hard_alive:
            hard_continue_score = hard_data(continue_score)
            if hard_continue_score > 0.0:
                hard_state = _loop_body_hard_value(candidate, state=state)
            else:
                hard_alive = False

        if mode == "exit_distribution":
            if index < step_count - 1:
                exit_mass = survival * (1.0 - gate)
                survival = survival * gate
            else:
                exit_mass = survival
            soft_out = soft_out + exit_mass * candidate

        alive = alive * gate
        state = alive * candidate + (1.0 - alive) * state
        # Do not execute hard metadata through later bodies after the original
        # program has stopped; only the relaxed shadow continues to unroll.
        if hard_alive:
            _attach_hard(state, hard_state)
        else:
            _clear_hard(state)
        if tr is not None:
            output_soft = soft_out.data if mode == "exit_distribution" else state.data
            tr.record_loop(
                LoopFrame(
                    id=tr.next_id(),
                    iteration_index=index,
                    carried_state=state.data,
                    continue_score=continue_score.data,
                    continue_gate=gate.data,
                    hard_carried_state=hard_state,
                    hard_continue_score=hard_continue_score,
                    hard_alive=hard_alive,
                    on_hard_path=decision_on_hard_path,
                    output_soft=output_soft,
                )
            )

    if mode == "exit_distribution":
        result = state if step_count == 0 else soft_out
        _attach_hard(result, hard_state)
        return result

    _attach_hard(state, hard_state)
    return state

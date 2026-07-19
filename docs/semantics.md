# ProgramGrad Semantics

ProgramGrad separates the hard program from the soft surrogate.

For a branch score `s(x, theta)`, true branch `T(x)`, and false branch `F(x)`,
the hard program is:

```text
H(x, theta) = 1[s(x, theta) > 0] * T(x)
            + (1 - 1[s(x, theta) > 0]) * F(x)
```

The default soft surrogate uses a sigmoid gate:

```text
g = sigmoid(beta * s(x, theta))
S(x, theta, beta) = g * T(x) + (1 - g) * F(x)
```

The useful branch-boundary gradient is a surrogate gradient:

```text
dS/dtheta = g * dT/dtheta + (1 - g) * dF/dtheta
            + (T - F) * dg/dtheta
```

The trace ledger records this distinction. ProgramGrad should not claim that
the hard step function has a useful exact derivative at the decision boundary.
It optimizes the surrogate and reports whether the surrogate remains faithful
to the hard program.

## Modes

`pathwise` evaluates the branch the hard program actually selected. Gradients
flow through that selected branch only.

`soft` evaluates a smooth mixture of branch values. Gradients flow through both
branch values and through the gate score.

`straight_through` uses a hard forward decision and a soft backward gate. It is
reported as a biased estimator in the ledger.

## Selection relaxations

`soft_select` supports three surrogate modes:

- `softmax`: expected value under softmax weights
- `gumbel`: Concrete / Gumbel-Softmax mixture (optional `seed` for reproducibility)
- `gumbel_st`: hard Gumbel sample forward with Concrete backward (straight-through)

In all modes the traced hard program remains argmax over hard scores. Soft
weights are what training sees.

`soft_argmax` is a low-level, untraced helper that returns softmax weights only.
It does not write search/ledger/fidelity events; use `soft_select` when you need
a traced hard argmax paired with a soft surrogate.

## Hybrid training objective

`hybrid_loss(result, target, gap_weight=1.0)` optimizes

```text
L = (soft - target)^2 + gap_weight * (soft - hard)^2
```

so the soft surrogate is pulled toward both the task target and the hard-program
value. Use `hard_squared_loss` to report the hard objective without gradients.

## Nested decisions

Every relaxed control-flow output carries its original hard-program value as a
shadow alongside the differentiable soft value. Deterministic tensor operations
propagate both values. If the hard shadow leaves the real domain while the soft
path remains valid, the soft result stays available and the tensor retains a
deferred hard error. Read that value with `hard_data(...)`: it raises when a
deferred hard error is present instead of silently substituting the soft value.
Valid hard decisions still consult the hard shadow while gates and softmax
weights use the soft score. Traces record both scores so path disagreement
remains visible. Nested decisions evaluated only to build an unselected soft
branch are marked `on_hard_path=False` and are omitted from `hard_path` /
hard-vs-soft evaluation rows. Those soft-only nested decisions use soft scores
for local metadata and must not abort a valid surrogate forward when a deferred
hard error sits on an off-path score.

## Bounded loops

`bounded_loop` validates a finite positive gate temperature and a non-negative
integer step bound. Two soft modes are available:

- `survival` (default): soft state is carried by a running survival gate
- `exit_distribution`: soft output is the expectation under the discrete exit
  mass over step candidates, while bodies still see survival-carried soft state

Hard early-stop tracking is shared by both modes. Once the carried state has a
hard shadow, the body must keep returning `Tensor` values that preserve
`hard_value` (do not escape through `.data`). This is still a controlled
relaxation rather than general Python loop differentiation.
Loop frames expose hard continue/stop decisions and hard/soft carried state in
JSON, text, hard-path, and SVG trace views. Fidelity / `hard_soft_rows` compare
the frozen hard state against the loop's returned soft value (`output_soft`):
for `survival` that matches the carried state; for `exit_distribution` it is the
exit-mass mixture, not the internal survival-carried state bodies still see.

## Fidelity

Trace fidelity reports currently include hard-soft output gap, path agreement,
gate entropy, and temperature. Demos should always evaluate the learned
parameters with the original hard decision rule, not only the soft loss.
Passing `fidelity=False` keeps structural trace and ledger events but omits the
metric payloads. `hard_soft_rows` still builds comparison rows from structural
branch, search, and final loop outputs; gap, entropy, and agreement stay unset
unless fidelity metrics were recorded.

## Training fast path

Optimization loops should avoid per-op tracing. `trace(...)` defaults to
`record_ops=False`. For pure training steps use:

```python
with training_mode(hard_shadow=False):
    y = soft_if(...)
    loss = (y - target) ** 2
    loss.backward()
```

`training_mode(hard_shadow=False)` skips hard-shadow propagation so only the soft
surrogate graph is built. Re-enable a full `trace(fidelity=True)` (or
`training_trace(...)` for a light decision log) when you need hard-path reports.
Nested hard-shadow semantics require `hard_shadow=True`. Note the default
asymmetry: `training_mode` defaults to `hard_shadow=False` (cheap soft steps),
while `training_trace` defaults to `hard_shadow=True` (decision fidelity).
`fidelity=True` also requires `hard_shadow=True`.

## Semantic ledger

Every relaxed decision writes a ledger entry with:

- the hard operation being approximated;
- the surrogate or estimator type;
- where gradients are allowed to flow;
- the bias warning for the estimator;
- recommended checks such as gradcheck, hard-vs-soft gap, path agreement, and
  temperature sensitivity.

This makes the gradient contract inspectable instead of implicit in the code.

## Temperature sensitivity

The same hard program can behave differently as beta or tau changes. ProgramGrad
therefore includes a lightweight temperature sensitivity report. A strong demo
should show not only that the soft loss decreased, but also how hard-soft gap,
entropy, and agreement behave across a small temperature sweep. When the run
returns a trace, soft values prefer the matched fidelity event (so
straight-through / `gumbel_st` rows report the surrogate soft value, not the
hard-forward `.data`). Without a trace, STE rows fall back to hard-forward
`.data` as `soft_value`.

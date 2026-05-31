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

## Fidelity

Trace fidelity reports currently include hard-soft output gap, path agreement,
gate entropy, and temperature. Demos should always evaluate the learned
parameters with the original hard decision rule, not only the soft loss.


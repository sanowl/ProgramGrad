# Evaluation Reports

ProgramGrad evaluates the surrogate and the original hard program separately.
This is a core design rule: improving the soft loss is useful only when the
learned parameters also improve the hard program.

## Hard-vs-soft tables

Use `hard_soft_rows` and `format_hard_soft_table` on a completed trace:

```python
from programgrad import format_hard_soft_table, hard_soft_rows

rows = hard_soft_rows(trace, target=3.2)
print(format_hard_soft_table(rows))
```

The table reports:

- hard value: the result selected by the original branch or argmax;
- soft value: the relaxed surrogate result;
- gap: absolute hard-soft output difference;
- agreement: whether the hard path matches the highest-probability soft path;
- entropy: uncertainty in the branch gate or candidate distribution;
- hard/soft loss: squared loss against the target when a target is provided.

## Temperature sensitivity

Temperature controls the optimization/fidelity tradeoff. For branches, beta
sharpens the sigmoid gate. For selection, tau smooths or sharpens softmax
weights.

```python
from programgrad import format_temperature_table, temperature_sensitivity

rows = temperature_sensitivity(run_once, [0.25, 0.5, 1.0], target=3.2)
print(format_temperature_table(rows))
```

`run_once(temp)` should return either a `Tensor` or a tuple containing a
`Tensor` and a `TraceContext`. When a trace is returned, ProgramGrad includes
the latest fidelity metrics in the report.

## Reading the report

Low gap and low entropy usually mean the surrogate is close to the hard
decision. High entropy can be useful early in training because it gives smoother
credit assignment, but it may indicate a weak approximation of the hard program.

The recommended workflow is:

1. Validate gradients on the surrogate with `gradcheck`.
2. Train using the surrogate.
3. Report hard-vs-soft evaluation on the final trace.
4. Sweep temperature to show whether the conclusion is stable.
5. Evaluate the final parameters with the original hard program.


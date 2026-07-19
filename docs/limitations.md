# Limitations

ProgramGrad v0.1 alpha is intentionally narrow.

- It is scalar-only and pure Python.
- It does not replace PyTorch, JAX, TensorFlow, Enzyme, or compiler AD.
- It does not support arbitrary Python syntax or automatic AST rewriting.
- Hard branches do not become truly differentiable. ProgramGrad exposes
  surrogate gradients and their bias risks instead.
- Bounded loop support is a controlled relaxation, not a universal treatment of
  data-dependent loops.
- Soft branch and bounded-loop bodies are evaluated to build the surrogate even
  when the original hard program would not execute them, so those relaxed paths
  must still be valid on their soft inputs.
- Hard-shadow arithmetic can diverge from the soft domain; the soft forward
  continues and the tensor keeps a deferred hard error. Later hard decisions
  that call `hard_data` raise instead of silently substituting the soft value.
- `gradcheck` validates the soft surrogate graph. Straight-through hard-forward
  behavior is not expected to match finite differences.
- `training_mode(hard_shadow=False)` is for soft-surrogate optimization only; it
  disables nested hard-shadow bookkeeping until you re-enter a normal trace.
- The SVG exporter is a lightweight trace view intended for examples and tests,
  not a full browser inspector.

This narrow scope is deliberate. The project is a trace laboratory for
decision-level differentiable programming, especially branches, thresholds,
argmax choices, and small reasoning/search programs.

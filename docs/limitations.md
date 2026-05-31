# Limitations

ProgramGrad v0.1 alpha is intentionally narrow.

- It is scalar-only and pure Python.
- It does not replace PyTorch, JAX, TensorFlow, Enzyme, or compiler AD.
- It does not support arbitrary Python syntax or automatic AST rewriting.
- Hard branches do not become truly differentiable. ProgramGrad exposes
  surrogate gradients and their bias risks instead.
- Bounded loop support is a controlled relaxation, not a universal treatment of
  data-dependent loops.
- The SVG exporter is a lightweight trace view intended for examples and tests,
  not a full browser inspector.

This narrow scope is deliberate. The project is a trace laboratory for
decision-level differentiable programming, especially branches, thresholds,
argmax choices, and small reasoning/search programs.


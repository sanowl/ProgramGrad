# Changelog

All notable changes to ProgramGrad are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [PEP 440](https://peps.python.org/pep-0440/) versioning.

## [Unreleased]

### Added

- `soft_select(..., mode="gumbel"|"gumbel_st")` Concrete / straight-through Gumbel selection
- `bounded_loop(..., mode="exit_distribution")` soft exit-mass mixture over step candidates
- `hybrid_loss` and `hard_squared_loss` for soft training with hard-gap regularization

## [0.1.0a0] - 2026-07-19

First public alpha of ProgramGrad: a scalar research lab for hard program
traces paired with soft surrogate gradients.

### What this alpha is

- Scalar reverse-mode autodiff with a readable operator surface
- Explicit control-flow primitives: `soft_if`, `diff_if`, `soft_select`,
  `soft_argmax`, `bounded_loop`
- Hard-shadow propagation for nested discrete decisions
- Trace IR with branches, searches, loops, ledger entries, and fidelity metrics
- Evaluation helpers: hard-vs-soft tables and temperature sensitivity
- Training fast path: `training_mode` / `training_trace` (cheap defaults)
- SVG/JSON trace export and two portfolio demos

### What this alpha is not

- Not a PyTorch / JAX / TensorFlow replacement
- Not array- or GPU-oriented
- Not automatic AST rewriting of arbitrary Python
- Not a claim that hard branches have exact useful derivatives

### Added

- Hard-shadow metadata (`hard_value` / deferred `hard_error`) through tensor ops
- Soft-only nested decision marking (`on_hard_path`, `soft_only_region`)
- Loop hard/soft continue-score frames and final-loop evaluation rows
- `training_mode(hard_shadow=...)` and `training_trace()` presets
- Context-local hot-path flags so op recording is skipped unless requested
- Typed package marker (`py.typed`)

### Changed

- `trace(..., record_ops=False)` is now the default
- Loop IR fields renamed to `continue_score` / `hard_continue_score`
- Demo training loops use the fast path, then re-trace with fidelity for reports

### Fixed

- Hard decisions no longer silently follow soft surrogate scores in nested graphs
- Hard-shadow domain failures no longer abort a valid soft forward
- Post-stop loop unrolls no longer pollute hard-path metadata
- Temperature sensitivity matches the returned tensor, not an unrelated later event

[0.1.0a0]: https://github.com/sanowl/ProgramGrad/releases/tag/v0.1.0a0

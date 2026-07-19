# ProgramGrad

[![CI](https://github.com/sanowl/ProgramGrad/actions/workflows/ci.yml/badge.svg)](https://github.com/sanowl/ProgramGrad/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![Status](https://img.shields.io/badge/status-v0.1--alpha-orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Gradients over programs, not only tensors.

ProgramGrad is a small research-oriented framework for differentiable discrete
program decisions. It records the hard execution trace of a Python-like
algorithm, builds a differentiable soft shadow trace, and reports what each
surrogate gradient means.

It is not a PyTorch replacement. It is a trace laboratory for branches, loops,
argmax choices, thresholds, and reasoning/search programs.

## What is implemented

- Scalar reverse-mode `Tensor` with operator overloads.
- Finite-difference `gradcheck`.
- Trace IR for ops, branches, searches, loop frames, and semantic ledger entries.
- `soft_if`, `diff_if`, `soft_argmax`, `soft_select`, and bounded loop relaxation.
- Selection modes: softmax, Gumbel-Softmax, and Gumbel straight-through.
- Hybrid soft+hard-gap training loss (`hybrid_loss`).
- Loop soft modes: survival gates and exit-mass mixtures.
- Hard-shadow propagation through nested arithmetic and discrete decisions.
- Hard-soft fidelity reports: output gap, path agreement, entropy, temperature.
- Hard-vs-soft evaluation tables and temperature sensitivity reports.
- Context-local traces that remain isolated across threads and async contexts.
- Training fast path: `training_mode` / `training_trace` with cheap defaults.
- SVG and JSON trace export.
- Demos for learnable branch thresholds and tiny differentiable tree search.

## Install

ProgramGrad is not published to PyPI yet. Install it from a checkout:

```bash
git clone https://github.com/sanowl/ProgramGrad.git
cd ProgramGrad
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Or install directly from GitHub:

```bash
python -m pip install "git+https://github.com/sanowl/ProgramGrad.git"
```

A manual GitHub Actions workflow (`.github/workflows/publish.yml`) is ready for
Trusted Publishing to PyPI when you want the first public upload. See
[CHANGELOG.md](CHANGELOG.md) for the alpha release notes.
## Example

```python
from programgrad import Tensor, soft_if, trace

x = Tensor(0.7, name="x")
threshold = Tensor(1.0, requires_grad=True, name="threshold")

with trace(mode="dual", relaxation="soft_gate", fidelity=True) as tr:
    score = x - threshold
    y = soft_if(score, true_value=2 * x, false_value=x**2, beta=5.0)
    loss = (y - 1.4) ** 2
    loss.backward()

print(threshold.grad)
print(tr.show())
tr.export_svg("threshold_trace.svg")
```

The trace reports the hard branch, the soft gate, the hard-soft output gap, and
a ledger warning that the gradient is a surrogate gradient rather than the true
derivative of a discontinuous branch.

Nested relaxed decisions retain both values: arithmetic uses the soft value for
gradients while later hard branches and argmax operations consult the propagated
hard shadow. Trace records expose both hard and soft decision scores when they
diverge.

The demos train inside `training_mode(hard_shadow=False)` for speed, then
re-evaluate with a full fidelity trace after the last optimizer update so the
reported trace, learned parameters, and hard-program evaluation describe the
same final state.

## Trace screenshots

The demo traces are intentionally inspectable. They show the hard decision,
soft surrogate value, fidelity gap, and semantic gradient warning.

### Learnable branch threshold

![ProgramGrad threshold trace](https://raw.githubusercontent.com/sanowl/ProgramGrad/main/docs/assets/threshold-demo.svg)

### Tiny differentiable tree search

![ProgramGrad tree search trace](https://raw.githubusercontent.com/sanowl/ProgramGrad/main/docs/assets/tree-search-demo.svg)

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Run demos

```bash
python examples/learnable_threshold.py
python examples/tiny_tree_search.py
```

Each demo prints the trace ledger and writes an SVG trace in the current
directory. They also print hard-vs-soft evaluation tables and temperature
sensitivity reports. The CI workflow runs both demos after installing the
package.

## Project position

ProgramGrad focuses on interpretability and semantics, not speed or broad
operator coverage. Existing AD systems are excellent for tensor programs and
compiler lowering. ProgramGrad targets a narrower gap: making the gradient of an
algorithmic decision understandable.

For deeper details, see [semantics](docs/semantics.md),
[evaluation reports](docs/evaluation.md), [limitations](docs/limitations.md),
and the [changelog](CHANGELOG.md).

The public thesis is:

> ProgramGrad is a research-oriented framework for differentiable program
> traces. It studies how hard algorithmic decisions can be paired with soft
> surrogate traces, semantic gradient contracts, and fidelity metrics,
> especially for reasoning and search algorithms.

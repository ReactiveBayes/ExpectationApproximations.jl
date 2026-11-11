# ExpectationApproximations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://reactivebayes.github.io/ExpectationApproximations.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://reactivebayes.github.io/ExpectationApproximations.jl/dev)
[![Build Status](https://github.com/ReactiveBayes/ExpectationApproximations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ReactiveBayes/ExpectationApproximations.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ReactiveBayes/ExpectationApproximations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ReactiveBayes/ExpectationApproximations.jl)
[![Code style: Runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

Numerical expectations and Gaussian approximations for approximate Bayesian inference in Julia.

- **Expectations** `E_q[f(x)]` by quadrature, cubature, sigma points, exact sums or sampling, as
  backends of the expectation protocol of [BayesBase](https://github.com/ReactiveBayes/BayesBase.jl).
  [`EADefault()`](https://reactivebayes.github.io/ExpectationApproximations.jl/stable/routing/)
  picks the rule made for each family: Gauss–Hermite for a Gaussian, Gauss–Laguerre for a Gamma,
  Gauss–Jacobi for a Beta, an exact sum for a categorical. Prepared expectations allocate nothing
  and pass ForwardDiff's dual numbers through.
- **Gaussian approximations** of a density given as a Julia function: Laplace's, the
  moment-matched Gaussian of expectation propagation, and the variational Gaussian, each with an
  approximation of the log-normaliser.
- **Constrained variables**: positive, bounded, probability or simplex-valued variables are
  approximated by a Gaussian in unconstrained coordinates, read from their prior's support.

```julia
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations

BayesBase.expectation(EA.EADefault(), log, Gamma(3.0, 2.0))    # E[log x], ≈ ψ(3) + log 2

EA.laplace(x -> logpdf(Gamma(3.0, 2.0), x), 1.0, EA.Positive()) # a log-normal approximation
EA.moment_matching(x -> x - exp(x) / 2, 0.0)                     # a Gaussian with its moments
```

The [documentation](https://reactivebayes.github.io/ExpectationApproximations.jl/stable) explains
the protocol, how to choose a backend and when each approximation fits.

ExpectationApproximations is part of the [ReactiveBayes](https://github.com/ReactiveBayes)
ecosystem. [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) uses it, through
[DeltaMessagePassingRules](https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl), to run
inference on models without hand-written message passing rules. Nothing in this package depends
on message passing, though: it works on any distribution and any Julia function.

## Installation

```julia
import Pkg
Pkg.add("ExpectationApproximations")
```

Julia 1.10 or later.

## Development

`make test` runs the test suite as CI does; `make test test_args="tag:quality"` selects items by
tag, by name or by file. `make docs` builds the documentation, and `make help` lists the other
targets.

## License

[MIT License](LICENSE). Copyright (c) 2022-2024 BIASlab, 2024-present ReactiveBayes.

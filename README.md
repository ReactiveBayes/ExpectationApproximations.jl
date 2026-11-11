# ExpectationApproximations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://bvdmitri.github.io/ExpectationApproximations.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bvdmitri.github.io/ExpectationApproximations.jl/dev)
<!-- [![Build Status](https://github.com/reactivebayes/ExpectationsApproximations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/reactivebayes/ExpectationsApproximations.jl/actions/workflows/CI.yml?query=branch%3Amain) -->
[![Coverage](https://codecov.io/gh/reactivebayes/ExpectationApproximations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/reactivebayes/ExpectationApproximations.jl)
<!-- [![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/E/ExpectationApproximations.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html) -->
[![Code style: Runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

Numerical backends for the expectation protocol of
[BayesBase](https://github.com/ReactiveBayes/BayesBase.jl): quadrature, cubature, sigma points
and sampling compute `E_q[f]` through `BayesBase.expectation`, alongside the local Gaussian
projections that approximate inference builds on (Laplace, moment matching, linearisation) and
transforms to unconstrained coordinates. Version 1.0 is a rewrite; see the
[CHANGELOG](CHANGELOG.md).

- Tests: `make test` runs the suite as CI does; `make test test_args="tag:quality"` selects
  items by tag, by name or by file. `make help` lists the other targets.
- Julia 1.10 or later. MIT licence.

[MIT License](LICENSE) Copyright (c) 2022-2024 BIASlab, 2024-present ReactiveBayes

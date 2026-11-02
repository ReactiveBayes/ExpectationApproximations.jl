# Changelog

All notable changes to ExpectationApproximations.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Gaussian quadrature rules by the Golub–Welsch algorithm, without FastGaussQuadrature:
  `gausshermite`, `gausslaguerre` (generalised), `gaussjacobi` and `gausslegendre`.

### Changed

- ExpectationApproximations 1.0 is a rewrite: numerical backends for the expectation protocol of
  BayesBase 1.7. The previous API (`approximate`, the kernels, CVI, CVMP, AdvancedHMC sampling,
  the constrained unscented transform) is removed.

- The code is formatted with [Runic](https://github.com/fredrikekre/Runic.jl) instead of
  JuliaFormatter: `make format` and `make check-format`, and the `Format` workflow in CI.
- The tooling follows the ReactiveBayes PackageTemplate: the Makefile, the CI, CompatHelper,
  TagBot and documentation preview workflows, and the documentation setup (Documenter 1).

[Unreleased]: https://github.com/ReactiveBayes/ExpectationApproximations.jl/commits/main

# Changelog

All notable changes to ExpectationApproximations.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Gaussian quadrature rules by the Golub–Welsch algorithm, without FastGaussQuadrature:
  `gausshermite`, `gausslaguerre` (generalised), `gaussjacobi` and `gausslegendre`.
- Backends for the expectation protocol of BayesBase, each preparing a point set: `GaussHermite`
  (Gaussians, and log-normal and logit-normal distributions through their Gaussian),
  `GaussLaguerre` (Gamma and Exponential, any shape), `GaussJacobi` (Beta), `QuantileLegendre`
  (any continuous univariate distribution), `ExactSum` (finite discrete distributions, exact) and
  `MonteCarlo`.
- `SNIS`, self-normalised importance sampling for targets that can be evaluated but not sampled,
  and `effective_sample_size`.
- Sigma points: `Unscented` (with the centre's covariance weight applied by `mean_cov`),
  `GenUnscented` (moment-matched, exact up to degree 4 for any univariate distribution) and
  `SphericalRadialCubature`.
- `EADefault`, the default backend: a router that sends each family to its rule, and any other
  continuous univariate distribution to `QuantileLegendre`.
- `mean_cov` and `mean_cov_cross`, the moments of a function of a random variable and its
  cross-covariance with the input, and `tilted_mean_cov`, the log-normaliser and moments of a
  tilted distribution, computed in log space.
- `laplace`, the Laplace approximation of a density with its log-normaliser: Newton's method with
  a backtracking line search that stays in the domain, and an eigenvalue correction where the
  log-density is not concave; for numbers, vectors and static vectors.
- `linearize`, the first-order expansion of a function of one or several inputs.
- Domain transforms: `RealLine`, `Positive`, `UnitInterval`, `LowerBounded`, `UpperBounded`,
  `Bounded`, `Simplex` and `Componentwise` for blocks of components, with `forward`, `inverse`,
  `logjacobian`, `domain` (read from a distribution's support, the domain of BayesBase's
  log-densities, or a product's first factor that has one) and `pushforward` (a `LogNormal` or a
  `TransformedGaussian`, which has an `entropy`); and `laplace` in a domain.
- `ExpectationApproximations.Testing`, the conformance suite every backend passes: correctness,
  exact labels, prepared against direct, reproducibility, type stability, no allocations, dual
  numbers and products.
- `ExpectationApproximations.MPRACompat`, the API of MessagePassingRulesApproximations 1.0
  (`Unscented`, `Linearization`, `GaussHermiteCubature`, `approximate`, `unscented_statistics`,
  `approximate_meancov`, `smoothRTS` and the rest), for its 1.1 to re-export; tested against 1.0
  itself, with Gauss–Hermite weights from this package's more accurate rule.
- Benchmarks with PkgBenchmark: `make benchmark` and `make benchmark-compare branch=main`.
- An extension for ExponentialFamily: its univariate Gaussians through `GaussHermite` and
  `GammaShapeRate` through `GaussLaguerre`; its multivariate Gaussians, with static points for a
  static mean, are covered by the core.

### Changed

- The backends implement BayesBase's expectation protocol under its final names, each saying it
  is about expectations: `supports_expectation`, `ExpectationPointSet`, `ExpectationFirstOf`,
  `with_expectation_rng`, `ExpectationExact` and the rest (PLAN.md, D36).
- ExpectationApproximations 1.0 is a rewrite: numerical backends for the expectation protocol of
  BayesBase 1.7. The previous API (`approximate`, the kernels, CVI, CVMP, AdvancedHMC sampling,
  the constrained unscented transform) is removed.

- The code is formatted with [Runic](https://github.com/fredrikekre/Runic.jl) instead of
  JuliaFormatter: `make format` and `make check-format`, and the `Format` workflow in CI.
- The tooling follows the ReactiveBayes PackageTemplate: the Makefile, the CI, CompatHelper,
  TagBot and documentation preview workflows, and the documentation setup (Documenter 1).

[Unreleased]: https://github.com/ReactiveBayes/ExpectationApproximations.jl/commits/main

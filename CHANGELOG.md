# Changelog

All notable changes to ExpectationApproximations.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

ExpectationApproximations 1.0 is a rewrite. It implements numerical backends for the expectation
protocol of BayesBase 1.7, and Gaussian approximations of a density.

### Added

- Gaussian quadrature rules computed by the Golub–Welsch algorithm, without FastGaussQuadrature:
  `gausshermite`, `gausslaguerre` (generalised), `gaussjacobi` and `gausslegendre`, with weights
  accurate to about `1e-13` of their size.
- Backends for the expectation protocol of BayesBase. Each prepares a point set, so prepared
  expectations do not allocate and accept dual numbers:
  - `GaussHermite` for Gaussians, and for log-normal and logit-normal distributions through their
    Gaussian;
  - `GaussLaguerre` for Gamma and Exponential distributions of any shape, and `GaussJacobi` for
    Beta distributions;
  - `QuantileLegendre` for any continuous univariate distribution;
  - `ExactSum` for finite discrete distributions, labelled exact;
  - `MonteCarlo`, and `SNIS` (self-normalised importance sampling) for targets that can be
    evaluated but not sampled, with `effective_sample_size`; both are labelled stochastic;
  - sigma-point rules: `Unscented`, prepared as a `SigmaPointSet` that stores the covariance
    weight of the centre point; `GenUnscented`, exact up to degree 4 for univariate distributions
    with a skewness and a kurtosis; and `SphericalRadialCubature`.
- `EADefault`, the default backend: a router that sends each family to its rule, and other
  continuous univariate distributions to `QuantileLegendre`.
- `register_backends!`, called when the package loads. It adds the rules of `EADefault` to the
  registry of backends in BayesBase, `QuantileLegendre` with the lowest priority, so a frozen
  `ExpectationDynamicRouter` chooses as `EADefault` does. The stochastic backends are not
  registered. Rules of the same kind and size compare equal, so registering again changes nothing.
- `mean_cov` and `mean_cov_cross`, the moments of a function of a random variable and its
  cross-covariance with the input, and `tilted_mean_cov`, the log-normaliser and moments of a
  tilted distribution, computed in log space.
- `linearize`, the first-order expansion of a function of one or several inputs.
- Gaussian approximations of a density given as a Julia function, for numbers, vectors and static
  vectors, each with an approximation of the log-normaliser:
  - `laplace`, by Newton's method with a backtracking line search that stays where the
    log-density is finite, and an eigenvalue correction where it is not concave;
  - `moment_matching`, the projection of expectation propagation. It places the quadrature points
    on the Laplace approximation of the density and weights them against it, so it resolves a
    factor much sharper than its cavity, which points placed on the cavity miss;
  - `gaussian_variational`, the Gaussian that minimises `KL(q ‖ p)`, with its evidence lower
    bound. Newton's method maximises the bound over the mean and the Cholesky factor. The result
    is the fixed point of conjugate-computation variational inference and of natural-gradient
    message passing.
- Domains for constrained variables: `RealLine`, `Positive`, `UnitInterval`, `LowerBounded`,
  `UpperBounded`, `Bounded`, `Simplex` and `Componentwise`. `forward`, `inverse` and
  `logjacobian` compute the transform, and `domain` reads the domain from the support of a
  distribution. On any domain but the real line, `pushforward` returns a
  `TransformedGaussian`, which has a `logpdf`, sampling, a mean, a variance and an entropy. `laplace`, `moment_matching` and `gaussian_variational` take a
  domain and approximate in unconstrained coordinates.
- `ExpectationApproximations.Testing`, the conformance suite that the backends of this package
  pass, for testing backends in other packages too.
- `ExpectationApproximations.MPRACompat`, the API of MessagePassingRulesApproximations 1.0, for
  its version 1.1 to re-export. It is tested against 1.0; the Gauss–Hermite weights come from the
  more accurate rule of this package.
- An extension for ExponentialFamily: its univariate Gaussians through `GaussHermite`,
  `Unscented`, `GenUnscented` and `SphericalRadialCubature`, and `GammaShapeRate` through
  `GaussLaguerre`.
- Documentation: a manual with tutorials on computing expectations, choosing a backend, the
  Gaussian approximations, constrained variables and writing a backend.
- Benchmarks with PkgBenchmark: `make benchmark` and `make benchmark-compare branch=main`.

### Changed

- The code is formatted with [Runic](https://github.com/fredrikekre/Runic.jl) instead of
  JuliaFormatter: `make format` and `make check-format`, and the `Format` workflow in CI.
- The tooling follows the ReactiveBayes PackageTemplate: the Makefile, the CI, CompatHelper,
  TagBot and documentation preview workflows, and the documentation setup (Documenter 1).

### Removed

- The previous API: `approximate` and its kernels, CVI, CVMP, sampling with AdvancedHMC and the
  constrained unscented transform, and with them the dependencies on AdvancedHMC, Optim,
  Optimisers, TensorOperations, FastGaussQuadrature and others.

[Unreleased]: https://github.com/ReactiveBayes/ExpectationApproximations.jl/commits/main

```@meta
CurrentModule = ExpectationApproximations
```

# Choosing a backend

The backends differ in accuracy, in cost and in the distributions they support.
[`EADefault`](@ref) chooses among some of them automatically. The others, such as sampling and
sigma points, are chosen explicitly.

```@example backends
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations
using SpecialFunctions: digamma
using Random: Xoshiro
nothing # hide
```

## Overview

| Backend | Distributions | Points | Exact for | Label |
|:--------|:--------------|:-------|:----------|:------|
| [`GaussHermite`](@ref)`(n)` | `Normal`, `MvNormal`, `LogNormal`, `LogitNormal`, [`TransformedGaussian`](@ref) | ``n^d`` | polynomials of degree ``\le 2n - 1`` in the Gaussian variable | none |
| [`GaussLaguerre`](@ref)`(n)` | `Gamma`, `Exponential`, any shape | ``n`` | polynomials of degree ``\le 2n - 1`` | none |
| [`GaussJacobi`](@ref)`(n)` | `Beta` | ``n`` | polynomials of degree ``\le 2n - 1`` | none |
| [`QuantileLegendre`](@ref)`(n)` | any continuous univariate | ``n`` | nothing in general: errors decrease algebraically in ``n`` | none |
| [`ExactSum`](@ref)`()` | finite discrete: `Categorical`, `Bernoulli`, `Binomial`, … | the support | everything | exact |
| [`MonteCarlo`](@ref)`(n, rng)` | anything that can be sampled | ``n`` | nothing | stochastic |
| [`SNIS`](@ref)`(proposal, n, rng)` | anything with a log-density | ``n`` | nothing | stochastic |
| [`Unscented`](@ref)`()` | Gaussians | ``2d + 1`` | polynomials of degree ``\le 3`` | none |
| [`GenUnscented`](@ref)`()` | univariate with a skewness and kurtosis; Gaussian vectors | ``2d + 1`` | polynomials of degree ``\le 4`` | none |
| [`SphericalRadialCubature`](@ref)`()` | Gaussians | ``2d`` | polynomials of degree ``\le 3`` | none |

All of them prepare a point set, do not allocate when evaluated, and pass dual numbers through
the integrand.

Gauss–Hermite quadrature on a ``d``-dimensional Gaussian is a tensor grid of ``n^d`` points: with
20 points per dimension, that is 400 points in two dimensions and 8000 in three. The number of
sigma points grows linearly in ``d``, but sigma points are exact only up to degree 3 or 4.

The nodes of Gauss–Laguerre and Gauss–Jacobi quadrature depend on the shape of the Gamma or Beta,
so ForwardDiff cannot differentiate through a shape parameter. It can differentiate through the
scale of a Gamma.

## Accuracy: an example

Take ``x \sim \mathrm{Gamma}(2.5, 1)`` and two integrands with closed forms: the smooth
``e^{-x}``, with ``\mathbb{E}[e^{-x}] = 2^{-2.5}``, and ``\log x``, with
``\mathbb{E}[\log x] = \psi(2.5)``, which is singular at zero.

```@example backends
q = Gamma(2.5, 1.0)
deviation(backend, f, exact) = abs(BayesBase.expectation(backend, f, q) - exact)
smooth(backend) = deviation(backend, x -> exp(-x), 2.0^-2.5)
singular(backend) = deviation(backend, log, digamma(2.5))
for (name, backend) in (
        "GaussLaguerre(16)" => EA.GaussLaguerre(16),
        "GaussLaguerre(64)" => EA.GaussLaguerre(64),
        "QuantileLegendre(16)" => EA.QuantileLegendre(16),
        "QuantileLegendre(64)" => EA.QuantileLegendre(64),
        "MonteCarlo(10_000)" => EA.MonteCarlo(10_000, Xoshiro(1)),
    )
    println(rpad(name, 22), "smooth error ", round(smooth(backend); sigdigits = 2), "   singular error ", round(singular(backend); sigdigits = 2))
end
```

For the smooth integrand, Gauss–Laguerre quadrature, the rule for the Gamma family, converges
much faster than the quantile rule, whose error decreases only algebraically.

The singularity of the logarithm at zero slows both rules down, to an error near ``10^{-3}`` at
16 points. Where a closed form exists, it is the better choice: a router can put a closed-form
backend first, and an integrand that has a closed form is then not integrated numerically.

Both quadratures are more accurate than sampling by orders of magnitude, with far fewer points.
Sampling, however, works for any distribution that can be sampled, in any dimension.

## Quadrature for families

```@docs
GaussHermite
GaussLaguerre
GaussJacobi
QuantileLegendre
ExactSum
```

## Sampling

Sampling makes no assumption about smoothness and works in any dimension, but its error
decreases only as ``1/\sqrt{n}`` and its results are random. Use it for a distribution that no
quadrature supports, or as a reference.

```@docs
MonteCarlo
SNIS
effective_sample_size
```

## Sigma points

Sigma-point rules use a few deterministic points, chosen to match the first moments of the
distribution. They suit smooth functions of Gaussians in several dimensions, as in nonlinear
filtering. For a linear function they are exact:

```@example backends
A, b = [1.0 2.0; 0.0 1.0], [0.5, -1.0]
m, V = EA.mean_cov(EA.Unscented(), x -> A * x + b, MvNormal([1.0, 2.0], [1.0 0.3; 0.3 2.0]))
m ≈ A * [1.0, 2.0] + b, V ≈ A * [1.0 0.3; 0.3 2.0] * A'
```

The centre point of the unscented transform has a covariance weight that differs from its mean
weight, so its prepared form is a [`SigmaPointSet`](@ref), which stores both. The centre weight
can be negative, and it is with the default parameters, so the unscented transform does not
support products of distributions or tilted moments. The spherical-radial cubature has equal
positive weights and neither restriction.

```@docs
Unscented
SigmaPointSet
GenUnscented
SphericalRadialCubature
```

## ExponentialFamily's distributions

When [ExponentialFamily](https://github.com/ReactiveBayes/ExponentialFamily.jl) is loaded, an
extension adds support for its distribution types:

- its univariate Gaussians, `NormalMeanVariance`, `NormalMeanPrecision` and
  `NormalWeightedMeanPrecision`, to [`GaussHermite`](@ref), [`Unscented`](@ref),
  [`GenUnscented`](@ref) and [`SphericalRadialCubature`](@ref);
- `GammaShapeRate` to [`GaussLaguerre`](@ref).

Its multivariate Gaussians are subtypes of Distributions' `AbstractMvNormal`, which the package
supports without the extension. With the extension, [`EADefault`](@ref) handles the distributions
that the rules of RxInfer produce.

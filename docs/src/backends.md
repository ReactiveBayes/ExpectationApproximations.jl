```@meta
CurrentModule = ExpectationApproximations
```

# Choosing a backend

Every backend trades accuracy, cost and generality. This page compares them, so that you can
pick one when [`EADefault`](@ref) is not what you need.

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

All of them prepare a point set, allocate nothing when evaluated, and pass dual numbers through
the integrand. Two limits matter in practice:

- **Dimension.** Gauss–Hermite on a ``d``-dimensional Gaussian is a tensor grid of ``n^d``
  points: 20 points per dimension is 400 points in two dimensions and 8000 in three. Sigma
  points grow linearly in ``d``, at the price of exactness only up to degree 3 or 4.
- **Derivatives through shapes.** The nodes of Gauss–Laguerre and Gauss–Jacobi depend on the
  shape of the Gamma or Beta, so ForwardDiff cannot differentiate through a shape parameter. It
  can differentiate through the scale of a Gamma.

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

Three lessons:

- for a smooth integrand, the rule matched to the family, Gauss–Laguerre for a Gamma, converges
  far faster than the generic quantile rule, whose error decreases only algebraically;
- a singularity, such as the logarithm's at zero, slows every rule down, matched or not, to an
  error near ``10^{-3}`` at 16 points: where a closed form exists, it should win. A router can put
  a closed-form backend first, and an integrand that has one is then never integrated
  numerically;
- every quadrature beats sampling by orders of magnitude at a fraction of the points, but sampling
  is the only one that works for any distribution in any dimension.

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
decreases only as ``1/\sqrt{n}`` and its results are random. Use it for a distribution no
quadrature covers, or as a reference.

```@docs
MonteCarlo
SNIS
effective_sample_size
```

## Sigma points

Sigma points are few and deterministic: they match the first moments of a distribution rather than
integrate it, so they suit smooth functions of Gaussians in several dimensions, as in nonlinear
filtering. For a linear function they are exact:

```@example backends
A, b = [1.0 2.0; 0.0 1.0], [0.5, -1.0]
m, V = EA.mean_cov(EA.Unscented(), x -> A * x + b, MvNormal([1.0, 2.0], [1.0 0.3; 0.3 2.0]))
m ≈ A * [1.0, 2.0] + b, V ≈ A * [1.0 0.3; 0.3 2.0] * A'
```

The unscented transform's centre point has a covariance weight that differs from its mean weight.
Its prepared form is therefore a [`SigmaPointSet`](@ref) rather than a plain point set, and its
negative centre weight rules out products of distributions and tilted moments. The spherical-radial
cubature has equal positive weights and none of these limits.

```@docs
Unscented
SigmaPointSet
GenUnscented
SphericalRadialCubature
```

## ExponentialFamily's distributions

When [ExponentialFamily](https://github.com/ReactiveBayes/ExponentialFamily.jl) is loaded, an
extension adds its own types:

- its univariate Gaussians, `NormalMeanVariance`, `NormalMeanPrecision` and
  `NormalWeightedMeanPrecision`, to [`GaussHermite`](@ref), [`Unscented`](@ref),
  [`GenUnscented`](@ref) and [`SphericalRadialCubature`](@ref);
- `GammaShapeRate` to [`GaussLaguerre`](@ref).

Its multivariate Gaussians are Distributions' `AbstractMvNormal`s, which the package covers
without the extension. So [`EADefault`](@ref) handles the distributions RxInfer's rules produce.

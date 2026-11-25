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
| [`GaussHermite`](@ref)`(n)` | `Normal`, `MvNormal`, `LogNormal`, `LogitNormal`, [`TransformedGaussian`](@ref) | ``n^d``, at most 10 000 | polynomials of degree ``\le 2n - 1`` in the Gaussian variable | none |
| [`UnconstrainedLegendre`](@ref)`(n)` | `Gamma`, `Exponential`, `Chisq`, `Erlang`, `InverseGamma`, `Weibull`, `Beta` | about ``14n`` to ``26n`` | nothing, but accurate for integrands singular at the boundary | none |
| [`GaussLaguerre`](@ref)`(n)` | `Gamma`, `Exponential`, any shape | ``n`` | polynomials of degree ``\le 2n - 1`` | none |
| [`GaussJacobi`](@ref)`(n)` | `Beta` | ``n`` | polynomials of degree ``\le 2n - 1`` | none |
| [`QuantileLegendre`](@ref)`(n)` | any continuous univariate | ``n`` | nothing in general: errors decrease algebraically in ``n`` | none |
| [`ExactSum`](@ref)`()` | finite discrete: `Categorical`, `Bernoulli`, `Binomial`, … | the support | everything | exact |
| [`MonteCarlo`](@ref)`(n, rng)` | anything that can be sampled | ``n`` | nothing | stochastic |
| [`SNIS`](@ref)`(proposal, n, rng)` | anything with a log-density | ``n`` | nothing | stochastic |
| [`Unscented`](@ref)`()` | Gaussians | ``2d + 1`` | polynomials of degree ``\le 3`` | none |
| [`GenUnscented`](@ref)`()` | common univariate families with finite moments; Gaussian vectors | ``2d + 1`` | polynomials of degree ``\le 4`` in one variable, ``\le 3`` in several | none |
| [`SphericalRadialCubature`](@ref)`()` | Gaussians | ``2d`` | polynomials of degree ``\le 3`` | none |

All of them prepare a point set, do not allocate when evaluated, and pass dual numbers through
the integrand.

Gauss–Hermite quadrature on a ``d``-dimensional Gaussian is a tensor grid of ``n^d`` points: with
20 points per dimension, that is 400 points in two dimensions and 8000 in three. The grid is
capped at `maxpoints`, 10 000 by default: beyond it the rule has fewer points per dimension, and
from nine dimensions it is the spherical-radial rule. The number of sigma points grows linearly in
``d``, but sigma points are exact only up to degree 3 or 4.

The nodes of Gauss–Laguerre and Gauss–Jacobi quadrature, and the panels of the rule in
unconstrained coordinates, depend on the shape of the Gamma or Beta, so ForwardDiff cannot
differentiate through a shape parameter. It can differentiate through the scale of a Gamma.

## Accuracy: an example

Take ``x \sim \mathrm{Gamma}(2.5, 1)``, and then ``\mathrm{Gamma}(0.3, 1)``, and two integrands with closed forms: the smooth
``e^{-x}``, with ``\mathbb{E}[e^{-x}] = 2^{-2.5}``, and ``\log x``, with
``\mathbb{E}[\log x] = \psi(2.5)``, which is singular at zero.

```@example backends
deviation(backend, f, q, exact) = abs(BayesBase.expectation(backend, f, q) - exact)
smooth(backend, a) = deviation(backend, x -> exp(-x), Gamma(a, 1.0), 2.0^-a)
singular(backend, a) = deviation(backend, log, Gamma(a, 1.0), digamma(a))
for a in (2.5, 0.3), (name, backend) in (
        "UnconstrainedLegendre()" => EA.UnconstrainedLegendre(),
        "GaussLaguerre(16)" => EA.GaussLaguerre(16),
        "GaussLaguerre(64)" => EA.GaussLaguerre(64),
        "QuantileLegendre(16)" => EA.QuantileLegendre(16),
        "QuantileLegendre(64)" => EA.QuantileLegendre(64),
        "MonteCarlo(10_000)" => EA.MonteCarlo(10_000, Xoshiro(1)),
    )
    println("shape ", a, "  ", rpad(name, 24), "smooth error ", round(smooth(backend, a); sigdigits = 2), "   singular error ", round(singular(backend, a); sigdigits = 2))
end
```

For the smooth integrand, Gauss–Laguerre quadrature converges much faster than the quantile rule,
whose error decreases only algebraically.

The logarithm is singular at zero, and its expectation is where Gauss–Laguerre and the quantile
rule fall short: both are off by about ``10^{-3}`` at 16 points for a shape of 2.5, and at a shape
of 0.3, where the mass piles up at zero, Gauss–Laguerre is off by about 1. In the logarithm of ``x`` the integrand is smooth, so
[`UnconstrainedLegendre`](@ref), the rule of [`EADefault`](@ref) for these families, is accurate
to rounding for both integrands at both shapes. Such integrands are the usual ones in inference:
the expected logarithm of a rate or a probability, and the log-density of a count or a binary
observation. Where a closed form exists, it is still the better choice: a router can put a
closed-form backend first, and an integrand that has a closed form is then not integrated
numerically.

Both quadratures are more accurate than sampling by orders of magnitude, with far fewer points.
Sampling, however, works for any distribution that can be sampled, in any dimension.

## Quadrature for families

```@docs
GaussHermite
UnconstrainedLegendre
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
- `GammaShapeRate` to [`UnconstrainedLegendre`](@ref) and [`GaussLaguerre`](@ref).

Its multivariate Gaussians are subtypes of Distributions' `AbstractMvNormal`, which the package
supports without the extension. With the extension, [`EADefault`](@ref) handles the distributions
that the rules of RxInfer produce.

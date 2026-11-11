```@meta
CurrentModule = ExpectationApproximations
```

# Moments through functions

Message passing with Gaussians keeps asking one question: if ``x`` is Gaussian and ``y = f(x)``,
what are the mean and covariance of ``y``, and how does ``y`` covary with ``x``? The answers give a
Gaussian message through a nonlinear node, as in the extended, unscented and cubature Kalman
filters. This page covers the functions that compute them through any backend.

```@example moments
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations
nothing # hide
```

## The moments of ``f(x)``

[`mean_cov`](@ref)`(backend, f, q)` returns ``m = \mathbb{E}[f(x)]`` and
``V = \mathbb{E}[(f(x) - m)(f(x) - m)^\top]`` for ``x \sim q``. Unlike `BayesBase.mean_cov(q)`, the
moments of the distribution, these are the moments of a function of it. For a polar-to-Cartesian
map, a classic test of nonlinear filters:

```@example moments
polar = MvNormal([1.0, π / 4], [0.01 0.0; 0.0 0.2])
tocartesian(x) = [x[1] * cos(x[2]), x[1] * sin(x[2])]
m, V = EA.mean_cov(EA.GaussHermite(10), tocartesian, polar)
```

The same call with [`SphericalRadialCubature`](@ref)`()` uses 4 points instead of 100, and
[`Unscented`](@ref)`()` uses 5:

```@example moments
EA.mean_cov(EA.SphericalRadialCubature(), tocartesian, polar)[1], EA.mean_cov(EA.Unscented(), tocartesian, polar)[1]
```

[`mean_cov_cross`](@ref) adds the cross-covariance ``C = \mathbb{E}[(x - \mathbb{E}x)(f(x) - m)^\top]``,
which a Kalman update needs to condition ``x`` on an observation of ``y``:

```@example moments
m, V, C = EA.mean_cov_cross(EA.GaussHermite(10), tocartesian, polar)
C
```

## Tilted moments

Expectation propagation needs the moments of a Gaussian *tilted* by a factor,
``q(x) e^{\log f(x)} / Z``. [`tilted_mean_cov`](@ref)`(backend, logf, q)` returns
``\log Z``, the mean and the covariance, computed in log space so that very small or very large
factors neither underflow nor overflow. For a Gaussian likelihood the result is close to the
conjugate posterior; not exact, since the factor is not a polynomial in ``x``:

```@example moments
prior = Normal(0.0, 2.0)
loglikelihood(x) = logpdf(Normal(x, 1.0), 1.5)
logZ, m, v = EA.tilted_mean_cov(EA.GaussHermite(20), loglikelihood, prior)
(logZ = logZ, m = m, v = v, exact_m = 1.5 * 4 / 5, exact_v = 4 / 5)
```

Its points come from ``q``, so it suits a factor no sharper than ``q``. For a sharp factor, use
[`moment_matching`](@ref), which places the points on the tilted density itself. See
[Gaussian approximations of a density](@ref).

## Linearisation

[`linearize`](@ref)`(f, x)` returns ``A`` and ``b`` with ``f(x') \approx A x' + b`` near ``x``, the
first-order expansion behind the extended Kalman filter. It takes several inputs too, and stacks
them:

```@example moments
A, b = EA.linearize((x, y) -> x * y + sin(x), 1.0, 2.0)
```

Linearisation needs no quadrature and is exact for linear functions, but it ignores the curvature
of ``f`` across the spread of ``x``. The moment-based functions above take it into account.

```@docs
mean_cov
mean_cov_cross
tilted_mean_cov
linearize
```

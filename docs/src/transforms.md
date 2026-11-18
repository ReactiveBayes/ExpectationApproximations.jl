```@meta
CurrentModule = ExpectationApproximations
```

# Constrained variables

A Gaussian has the whole real line as its support, but many variables are constrained: a
precision is positive, a probability lies between zero and one, and mixture weights sum to one. A
Gaussian placed directly on such a variable puts mass on values the variable cannot take. Its
Laplace approximation can even be centred at a mode on the boundary, as for a Beta density that is
infinite at zero.

The usual remedy is to approximate in *unconstrained coordinates*. Write ``x = g(u)`` for a
smooth bijection ``g`` from the real line onto the domain, approximate the density of ``u`` by a
Gaussian, and push it forward through ``g``. The density of ``u`` gains the log-Jacobian of
``g``:

```math
\log p_u(u) = \log p_x(g(u)) + \log \left|\det \frac{\partial g}{\partial u}\right|.
```

```@example transforms
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations
nothing # hide
```

## Domains

| Domain | Values | ``x = g(u)`` | A Gaussian in ``u`` becomes |
|:-------|:-------|:-------------|:----------------------------|
| [`RealLine`](@ref)`()` | ``\mathbb{R}`` or ``\mathbb{R}^d`` | ``u`` | the Gaussian |
| [`Positive`](@ref)`()` | ``x > 0`` | ``e^u`` | a log-normal |
| [`UnitInterval`](@ref)`()` | ``0 < x < 1`` | ``\sigma(u)``, the logistic function | a logit-normal |
| [`LowerBounded`](@ref)`(a)` | ``x > a`` | ``a + e^u`` | a shifted log-normal |
| [`UpperBounded`](@ref)`(b)` | ``x < b`` | ``b - e^u`` | a reflected log-normal |
| [`Bounded`](@ref)`(a, b)` | ``a < x < b`` | ``a + (b - a)\,\sigma(u)`` | a scaled logit-normal |
| [`Simplex`](@ref)`(K)` | probability vectors of length ``K`` | additive log-ratio from ``K - 1`` coordinates | a logistic-normal |
| [`Componentwise`](@ref)`(d₁, d₂, …)` | a vector with one domain per component | each component separately | a joint distribution over the block |

[`forward`](@ref), [`inverse`](@ref) and [`logjacobian`](@ref) compute ``g``, ``g^{-1}`` and the
log-Jacobian.

## The domain of a distribution

[`domain`](@ref) reads the domain from the support of a distribution, usually the prior of the
variable. A model can then keep priors such as a Gamma, a Beta or a Dirichlet, and the transform is
used only in the approximation:

```@example transforms
EA.domain(Gamma(2.0, 1.0)), EA.domain(Beta(1.0, 1.0)), EA.domain(Uniform(-1.0, 3.0)), EA.domain(Dirichlet(3, 1.0))
```

## Approximations in a domain

[`laplace`](@ref), [`moment_matching`](@ref) and [`gaussian_variational`](@ref) take a domain as
their third argument. Take a Gamma posterior for a rate, ``\mathrm{Gamma}(23, 1/21)``:

```@example transforms
a, b = 23.0, 21.0
result = EA.laplace(x -> logpdf(Gamma(a, 1 / b), x), 1.0, EA.Positive())
result.distribution
```

The result is a [`TransformedGaussian`](@ref): a Gaussian in ``u = \log x``, here
``\mathcal{N}(\log(a/b), 1/a)``, pushed forward onto the positive half-line. It has a `logpdf`
and can be sampled. Its mean, variance and entropy are computed by quadrature over its Gaussian:

```@example transforms
mean(result.distribution), mean(Gamma(a, 1 / b))
```

[`GaussHermite`](@ref), and so [`EADefault`](@ref), computes expectations under it through its
Gaussian:

```@example transforms
BayesBase.expectation(EA.GaussHermite(16), log, result.distribution)
```

In a domain, `logZ` approximates the log-normaliser in the *original* coordinates. The
log-Jacobian accounts for the change of variables, so for a normalised density `logZ` is close to
zero. The `mode` field of `laplace` and the `mean` field of the other two are the images
``g(\hat u)`` of the unconstrained mode or mean. They are not the mode or mean of the returned
distribution, because a nonlinear transform moves both:

```@example transforms
result.logZ, result.mode, a / b
```

A density that is infinite at the boundary, such as ``\mathrm{Beta}(0.8, 3)`` at zero, has no mode
inside the unit interval, but in logit coordinates it has one:

```@example transforms
beta = EA.laplace(x -> logpdf(Beta(0.8, 3.0), x), 0.5, EA.UnitInterval())
beta.converged, mean(beta.distribution)
```

A block of parameters in different domains, such as a probability and a rate that are correlated
in the posterior, has a [`Componentwise`](@ref) domain. It is approximated jointly by one Gaussian
in two unconstrained coordinates.

```@docs
Domain
RealLine
Positive
UnitInterval
LowerBounded
UpperBounded
Bounded
Simplex
Componentwise
domain
forward
inverse
logjacobian
pushforward
TransformedGaussian
```

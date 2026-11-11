```@meta
CurrentModule = ExpectationApproximations
```

# Computing expectations

This page walks through the expectation protocol of BayesBase as this package implements it. The
protocol has two entry points:

- `BayesBase.expectation(backend, f, q)` computes ``\mathbb{E}_q[f(x)]`` in one call;
- `BayesBase.prepare_expectation(backend, q)` does the work that depends only on ``q`` once,
  and `BayesBase.expectation(prepared, f)` then evaluates any number of functions.

```@example tutorial
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations
nothing # hide
```

## One expectation

A backend is a small value that says how to compute: [`GaussHermite`](@ref)`(n)` is Gauss–Hermite
quadrature with `n` points, for Gaussians. For ``x \sim \mathcal{N}(1, 2^2)``,
``\mathbb{E}[x^2] = 1 + 4 = 5``:

```@example tutorial
BayesBase.expectation(EA.GaussHermite(8), x -> x^2, Normal(1.0, 2.0))
```

Gauss–Hermite with `n` points is exact for polynomials up to degree ``2n - 1``. For other
functions it is an approximation that improves quickly with `n` when the function is smooth:

```@example tutorial
# E[cos(x)] under N(0, 1) is exp(-1/2)
[BayesBase.expectation(EA.GaussHermite(n), cos, Normal(0.0, 1.0)) - exp(-1 / 2) for n in (2, 4, 8, 16)]
```

## Prepared expectations

Every backend in this package prepares a *point set* for a distribution: points ``x_i`` and
weights ``w_i`` with ``\mathbb{E}_q[f] \approx \sum_i w_i f(x_i)``. Preparing computes the
points once; evaluating is then a weighted sum.

```@example tutorial
prepared = BayesBase.prepare_expectation(EA.GaussHermite(16), Normal(0.5, 2.0))
BayesBase.expectation(prepared, sin), BayesBase.expectation(prepared, abs2)
```

Evaluating a prepared expectation allocates nothing, so it can sit in the inner loop of an
optimiser or of message passing:

```@example tutorial
allocations(prepared, f) = @allocated BayesBase.expectation(prepared, f)
allocations(prepared, sin) # the first call compiles
allocations(prepared, sin)
```

## Derivatives

The points are typed from the distribution's parameters, and the weighted sum is generic Julia
code, so ForwardDiff differentiates through both the integrand and the parameters. For
``x \sim \mathcal{N}(\mu, 1)``, ``\frac{d}{d\mu} \mathbb{E}[x^2] = 2\mu``:

```@example tutorial
using ForwardDiff
ForwardDiff.derivative(μ -> BayesBase.expectation(EA.GaussHermite(8), abs2, Normal(μ, 1.0)), 0.7)
```

This is what lets [`laplace`](@ref) and [`gaussian_variational`](@ref) optimise functions that
are themselves expectations. One limit: the nodes of Gauss–Laguerre and Gauss–Jacobi depend on
the shape of a Gamma or a Beta, so a derivative through a shape parameter throws an
`ArgumentError`. A derivative through the scale of a Gamma works.

## Products of distributions of different kinds

A joint distribution of independent components of different kinds, such as a categorical
assignment, a Gaussian vector and a Gamma precision, is a `BayesBase.FactorizedJoint`. A router
such as [`EADefault`](@ref) sends each component to its own rule and combines the point sets as a
product rule:

```@example tutorial
mixed = BayesBase.FactorizedJoint((Categorical([0.4, 0.6]), MvNormal([1.0, 2.0], [1.0 0.0; 0.0 2.0]), Gamma(2.0, 1.0)))
f((z, m, λ)) = λ * (3.0 - m[z])^2
BayesBase.expectation(EA.EADefault(), f, mixed)
```

The exact value is ``\mathbb{E}[\lambda] \sum_k \pi_k ((3 - \mu_k)^2 + \sigma_k^2) = 2 (0.4 \cdot 5 + 0.6 \cdot 3) = 7.6``:
the categorical component is summed exactly and the other two are integrated by their
quadratures. A product rule has as many points as the product of its components' point sets, so
it suits a few components, not hundreds.

## Random number generators

Sampling backends, [`MonteCarlo`](@ref) and [`SNIS`](@ref), need a random number generator. They
refuse to run without one, so that no result depends on a global state by accident:

```@example tutorial
using Random
BayesBase.expectation(EA.MonteCarlo(10_000, Xoshiro(1)), log, Gamma(3.0, 2.0))
```

A caller that owns the random state, such as an inference engine, provides it from above with
`BayesBase.with_expectation_rng`, which fills in a generator only where none was set:

```@example tutorial
backend = BayesBase.with_expectation_rng(EA.MonteCarlo(10_000), Xoshiro(2))
BayesBase.expectation(backend, log, Gamma(3.0, 2.0))
```

[`SNIS`](@ref), self-normalised importance sampling, computes expectations under a density that
can be evaluated but not sampled, such as the unnormalised product of two densities. Here,
``\mathcal{N}(0, 1) \times \mathcal{N}(1, 1) \propto \mathcal{N}(1/2, 1/2)``:

```@example tutorial
target = BayesBase.ProductOf(Normal(0.0, 1.0), Normal(1.0, 1.0))
BayesBase.expectation(EA.SNIS(Normal(0.0, 2.0), 50_000, Xoshiro(3)), identity, target)
```

[`effective_sample_size`](@ref) says how many samples the weights are worth: close to `n` when
the proposal resembles the target, far fewer when it does not.

## Labels: exact or not

A backend says what holds for its results. [`ExactSum`](@ref) over a finite distribution is
exact; a sampling backend is stochastic; a quadrature is neither, since it is exact only for
polynomials:

```@example tutorial
(
    BayesBase.is_expectation_exact(EA.ExactSum(), sin, Categorical([0.5, 0.5])),
    BayesBase.is_expectation_stochastic(EA.MonteCarlo(10), sin, Normal(0.0, 1.0)),
    BayesBase.is_expectation_exact(EA.GaussHermite(8), sin, Normal(0.0, 1.0)),
)
```

An inference engine uses these labels to report what was approximated.

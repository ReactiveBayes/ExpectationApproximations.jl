```@meta
CurrentModule = ExpectationApproximations
```

# Computing expectations

The expectation protocol of BayesBase has two entry points:

- `BayesBase.expectation(backend, f, q)` computes ``\mathbb{E}_q[f(x)]`` in one call;
- `BayesBase.prepare_expectation(backend, q)` does the work that depends only on ``q``, and
  `BayesBase.expectation(prepared, f)` then evaluates the expectation of each function `f`.

```@example tutorial
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations
nothing # hide
```

## One expectation

A backend is a value that selects the method. For example, [`GaussHermite`](@ref)`(n)` is
Gauss–Hermite quadrature with `n` points for Gaussian distributions. For
``x \sim \mathcal{N}(1, 2^2)``, ``\mathbb{E}[x^2] = 1 + 4 = 5``:

```@example tutorial
BayesBase.expectation(EA.GaussHermite(8), x -> x^2, Normal(1.0, 2.0))
```

Gauss–Hermite quadrature with `n` points is exact for polynomials up to degree ``2n - 1``. For
other functions it is an approximation, which improves quickly with `n` when the function is
smooth:

```@example tutorial
# E[cos(x)] under N(0, 1) is exp(-1/2)
[BayesBase.expectation(EA.GaussHermite(n), cos, Normal(0.0, 1.0)) - exp(-1 / 2) for n in (2, 4, 8, 16)]
```

## Prepared expectations

The backends of this package prepare a *point set* for a distribution: points ``x_i`` and
weights ``w_i`` with ``\mathbb{E}_q[f] \approx \sum_i w_i f(x_i)``. The points are computed once,
when the expectation is prepared, and each evaluation is a weighted sum.

```@example tutorial
prepared = BayesBase.prepare_expectation(EA.GaussHermite(16), Normal(0.5, 2.0))
BayesBase.expectation(prepared, sin), BayesBase.expectation(prepared, abs2)
```

Evaluating a prepared expectation does not allocate, so it can be used in the inner loop of an
optimiser or of a message passing algorithm:

```@example tutorial
allocations(prepared, f) = @allocated BayesBase.expectation(prepared, f)
allocations(prepared, sin) # the first call compiles
allocations(prepared, sin)
```

## Derivatives

The element type of the points follows the parameters of the distribution, and the weighted sum
is generic Julia code, so ForwardDiff can differentiate through both the integrand and the
parameters. For
``x \sim \mathcal{N}(\mu, 1)``, ``\frac{d}{d\mu} \mathbb{E}[x^2] = 2\mu``:

```@example tutorial
using ForwardDiff
ForwardDiff.derivative(μ -> BayesBase.expectation(EA.GaussHermite(8), abs2, Normal(μ, 1.0)), 0.7)
```

[`laplace`](@ref) and [`gaussian_variational`](@ref) use this to optimise functions that are
themselves expectations. The nodes of Gauss–Laguerre and Gauss–Jacobi quadrature depend on the
shape of a Gamma or a Beta, so a derivative through a shape parameter throws an `ArgumentError`.
A derivative through the scale of a Gamma works.

## Products of distributions of different kinds

A joint distribution of independent components of different kinds, such as a categorical
assignment, a Gaussian vector and a Gamma precision, is a `BayesBase.FactorizedJoint`. A router
such as [`EADefault`](@ref) chooses a rule for each component and combines their point sets into a
product rule:

```@example tutorial
mixed = BayesBase.FactorizedJoint((Categorical([0.4, 0.6]), MvNormal([1.0, 2.0], [1.0 0.0; 0.0 2.0]), Gamma(2.0, 1.0)))
f((z, m, λ)) = λ * (3.0 - m[z])^2
BayesBase.expectation(EA.EADefault(), f, mixed)
```

The exact value is ``\mathbb{E}[\lambda] \sum_k \pi_k ((3 - \mu_k)^2 + \sigma_k^2) = 2 (0.4 \cdot 5 + 0.6 \cdot 3) = 7.6``.
The categorical component is summed exactly, and the other two are integrated by quadrature. The
number of points of a product rule is the product of the sizes of the component point sets, so it
is practical only for a few components.

## Random number generators

The sampling backends, [`MonteCarlo`](@ref) and [`SNIS`](@ref), need a random number generator.
They throw an error without one, so that no result depends on the global random state by
accident:

```@example tutorial
using Random
BayesBase.expectation(EA.MonteCarlo(10_000, Xoshiro(1)), log, Gamma(3.0, 2.0))
```

A caller that manages the random state, such as an inference engine, passes a generator with
`BayesBase.with_expectation_rng`. It sets the generator only on a backend that does not have one:

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

[`effective_sample_size`](@ref) estimates how many equally weighted samples the weighted ones are
worth. It is close to `n` when the proposal is close to the target, and much smaller when it is
not.

## Labels

A backend can label its results. [`ExactSum`](@ref) over a finite distribution is labelled exact,
and the sampling backends are labelled stochastic. A quadrature has neither label, because it is
exact only for polynomials:

```@example tutorial
(
    BayesBase.is_expectation_exact(EA.ExactSum(), sin, Categorical([0.5, 0.5])),
    BayesBase.is_expectation_stochastic(EA.MonteCarlo(10), sin, Normal(0.0, 1.0)),
    BayesBase.is_expectation_exact(EA.GaussHermite(8), sin, Normal(0.0, 1.0)),
)
```

An inference engine can use the labels to report which expectations were approximated.

```@meta
CurrentModule = ExpectationApproximations
```

# ExpectationApproximations

ExpectationApproximations computes expectations ``\mathbb{E}_q[f(x)]`` of a function ``f``
under a distribution ``q`` numerically. Examples are ``\mathbb{E}[\log x]`` under a Gamma
distribution, ``\mathbb{E}[\sigma(a + b x)]`` under a Gaussian, and the expected log-likelihood of
a factor in a model. The package implements quadrature, cubature, sigma points, exact sums and
sampling as backends of the expectation protocol of
[BayesBase](https://github.com/ReactiveBayes/BayesBase.jl).

The package also approximates a density, given as a Julia function, by a Gaussian. Approximate
inference uses three such Gaussians: the Laplace approximation at the mode ([`laplace`](@ref)),
the Gaussian with the moments of the density, as in expectation propagation
([`moment_matching`](@ref)), and the Gaussian closest to the density, as in variational inference
([`gaussian_variational`](@ref)). Related functions compute the moments of functions of random
variables. For a variable that must be positive, a probability or a probability vector, the
Gaussian approximation is made in unconstrained coordinates and transformed back.

## A first example

Gauss–Legendre quadrature in the logarithm of the variable computes the expected logarithm of a
Gamma variable:

```@example index
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations

q = Gamma(3.0, 2.0)
BayesBase.expectation(EA.UnconstrainedLegendre(), log, q)
```

The exact value is ``\psi(3) + \log 2``:

```@example index
using SpecialFunctions: digamma
digamma(3.0) + log(2.0)
```

Usually the rule is not picked by hand: [`EADefault`](@ref) chooses the rule for the family of
each distribution. A backend can also be *prepared* once for a distribution and then evaluated
for many functions without allocating:

```@example index
prepared = BayesBase.prepare_expectation(EA.EADefault(), q)
BayesBase.expectation(prepared, x -> x^2), BayesBase.expectation(prepared, sqrt)
```

[`laplace`](@ref) takes the logarithm of an unnormalised density and a starting point. For a
skewed density on the real line:

```@example index
result = EA.laplace(x -> x - exp(x) / 2, 0.0)
result.distribution, result.converged
```

## Related packages

ExpectationApproximations is part of the [ReactiveBayes](https://github.com/ReactiveBayes)
ecosystem:

| Package | Role |
|:--------|:-----|
| [BayesBase](https://github.com/ReactiveBayes/BayesBase.jl) | defines the *expectation protocol*: `expectation`, `prepare_expectation`, and how a backend says what it supports |
| ExpectationApproximations | implements backends for the protocol, and the Gaussian approximations built on them |
| [DeltaMessagePassingRules](https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl) | turns these approximations into message passing rules for nodes that have no hand-written ones |
| [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) | runs inference on models without hand-written rules with `infer(...; fallback = LocalLaplace())`, `LaplacePropagation()`, `ExpectationPropagation()` or `NaturalGradientPropagation()` |

The package does not depend on message passing, and its backends and approximations can be used
on their own.

## Installation

```julia
import Pkg
Pkg.add("ExpectationApproximations")
```

The package exports no names. Its public names are used qualified, as in the examples above,
usually through `const EA = ExpectationApproximations`. The protocol functions `expectation` and
`prepare_expectation` belong to BayesBase and are called as `BayesBase.expectation`.

## Where to go next

- [Computing expectations](@ref) is a tutorial on the protocol: direct and prepared
  expectations, derivatives, products of distributions and random number generators.
- [How a backend is chosen](@ref) explains routers, [`EADefault`](@ref) and the registry.
- [Choosing a backend](@ref) compares the rules, with their accuracy, cost and limits.
- [Gaussian approximations of a density](@ref) compares Laplace, moment matching and the
  variational Gaussian on one density.
- [Moments through functions](@ref) covers the moments of ``f(x)``, tilted distributions and
  linearisation.
- [Constrained variables](@ref) explains domains and transforms.
- [Quadrature rules](@ref) describes the classical rules behind the quadrature backends.
- [Writing a backend](@ref) shows how to implement and test a new backend.
- [Migrating](@ref) maps the API of earlier versions and of MessagePassingRulesApproximations to
  the protocol.

```@docs
ExpectationApproximations
```

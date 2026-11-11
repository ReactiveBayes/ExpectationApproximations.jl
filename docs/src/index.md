```@meta
CurrentModule = ExpectationApproximations
```

# ExpectationApproximations

Approximate inference spends most of its time on two questions:

- **What is the expectation of a function?** For example ``\mathbb{E}_q[f(x)]`` under a
  distribution ``q`` that has no closed form for it: ``\mathbb{E}[\log x]`` under a Gamma,
  ``\mathbb{E}[\sigma(a + b x)]`` under a Gaussian, the expected log-likelihood of a model's
  factor.
- **Which Gaussian best represents a density?** Approximate inference often has to replace an
  intractable density by a Gaussian: at its mode (Laplace), with its moments (expectation
  propagation) or as the Gaussian closest to it (variational inference).

ExpectationApproximations answers both, numerically. It provides:

- **backends** that compute expectations by quadrature, cubature, sigma points, exact sums or
  sampling, through the expectation protocol of
  [BayesBase](https://github.com/ReactiveBayes/BayesBase.jl);
- **Gaussian approximations** of a density given as a Julia function: [`laplace`](@ref),
  [`moment_matching`](@ref), [`gaussian_variational`](@ref), and the moments of functions of
  random variables;
- **domains and transforms**, so a variable that must be positive, a probability or a
  probability vector is approximated by a Gaussian in unconstrained coordinates.

## A first example

The expected logarithm of a Gamma variable has no elementary closed form in most libraries, but
Gauss–Laguerre quadrature computes it accurately with a handful of points:

```@example index
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations

q = Gamma(3.0, 2.0)
BayesBase.expectation(EA.GaussLaguerre(32), log, q)
```

The exact value is ``\psi(3) + \log 2``:

```@example index
using SpecialFunctions: digamma
digamma(3.0) + log(2.0)
```

Most of the time one does not pick the rule by hand. [`EADefault`](@ref) sends each distribution
to the rule made for its family, and a backend can be *prepared* once for a distribution and then
evaluated for many functions, without allocating:

```@example index
prepared = BayesBase.prepare_expectation(EA.EADefault(), q)
BayesBase.expectation(prepared, x -> x^2), BayesBase.expectation(prepared, sqrt)
```

And a Gaussian approximation of a density is one call. Here, the Laplace approximation of a
skewed density on the real line:

```@example index
result = EA.laplace(x -> x - exp(x) / 2, 0.0)
result.distribution, result.converged
```

## Where it fits

ExpectationApproximations is one layer of the [ReactiveBayes](https://github.com/ReactiveBayes)
ecosystem:

| Package | Role |
|:--------|:-----|
| [BayesBase](https://github.com/ReactiveBayes/BayesBase.jl) | defines the *expectation protocol*: `expectation`, `prepare_expectation`, and how a backend says what it supports |
| **ExpectationApproximations** | implements backends for the protocol, and the Gaussian approximations built on them |
| [DeltaMessagePassingRules](https://github.com/ReactiveBayes/DeltaMessagePassingRules.jl) | turns these approximations into message passing rules for nodes that have no hand-written ones |
| [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) | runs inference on models without hand-written rules with `infer(...; fallback = LocalLaplace())`, `LaplacePropagation()`, `ExpectationPropagation()` or `NaturalGradientPropagation()` |

Nothing in this package depends on message passing: the backends and the approximations work on
any distribution and any Julia function.

## Installation

```julia
import Pkg
Pkg.add("ExpectationApproximations")
```

Nothing is exported: every name is public and used qualified, as in the examples above, usually
with `const EA = ExpectationApproximations`. The functions of the expectation protocol,
`expectation` and `prepare_expectation`, are BayesBase's.

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
- [Quadrature rules](@ref), [Writing a backend](@ref) and [Migrating](@ref) complete the manual.

```@docs
ExpectationApproximations
```

```@meta
CurrentModule = ExpectationApproximations
```

# How a backend is chosen

Each quadrature rule is made for one family of distributions: Gauss–Hermite for Gaussians,
Gauss–Legendre in the logarithm for Gammas, an exact sum for categorical distributions. A model usually contains
several families, so it needs a backend that chooses a rule for each distribution.

```@example routing
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations
nothing # hide
```

## What a backend supports

A backend declares, on types, which integrands and distributions it handles:
`BayesBase.supports_expectation(backend, F, Q)` is `true` when it computes ``\mathbb{E}_q[f]``
for `f::F` and `q::Q`. The integrand type `Any` means any integrand. A quadrature supports it,
because a quadrature works the same way for every function.

```@example routing
(
    BayesBase.supports_expectation(EA.GaussHermite(8), Any, Normal{Float64}),
    BayesBase.supports_expectation(EA.GaussHermite(8), Any, Gamma{Float64}),
)
```

Because support is declared on types, the compiler resolves the choice of a backend, and the
choice has no run-time cost.

## Routers

`BayesBase.ExpectationFirstOf(backends...)` tries its backends in order and uses the first one
that supports the integrand and the distribution. [`EADefault`](@ref) is such a router, with one
rule per family:

```julia
EADefault(n = 16) = ExpectationFirstOf(ExactSum(), GaussHermite(n), UnconstrainedLegendre(), QuantileLegendre(n))
```

`BayesBase.which_expectation_backend` shows the choice:

```@example routing
backend = EA.EADefault()
distributions = (Normal(0.0, 1.0), Gamma(2.0, 1.0), Beta(2.0, 3.0), Categorical([0.5, 0.5]), Weibull(2.0, 1.0))
[nameof(typeof(q)) => nameof(typeof(BayesBase.which_expectation_backend(backend, sin, q))) for q in distributions]
```

The Weibull distribution has no rule of its own, so it goes to [`QuantileLegendre`](@ref), the
fallback for continuous univariate distributions. A distribution that none of the rules supports,
such as `Poisson`, has no backend, and an expectation under it throws
`BayesBase.UnsupportedExpectationError`:

```@example routing
BayesBase.which_expectation_backend(backend, sin, Poisson(3.0)) === nothing
```

To change the choice, build another router. For example, put a more accurate rule first, or a
closed form from another package before the quadratures:

```@example routing
accurate = BayesBase.ExpectationFirstOf(EA.GaussHermite(40), EA.EADefault())
BayesBase.which_expectation_backend(accurate, sin, Normal(0.0, 1.0))
```

## The registry

A library that runs inference does not know which backend packages the user has loaded.
BayesBase therefore keeps a registry of backends, and `BayesBase.ExpectationDynamicRouter()` chooses
among the registered backends when it is called. Loading this package registers the rules of
[`EADefault`](@ref) through [`register_backends!`](@ref). A package with closed forms registers
its own backends, and they are chosen first where they are exact.

The dynamic router chooses at run time, which takes microseconds per call and allocates. Wrapping
it in `ExpectationFirstOf` fixes the current choice of the registry in a router that the compiler
resolves, which is as fast as `EADefault()`:

```@example routing
frozen = BayesBase.ExpectationFirstOf(BayesBase.ExpectationDynamicRouter())
BayesBase.which_expectation_backend(frozen, sin, Gamma(2.0, 1.0))
```

RxInfer's fallback schemes use this to find a backend when the model does not specify one.

```@docs
EADefault
register_backends!
```

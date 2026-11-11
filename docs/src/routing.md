```@meta
CurrentModule = ExpectationApproximations
```

# How a backend is chosen

No single rule is best for every distribution: Gauss–Hermite suits a Gaussian, Gauss–Laguerre a
Gamma, an exact sum a categorical. A model mixes them, so it needs a backend that chooses. This
page explains how that choice is made, from the most explicit way to the most automatic.

```@example routing
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations
nothing # hide
```

## What a backend supports

A backend declares, on types, which distributions and integrands it handles:
`BayesBase.supports_expectation(backend, F, Q)` is `true` when it computes ``\mathbb{E}_q[f]``
for `f::F` and `q::Q`. Integrand type `Any` means "every integrand", which is what a quadrature
declares, since a quadrature does not care what the function is.

```@example routing
(
    BayesBase.supports_expectation(EA.GaussHermite(8), Any, Normal{Float64}),
    BayesBase.supports_expectation(EA.GaussHermite(8), Any, Gamma{Float64}),
)
```

Because the declaration is on types, the compiler decides it: choosing a backend costs nothing at
run time.

## Routers: the first backend that supports it

`BayesBase.ExpectationFirstOf(backends...)` tries its backends in order and uses the first that
supports the pair. [`EADefault`](@ref) is such a router, with one rule per family:

```julia
EADefault(n = 16) = ExpectationFirstOf(ExactSum(), GaussHermite(n), GaussLaguerre(n), GaussJacobi(n), QuantileLegendre(n))
```

`BayesBase.which_expectation_backend` shows the choice:

```@example routing
backend = EA.EADefault()
distributions = (Normal(0.0, 1.0), Gamma(2.0, 1.0), Beta(2.0, 3.0), Categorical([0.5, 0.5]), Weibull(2.0, 1.0))
[nameof(typeof(q)) => nameof(typeof(BayesBase.which_expectation_backend(backend, sin, q))) for q in distributions]
```

The Weibull distribution has no rule of its own, so it goes to [`QuantileLegendre`](@ref), the
fallback for any continuous univariate distribution. A distribution none of them covers, a
`Poisson` for one, has no backend, and an expectation under it throws
`BayesBase.UnsupportedExpectationError`:

```@example routing
BayesBase.which_expectation_backend(backend, sin, Poisson(3.0)) === nothing
```

To change the choice, build your own router: put a more accurate rule first, or a closed form
from another package before the quadratures.

```@example routing
accurate = BayesBase.ExpectationFirstOf(EA.GaussHermite(40), EA.EADefault())
BayesBase.which_expectation_backend(accurate, sin, Normal(0.0, 1.0))
```

## The registry: a choice the session makes

A library that runs inference does not know which backends the user's session has loaded.
BayesBase keeps a registry of backends, and `BayesBase.ExpectationDynamicRouter()` chooses among
the registered ones when it is called. Loading this package registers the rules of
[`EADefault`](@ref), through [`register_backends!`](@ref); a package with closed forms registers
its own, and they come first where they are exact.

Choosing at run time costs microseconds per call and allocates. Wrapping the dynamic router in
`ExpectationFirstOf` freezes the choice the registry makes now into a router the compiler
resolves, as fast as `EADefault()`:

```@example routing
frozen = BayesBase.ExpectationFirstOf(BayesBase.ExpectationDynamicRouter())
BayesBase.which_expectation_backend(frozen, sin, Gamma(2.0, 1.0))
```

This is how RxInfer's fallback schemes find a backend when the model names none.

```@docs
EADefault
register_backends!
```

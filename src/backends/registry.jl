"""
    register_backends!(; kwargs...)

Add the rules of [`EADefault`](@ref), with 16 points, to a registry of backends for automatic
choice, BayesBase's global one by default. A `BayesBase.ExpectationDynamicRouter` chooses among
the registered backends when it is called; `BayesBase.ExpectationFirstOf(ExpectationDynamicRouter())`
freezes that choice into a router the compiler resolves. With only these rules registered, both
choose as `EADefault()` does. A backend registered by another package comes before them where it
is exact for a pair or has a higher priority. This is how a model that names no backend, such as
RxInfer's `fallback = LocalLaplace()`, finds one.

- `QuantileLegendre`, which takes any continuous univariate distribution, is registered with the
  lowest priority, `-100`, so that it is chosen only where no other rule applies.
- The stochastic backends, `MonteCarlo` and `SNIS`, are not registered: they need a random
  number generator, and an automatic choice stays deterministic.

Called when the package loads; registering again changes nothing. The keywords are passed on to
`BayesBase.register_expectation_backend!`: `registry` names another
`BayesBase.ExpectationBackendRegistry`.
"""
function register_backends!(; kwargs...)
    for backend in (ExactSum(), GaussHermite(16), GaussLaguerre(16), GaussJacobi(16))
        BayesBase.register_expectation_backend!(backend; kwargs...)
    end
    return BayesBase.register_expectation_backend!(QuantileLegendre(16); priority = -100, kwargs...)
end

__init__() = (register_backends!(); nothing)

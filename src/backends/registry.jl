"""
    register_backends!(; kwargs...)

Add the rules of [`EADefault`](@ref), with 16 points, to a registry of backends for automatic
choice, by default the global registry of BayesBase. A `BayesBase.ExpectationDynamicRouter`
chooses among the registered backends when it is called, and
`BayesBase.ExpectationFirstOf(ExpectationDynamicRouter())` fixes that choice in a router that the
compiler resolves. With only these rules registered, both choose as `EADefault()` does. A backend
registered by another package is chosen before them where it is exact for the integrand and the
distribution, or where it has a higher priority. A model that specifies no backend, such as one
run with RxInfer's `fallback = LocalLaplace()`, finds one this way.

No two of these rules cover the same family, so the choice among them never depends on the order
in which backends are registered or on their names. `QuantileLegendre`, which supports any
continuous univariate distribution with a quantile function, is registered with a priority 100
below the others, so that it is chosen only where no other rule applies.
[`GaussLaguerre`](@ref) and [`GaussJacobi`](@ref) are not registered. The stochastic
backends, `MonteCarlo` and `SNIS`, are not registered: they need a random number generator, and
the automatic choice is kept deterministic.

The package calls this function when it loads; registering again changes nothing. The keywords
are passed on to `BayesBase.register_expectation_backend!`; for example, `registry` selects
another `BayesBase.ExpectationBackendRegistry`.
"""
function register_backends!(; priority::Integer = 0, kwargs...)
    for backend in (ExactSum(), GaussHermite(16), UnconstrainedLegendre(8))
        BayesBase.register_expectation_backend!(backend; priority, kwargs...)
    end
    return BayesBase.register_expectation_backend!(QuantileLegendre(16); priority = priority - 100, kwargs...)
end

__init__() = (register_backends!(); nothing)

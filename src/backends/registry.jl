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

`QuantileLegendre`, which supports any continuous univariate distribution, is registered with the
lowest priority, `-100`, so that it is chosen only where no other rule applies. The stochastic
backends, `MonteCarlo` and `SNIS`, are not registered: they need a random number generator, and
the automatic choice is kept deterministic.

The package calls this function when it loads; registering again changes nothing. The keywords
are passed on to `BayesBase.register_expectation_backend!`; for example, `registry` selects
another `BayesBase.ExpectationBackendRegistry`.
"""
function register_backends!(; kwargs...)
    for backend in (ExactSum(), GaussHermite(16), GaussLaguerre(16), GaussJacobi(16))
        BayesBase.register_expectation_backend!(backend; kwargs...)
    end
    return BayesBase.register_expectation_backend!(QuantileLegendre(16); priority = -100, kwargs...)
end

__init__() = (register_backends!(); nothing)

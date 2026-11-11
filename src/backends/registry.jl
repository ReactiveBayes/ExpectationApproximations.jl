"""
    register_backends!(; registry)

Offer the rules of [`EADefault`](@ref) to automatic choice: BayesBase's
`ExpectationDynamicRouter`, and so the default backend of a fallback scheme such as RxInfer's
`LocalLaplace`. `ExpectationFirstOf(ExpectationDynamicRouter())` then chooses as
`EADefault()` does, and a backend registered by another package, exact for a pair or of a higher
priority, comes before these rules.

`QuantileLegendre`, which takes any continuous distribution, has the lowest priority, so that it is
chosen only where no other rule applies. The stochastic backends, `MonteCarlo` and `SNIS`, are
not registered: they need a random number generator, and an automatic choice stays
deterministic.

Called when the package loads, on BayesBase's global registry; registering again changes nothing.
`registry` is any other `BayesBase.ExpectationBackendRegistry`.
"""
function register_backends!(; kwargs...)
    for backend in (ExactSum(), GaussHermite(16), GaussLaguerre(16), GaussJacobi(16))
        BayesBase.register_expectation_backend!(backend; kwargs...)
    end
    return BayesBase.register_expectation_backend!(QuantileLegendre(16); priority = -100, kwargs...)
end

__init__() = (register_backends!(); nothing)

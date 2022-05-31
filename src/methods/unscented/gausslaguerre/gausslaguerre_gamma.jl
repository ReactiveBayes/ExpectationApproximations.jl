using Distributions

## Gauss Laguerre Optimized Rules for Gamma-shaped Distribution ##

call_approximate(::Val{false}, method::GaussLaguerre, q::Gamma, f::LogFn) = call_approximate(Val(true), method, q, f)

"""
    call_approximate(::Val{true}, method::EvaluatedGaussLaguerre, q::Gamma, f)
                     ^^^^^^^^^^^
                    force log computations flag

This function calculates the Gauss-laguerre integral by fusing Gamma distribution `q(x)` into `f` and by making use of the log of the integrable function.
    ∫ q(x) * f(x) dx = ∫ exp(-x) * g(x) dx
    ≈ ln ( ∑ wi * g(xi) ) 
    = ln ( ∑ exp( ln(wi) + logg(xi) ) )
    = ln ( ∑ exp( yi ) )
    = max(yi) + ln ( ∑ exp( yi - max(yi) ) )
    where we make use of the numerically stable log-sum-exp trick: https://en.wikipedia.org/wiki/LogSumExp and g(x) = f(x/β) * x^(α - 1)/ Γ(x).

In case if `f` is an instance of `LogFn` the result is expected to be more accurate. However, generic `f` is also supported. In the case if `f` returns negative 
values this approximation switches to the `Complex` numbers and checks that result in the real domain. 

See also: [`LogFn`](@ref)
"""
function call_approximate(::Val{true}, method::GaussLaguerre, q::Gamma, f)
    points     = method.points
    logweights = method.logweights
    logf       = _gausslaguerre_force_log(f)
    α          = shape(q)
    invβ       = inv(rate(q))
    logΓα      = loggamma(α)

    # g(x) = f(x/β) * x^(α - 1)/ Γ(x)
    # logg(x) = logf(x/β) + (α - 1) * log(x) - loggamma(α)
    logg = (x) -> logf(invβ * x) + (α - 1) * log(x) # We subtract `loggamma(α)` later on 

    # Iterators.map creates a lazy mapping
    logresult = Iterators.map(zip(points, logweights)) do (x, logw)
        # calculate the ln(wi) + logg(xi) terms
        return logw + logg(x)
    end

    return _gausslaguerre_force_log_process_result(f, logsumexp(logresult) - logΓα)
end

"""
    call_approximate(_, method::EvaluatedGaussLaguerre, q::Gamma, f)

This function calculates the Gauss-laguerre integral ∫ q(x) * f(x) dx = ∫ exp(-x) * g(x) dx by fusing Gamma distribution `q(x)` into `f` (g(x) = f(x/β) * x^(α - 1) / Γ(x)).
"""
function call_approximate(::Val{false}, method::GaussLaguerre, q::Gamma, f)
    points     = method.points
    weights    = method.weights
    α          = shape(q)
    invβ       = inv(rate(q))
    logΓα      = loggamma(α)

    # g(x) = f(x/β) * x^(α - 1) / Γ(x)
    g = (x) -> f(invβ * x) * x ^ (α - 1) # we divide by Γ(α) later on

    result = Iterators.mapreduce(+, zip(points, weights)) do (x, w)
        return w * g(x)
    end

    return exp(log(result) - logΓα)
end

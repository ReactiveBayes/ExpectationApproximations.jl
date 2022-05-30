export Unscented, GaussLaguerre

using FastGaussQuadrature
using Distributions
using LogarithmicNumbers

import Base: length
import StatsFuns: logsumexp
import SpecialFunctions: loggamma

struct Unscented{R}
    rule::R
end

call_approximate(method::Unscented, q, f) = call_approximate(evaluate(method.rule, q), q, f)
call_mean_var(method::Unscented, q, f)    = call_mean_var(evaluate(method.rule, q), q, f)

## Gauss Laguerre ##

struct GaussLaguerre
    length::Int
end

Base.length(method::GaussLaguerre) = method.length

struct EvaluatedGaussLaguerre{P, W, L}
    points     :: P
    weights    :: W
    logweights :: L
end

Base.length(method::EvaluatedGaussLaguerre) = length(method.points)

function evaluate(rule::GaussLaguerre, q::Gamma)
    points, weights = gausslaguerre(length(rule), shape(q))
    return EvaluatedGaussLaguerre(points, weights, map(log, weights))
end

"""
    call_approximate(method::EvaluatedGaussLaguerre, q::Gamma, f::LogFn)

This function calculates the log of the Gauss-laguerre integral by making use of the log of the integrable function.
    ln ( ∫ x^α * exp(-x) * f(x) dx ) 
    ≈ ln ( ∑ wi * f(xi) ) 
    = ln ( ∑ exp( ln(wi) + logf(xi) ) )
    = ln ( ∑ exp( yi ) )
    = max(yi) + ln ( ∑ exp( yi - max(yi) ) )
    where we make use of the numerically stable log-sum-exp trick: https://en.wikipedia.org/wiki/LogSumExp

See also: [`LogFn`](@ref)
"""
function call_approximate(method::EvaluatedGaussLaguerre, q::Gamma, f::LogFn)
    points     = method.points
    logweights = method.logweights
    logf       = log(f)
    α          = shape(q)
    invβ       = inv(rate(q))

    # g(x)    = f(x) / (x * Γ(x))
    # logg(x) = logf(x) - log(x) - loggamma(x)
    logg = (x) -> logf(invβ * x) - log(x) - loggamma(α)

    # Iterators.map creates a lazy mapping
    logresult = Iterators.map(zip(points, logweights)) do (x, logw)
        # calculate the ln(wi) + logg(xi) terms
        return logw + logg(x)
    end

    # return log sum exp in form of the `Logarithmic` number from `LogarithmicNumbers.jl` package
    return exp(Logarithmic, logsumexp(logresult))
end

# TODO for testing pusposes only
function call_mean_var(method::EvaluatedGaussLaguerre, q::Gamma, f::LogFn)
    logf = log(f)

    # First we compute a normalisation constant in log-domain
    C    = call_approximate(method, q, f)
    logC = float(log(C))

    # Second we compute mean using previously computed normalisation constant in log domain
    # m(x)    = x * f(x) / C
    # logm(x) = logf(x) + log(x) - log(C)
    logm = (x) -> logf(x) + log(x) - logC
    m    = float(call_approximate(method, q, LogFn(logm)))

    # At third we compute var using previously computed C and m in log domain
    # v(x)    = f(x) * (x - m)^2 / C
    # logv(x) = logf(x) + 2log(abs(x - m)) - log(C) 
    logv = (x) -> logf(x) + 2log(abs(x - m)) - logC
    v    = float(call_approximate(method, q, LogFn(logv)))

    return m, v
end
export Unscented, GaussLaguerre

using FastGaussQuadrature
using Distributions
using LogarithmicNumbers

import Base: length
import LogExpFunctions: logsumexp
import SpecialFunctions: loggamma, gamma

struct Unscented{R}
    rule::R
end

call_approximate(method::Unscented, q, f) = call_approximate(evaluate(method.rule, q), q, f)
call_mean_var(method::Unscented, q, f)    = call_mean_var(evaluate(method.rule, q), q, f)

## Gauss Laguerre ##

struct GaussLaguerre{ForceLogComputations, P, W, L}
    length     :: Int
    points     :: P
    weights    :: W
    logweights :: L

    GaussLaguerre(::Val{ForceLogComputations}, length::Int, points::P, weights::W, logweights::L) where { ForceLogComputations, P, W, L } = begin 
        return new{ForceLogComputations, P, W, L}(length, points, weights, logweights)
    end
end

GaussLaguerre(length::Int) = GaussLaguerre(length, Val(false))

function GaussLaguerre(length::Int, ::Val{ForceLogComputations}) where { ForceLogComputations }
    points, weights = gausslaguerre(length, 0.0)
    logweights      = map(log, weights)
    return GaussLaguerre(Val{ForceLogComputations}(), length, points, weights, logweights)
end

Base.length(method::GaussLaguerre) = method.length

function evaluate(rule::GaussLaguerre, q)
    return rule
end

function call_approximate(method::GaussLaguerre{ForceLogComputations}, q, f) where { ForceLogComputations }
    return call_approximate(Val{ForceLogComputations}(), method, q, f)
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
function call_approximate(_, method::GaussLaguerre, q::Gamma, f::LogFn)
    points     = method.points
    logweights = method.logweights
    logf       = log(f)
    α          = shape(q)
    invβ       = inv(rate(q))
    logΓα      = loggamma(α)

    # g(x)    = f(x) / (x * Γ(x))
    # logg(x) = logf(x) - log(x) - loggamma(α)
    logg = (x) -> logf(invβ * x) + (α - 1) * log(x)

    # Iterators.map creates a lazy mapping
    logresult = Iterators.map(zip(points, logweights)) do (x, logw)
        # calculate the ln(wi) + logg(xi) terms
        return logw + logg(x)
    end

    # return log sum exp in form of the `Logarithmic` number from `LogarithmicNumbers.jl` package
    return exp(Logarithmic, logsumexp(logresult) - logΓα)
end

# Force log version with possibly complex numbers
# TODO: remove ::Function constraint
function call_approximate(::Val{true}, method::GaussLaguerre, q::Gamma, f::Function)
    points     = method.points
    logweights = method.logweights
    α          = shape(q)
    invβ       = inv(rate(q))
    logΓα      = loggamma(α)

    # g(x) = f(invβ * x) * x ^ (α - 1) # we divide by Γ(α) later on
    # logg(x) = real(exp(log(Complex(f(invβ * x))) + (α - 1) * log(x) - logΓα))
    logg = (x) -> log(Complex(f(invβ * x))) + (α - 1) * log(x)

    # Iterators.map creates a lazy mapping
    logresult = Iterators.map(zip(points, logweights)) do (x, logw)
        # calculate the ln(wi) + logg(xi) terms
        return logw + logg(x)
    end

    # return result in form of the `Logarithmic` number from `LogarithmicNumbers.jl` package
    result = exp(logsumexp(logresult) - logΓα)
    if !(imag(result) < 1e-8)
        error("GaussLaguerre failed to compute an approximation, result is a complex number: $(result)")
    end
    return real(result)
end

# TODO: remove ::Function constraint
function call_approximate(::Val{false}, method::GaussLaguerre, q::Gamma, f::Function)
    points     = method.points
    weights    = method.weights
    α          = shape(q)
    invβ       = inv(rate(q))
    logΓα      = loggamma(α)

    # g(x)    = f(x) / (x * Γ(x))
    # logg(x) = logf(x) - log(x) - loggamma(α)
    g = (x) -> f(invβ * x) * x ^ (α - 1) # we divide by Γ(α) later on

    result = Iterators.mapreduce(+, zip(points, weights)) do (x, w)
        return w * g(x)
    end

    return exp(log(result) - logΓα)
end

# TODO for testing pusposes only
function call_mean_var(method::GaussLaguerre, q::Gamma, f::LogFn)
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

# TODO for testing pusposes only
# TODO: remove ::Function constraint
function call_mean_var(method::GaussLaguerre, q::Gamma, f::Function)

    # First we compute a normalisation constant
    C = float(call_approximate(method, q, f))

    # Second we compute mean using previously computed normalisation constant
    # m(x)    = x * f(x) / C
    mf = (x) -> x * f(x) / C
    m  = float(call_approximate(method, q, mf))

    # At third we compute var using previously computed C and m
    # v(x)    = f(x) * (x - m)^2 / C
    vf = (x) -> (x - m)^2 * f(x) / C
    v  = float(call_approximate(method, q, vf))

    return m, v
end
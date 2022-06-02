export Unscented, GaussLaguerre, UseLogWeights, UseWeights

using FastGaussQuadrature
using Distributions
using LogarithmicNumbers

import Base: length, show
import LogExpFunctions: logsumexp
import SpecialFunctions: loggamma, gamma

abstract type UnscentedRule end

struct UnscentedStrategyAutomatic end

struct UnscentedRuleAutomatic 
    npoints :: Int
end

struct UseWeights end
struct UseLogWeights end

struct Unscented{S, R}
    strategy :: S
    rule     :: R
end

Unscented(npoints::Int)        = Unscented(UnscentedStrategyAutomatic(), UnscentedRuleAutomatic(npoints))
Unscented(rule::UnscentedRule) = Unscented(UnscentedStrategyAutomatic(), rule)

Base.show(io::IO, method::Unscented) = print(io, "Unscented(", getstrategy(method), ", ", getrule(method), ")")

getstrategy(method::Unscented, f)       = method.strategy
getstrategy(method::Unscented, ::LogFn) = UseLogWeights() # In case if input is a `LogFn` we always use log weights

getrule(method::Unscented)     = method.rule

function call_approximate(method::Unscented, q, f) 
    compatibility = iscompatible(getrule(method), q)
    # TODO Check compatibility
    strategy      = getstrategy(method, f)
    # TODO Check evaluate(rule, q) -> getpoints(rule, q)
    rule          = evaluate(getrule(method), q)
    return unscented_approximate(strategy, rule, q, f)
end

call_mean_var(method::Unscented, q, f) = unscented_mean_var(evaluate(method.rule, q), q, f)

getpoints(method::Unscented)     = getpoints(method.rule)
getweights(method::Unscented)    = getweights(method.rule) 
getlogweights(method::Unscented) = getlogweights(method.rule)

## Generic implementation

function unscented_approximate(::UseWeights, rule, q, f)
    points  = getpoints(rule)
    weights = getweights(rule)
    g       = adjust_integrand(rule, q, f)

    result = Iterators.mapreduce(+, zip(points, weights)) do (x, w)
        return w * g(x)
    end

    return adjust_result(rule, q, result)
end

function unscented_approximate(::UseLogWeights, rule, q, f)
    points     = getpoints(rule)
    logweights = getlogweights(rule)
    logf       = log(convert(LogFn, f))

    logg = adjust_logintegrand(rule, q, logf)

    # ln ( ∑ wi * g(xi) ) 
    # = ln ( ∑ exp( ln(wi) + logg(xi) ) )
    # = ln ( ∑ exp( yi ) )
    # = max(yi) + ln ( ∑ exp( yi - max(yi) ) )  <-- `logsumexp`

    # Iterators.map creates a lazy mapping
    logresult = Iterators.map(zip(points, logweights)) do (x, logw)
        # calculate the ln(wi) + logg(xi) terms
        return logw + logg(x)
    end

    logadjusted = adjust_logresult(rule, q, logsumexp(logresult))

    return _unscented_force_log_process_result(f, rule, logadjusted)
end

# Utility functions

_unscented_is_complex(x::Number)  = false
_unscented_is_complex(x::Complex) = true

function _unscented_force_log_process_result(f, rule, logresult)
    # In general case it might be that `logresult` is complex, but imaginary part should be almost zero
    # Otherwise it indicates that something went wrong
    result = exp(logresult)
    if _unscented_is_complex(result) && !(imag(result) < 1e-8) # 1e-8 is sort of arbitrary
        error("Unscented failed to compute an approximation with $(name(rule)) rule, result is a complex number with non-zero imaginary part: $(result)")
    end
    return real(result)
end

function _unscented_force_log_process_result(::LogFn, rule, logresult)
    # In case of LogFn we assume that `logresult` is not complex
    # Otherwise it indicates that something went wrong, and `Logarithmic` will throw anyway
    # return `logresult` in a form of the `Logarithmic` number from `LogarithmicNumbers.jl` package
    return exp(Logarithmic, logresult)
end

##

include("gausslaguerre/gausslaguerre.jl")

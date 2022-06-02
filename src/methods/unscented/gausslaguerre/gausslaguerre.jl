export GaussLaguerre

import Base: show

struct GaussLaguerre{P, W, L} <: UnscentedRule
    length     :: Int
    points     :: P
    weights    :: W
    logweights :: L
end

function GaussLaguerre(length::Int)
    # α = 0.0 is on purpose as it is nicely computed and numerically stable
    # Revise `α` as a parameter later on, but we observed that with `α > 30.0` sigma points and weights are extremelly unstable
    points, weights = gausslaguerre(length, 0.0) 

    # Filter out zero weights
    nonzeroweights = map(>(zero(eltype(weights))), weights)

    # `keepat!` exists only in 1.7 :(
    keepat!(points, nonzeroweights)
    keepat!(weights, nonzeroweights)

    logweights = map(log, weights)

    return GaussLaguerre(length, points, weights, logweights)
end

Base.length(method::GaussLaguerre)       = method.length
Base.show(io::IO, method::GaussLaguerre) = print(io, "GaussLaguerre(", length(method), ")")

evaluate(rule::GaussLaguerre, q) = rule

getpoints(method::GaussLaguerre)     = method.points
getweights(method::GaussLaguerre)    = method.weights
getlogweights(method::GaussLaguerre) = method.logweights

##

# TODO for testing pusposes only
# function call_mean_var(method::GaussLaguerre, q::Any, f::LogFn)
#     logf = log(f)

#     # First we compute a normalisation constant in log-domain
#     C    = call_approximate(method, q, f)
#     logC = float(log(C))

#     # Second we compute mean using previously computed normalisation constant in log domain
#     # m(x)    = x * f(x) / C
#     # logm(x) = logf(x) + log(x) - log(C)
#     logm = (x) -> logf(x) + log(x) - logC
#     m    = float(call_approximate(method, q, LogFn(logm)))

#     # At third we compute var using previously computed C and m in log domain
#     # v(x)    = f(x) * (x - m)^2 / C
#     # logv(x) = logf(x) + 2log(abs(x - m)) - log(C) 
#     logv = (x) -> logf(x) + 2log(abs(x - m)) - logC
#     v    = float(call_approximate(method, q, LogFn(logv)))

#     return m, v
# end

# TODO for testing pusposes only
# TODO: remove ::Function constraint
# function call_mean_var(method::GaussLaguerre, q::Any, f)

#     # First we compute a normalisation constant
#     C = float(call_approximate(method, q, f))

#     # Second we compute mean using previously computed normalisation constant
#     # m(x)    = x * f(x) / C
#     mf = (x) -> x * f(x) / C
#     m  = float(call_approximate(method, q, mf))

#     # At third we compute var using previously computed C and m
#     # v(x)    = f(x) * (x - m)^2 / C
#     vf = (x) -> (x - m)^2 * f(x) / C
#     v  = float(call_approximate(method, q, vf))

#     return m, v
# end

# Specific tricks/implementations for distributions from Distributions.jl
include("gausslaguerre_gamma.jl")
# include("gausslaguerre_exponential.jl")
# include("gausslaguerre_nothing.jl")
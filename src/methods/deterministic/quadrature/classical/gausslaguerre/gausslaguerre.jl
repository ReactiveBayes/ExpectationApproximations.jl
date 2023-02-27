export GaussLaguerre

using FastGaussQuadrature

import Base: show

struct GaussLaguerre{P, W, L} <: ClassicalQuadratureRule
    length     :: Int
    points     :: P
    weights    :: W
    logweights :: L
end

name(::GaussLaguerre) = "GaussLaguerre"

function GaussLaguerre(length::Int)
    # α = 0.0 is on purpose as it is nicely computed and numerically stable
    # Revise `α` as a parameter later on, but we observed that with `α > 30.0` sigma points and weights are extremelly unstable
    points, weights = gausslaguerre(length, 0.0) 

    # Filter out zero weights
    zeroweights = map(<=(zero(eltype(weights))), weights)

    deleteat!(points, zeroweights)
    deleteat!(weights, zeroweights)

    logweights = map(log, weights)

    return GaussLaguerre(length, points, weights, logweights)
end

Base.length(method::GaussLaguerre)       = method.length
Base.show(io::IO, method::GaussLaguerre) = print(io, "GaussLaguerre(", length(method), ")")

evaluate(rule::GaussLaguerre, q) = rule # TODO revise, maybe `getpoints(rule, q)`?

getpoints(method::GaussLaguerre)     = method.points
getweights(method::GaussLaguerre)    = method.weights
getlogweights(method::GaussLaguerre) = method.logweights

## Gauss Laguerre without any distribution (::Nothing)

function adjust_integrand(::GaussLaguerre, ::Nothing, f::F) where F
    return (x) -> f(x) * exp(x)
end

function adjust_integrand(::GaussLaguerre, ::Nothing, f::LogIntegrand)
    return let logf = log(f)
        LogIntegrand((x) -> logf(x) + x)
    end
end

adjust_result(::GaussLaguerre, ::Nothing, result) = result   

## Gauss Laguerre with some arbitrary distribution (::Distribution)

function adjust_integrand(::GaussLaguerre, q::Distribution, f::F) where F 
    return (x) -> pdf(q, x) * f(x) * exp(x)
end

function adjust_integrand(::GaussLaguerre, q::Distribution, f::LogIntegrand) 
    return let logf = log(f)
        LogIntegrand((x) -> logpdf(q, x) + logf(x) + x)
    end
end

adjust_result(::GaussLaguerre, ::Distribution, result) = result    

# Specific tricks/implementations for distributions from Distributions.jl
include("gausslaguerre_gamma.jl")
include("gausslaguerre_exponential.jl")
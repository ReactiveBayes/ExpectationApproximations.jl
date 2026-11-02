export GaussHermite

using FastGaussQuadrature

import Base: show
import StatsFuns: sqrtπ, sqrt2

const invsqrtπ = inv(sqrtπ)
const logsqrtπ = log(sqrtπ)

struct GaussHermite{P, W, L} <: ClassicalQuadratureRule
    length::Int
    points::P
    weights::W
    logweights::L
end

name(::GaussHermite) = "GaussHermite"

function GaussHermite(length::Int)
    points, weights = gausshermite(length)

    # Filter out zero weights
    zeroweights = map(<=(zero(eltype(weights))), weights)

    deleteat!(points, zeroweights)
    deleteat!(weights, zeroweights)

    logweights = map(log, weights)

    return GaussHermite(length, points, weights, logweights)
end

Base.length(method::GaussHermite) = method.length
Base.show(io::IO, method::GaussHermite) = print(io, "GaussHermite(", length(method), ")")

evaluate(rule::GaussHermite, q) = rule # TODO revise, maybe `getpoints(rule, q)`?

getpoints(method::GaussHermite) = method.points
getweights(method::GaussHermite) = method.weights
getlogweights(method::GaussHermite) = method.logweights

## Gauss Hermite without any distribution (::Nothing)

function adjust_integrand(::GaussHermite, ::Nothing, f::F) where {F}
    return (x) -> f(x) * exp(x^2)
end

function adjust_integrand(::GaussHermite, ::Nothing, f::LogIntegrand)
    return let logf = log(f)
        LogIntegrand((x) -> logf(x) + x^2)
    end
end

adjust_result(::GaussHermite, ::Nothing, result) = result

## Gauss Hermite with some arbitrary distribution (::Distribution)

function adjust_integrand(::GaussHermite, q::Distribution, f::F) where {F}
    return (x) -> pdf(q, x) * f(x) * exp(x^2)
end

function adjust_integrand(::GaussHermite, q::Distribution, f::LogIntegrand)
    return let logf = log(f)
        LogIntegrand((x) -> logpdf(q, x) + logf(x) + x^2)
    end
end

adjust_result(::GaussHermite, ::Distribution, result) = result

# Specific tricks/implementations for distributions from Distributions.jl
include("gausshermite_normal.jl")

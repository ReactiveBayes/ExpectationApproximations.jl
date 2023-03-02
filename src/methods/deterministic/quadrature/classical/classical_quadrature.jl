export ClassicalQuadrature, UseLogWeights, UseWeights

using FastGaussQuadrature
using Distributions
using LogarithmicNumbers

import Base: length, show
import LogExpFunctions: logsumexp
import SpecialFunctions: loggamma, gamma

abstract type ClassicalQuadratureRule end

struct ClassicalQuadratureStrategyAutomatic end

struct ClassicalQuadratureRuleAutomatic
    npoints::Int
end

struct UseWeights end
struct UseLogWeights end

struct ClassicalQuadrature{R <: ClassicalQuadratureRule, S} <: AbstractApproximationMethod
    rule     :: R
    strategy :: S
end

ClassicalQuadrature(npoints::Int) = ClassicalQuadrature(ClassicalQuadratureRuleAutomatic(npoints), ClassicalQuadratureStrategyAutomatic())
ClassicalQuadrature(rule::ClassicalQuadratureRule) = ClassicalQuadrature(rule, ClassicalQuadratureStrategyAutomatic())

Base.show(io::IO, method::ClassicalQuadrature) = print(io, "ClassicalQuadrature(", method.rule, ", ", method.strategy, ")")

getstrategy(method::ClassicalQuadrature, q, ::LogIntegrand) = UseLogWeights() # In case if input is a `LogIntegrand` we always use log weights
getstrategy(method::ClassicalQuadrature, q, f::F) where {F} = getstrategy(method.strategy, q, f)

getstrategy(strategy::Union{UseLogWeights, UseWeights}, q, f) = strategy
getstrategy(::ClassicalQuadratureStrategyAutomatic, q, f) = UseWeights()

getrule(method::ClassicalQuadrature) = method.rule

function approximate_expectation(method::ClassicalQuadrature, q, f::F) where {F}
    # compatibility = iscompatible(getrule(method), q)
    # TODO Check compatibility
    strategy = getstrategy(method, q, f)
    # TODO Check evaluate(rule, q) -> getpoints(rule, q)
    rule = evaluate(getrule(method), q)
    return ClassicalQuadrature_approximate_expectation(strategy, rule, q, f)
end

getpoints(method::ClassicalQuadrature)     = getpoints(method.rule)
getweights(method::ClassicalQuadrature)    = getweights(method.rule)
getlogweights(method::ClassicalQuadrature) = getlogweights(method.rule)

## Generic implementation

function ClassicalQuadrature_approximate_expectation(::UseWeights, rule, q, f::F) where {F}
    points  = getpoints(rule)
    weights = getweights(rule)
    g       = adjust_integrand(rule, q, f)

    result = Iterators.mapreduce(+, zip(points, weights)) do (x, w)
        return w * g(x)
    end

    return adjust_result(rule, q, result)
end

function ClassicalQuadrature_approximate_expectation(::UseLogWeights, rule, q, f::F) where {F}
    points     = getpoints(rule)
    logweights = getlogweights(rule)
    g          = adjust_integrand(rule, q, convert(LogIntegrand, f))
    logg       = log(g)

    # ln ( ∑ wi * g(xi) ) 
    # = ln ( ∑ exp( ln(wi) + logg(xi) ) )
    # = ln ( ∑ exp( yi ) )
    # = max(yi) + ln ( ∑ exp( yi - max(yi) ) )  <-- `logsumexp`

    # Iterators.map creates a lazy mapping
    logresult = Iterators.map(zip(points, logweights)) do (x, logw)
        # calculate the ln(wi) + logg(xi) terms
        return logw + logg(x)
    end

    logadjusted = adjust_result(rule, q, LogResult(logsumexp(logresult)))

    return convert(Logarithmic, logadjusted)
end

##

include("gausslaguerre/gausslaguerre.jl")
include("gausshermite/gausshermite.jl")

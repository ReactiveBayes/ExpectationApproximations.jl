export approximate
export Expectation, RawMoment, CentralMoment

using Distributions

# TODO revise later on
# struct Compatible end
# struct CompatibilityUnknown end

# iscompatible(method, q) = CompatibilityUnknown()

function adjust_logintegrand end
function adjust_integrand end

function adjust_logresult end
function adjust_result end

##

struct Expectation end

##

abstract type Moment end

struct RawMoment{N, C} <: Moment
    c :: C
end

RawMoment(::Val{N}, c::C) where { N, C } = RawMoment{N, C}(c)

adjust_logintegrand(moment::RawMoment{N}, logf) where { N } = (x) -> N * log(abs(x)) + logf(x) - log(moment.c)
adjust_integrand(moment::RawMoment{N}, f)       where { N } = (x) -> x^N * f(x) / moment.c

## 

struct CentralMoment{N, C, M} <: Moment
    c :: C
    μ :: M
end

CentralMoment(::Val{N}, c::C, μ::M) where {N, C, M} = CentralMoment{N, C, M}(c, μ)

adjust_logintegrand(moment::CentralMoment{N}, logf) where {N} = (x) -> N * log(abs(x - moment.μ)) + logf(x) - log(moment.c)
adjust_integrand(moment::CentralMoment{N}, f) where {N}       = (x) -> (x - moment.μ)^N * f(x) / moment.c

##

approximate(method, f)    = approximate(method, nothing, f)
approximate(method, q, f) = approximate(Expectation(), method, q, f)

approximate(method, q, f::Distribution) = approximate(method, q, LogFn((x) -> logpdf(f, x)))

approximate(::Expectation, method, q, f) = approximate_expectation(method, q, f)

approximate(moment::Moment, method, q, f)        = approximate(method, q, adjust_integrand(moment, f))
approximate(moment::Moment, method, q, f::LogFn) = approximate(method, q, LogFn(adjust_logintegrand(moment, log(f))))
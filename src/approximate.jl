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

## TODO 

adjust(some, f::F) where {F} = adjust_integrand(some, f)
adjust(some, logf::LogFn)    = LogFn(adjust_logintegrand(some, log(logf)))

##

struct Expectation end

##

abstract type Moment end

struct RawMoment{N, C} <: Moment
    c :: C
end

RawMoment(::Val{N}, c::C) where { N, C } = RawMoment{N, C}(c)

adjust_logintegrand(moment::RawMoment{N}, logf::F) where { N, F } = (x) -> N * log(abs(x)) + logf(x) - log(moment.c)
adjust_integrand(moment::RawMoment{N}, f::F)       where { N, F } = (x) -> x^N * f(x) / moment.c

## 

struct CentralMoment{N, C, M} <: Moment
    c :: C
    μ :: M
end

CentralMoment(::Val{N}, c::C, μ::M) where {N, C, M} = CentralMoment{N, C, M}(c, μ)

adjust_logintegrand(moment::CentralMoment{N}, logf::F) where {N, F} = (x) -> N * log(abs(x - moment.μ)) + logf(x) - log(moment.c)
adjust_integrand(moment::CentralMoment{N}, f::F) where {N, F}       = (x) -> (x - moment.μ)^N * f(x) / moment.c

##

as_integrand(f::F) where { F <: Function } = f
as_integrand(f::LogFn)                     = f
as_integrand(f::Distribution)              = LogFn((x) -> logpdf(f, x))

##

approximate(method, f::F) where F    = approximate(method, nothing, f)
approximate(method, q, f::F) where F = approximate(Expectation(), method, q, f)

approximate(::Expectation, method, q, f::F) where F  = approximate_expectation(method, q, as_integrand(f))
approximate(moment::Moment, method, q, f::F) where F = approximate_expectation(method, q, adjust(moment, as_integrand(f)))


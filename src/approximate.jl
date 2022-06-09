export approximate
export Expectation, RawMoment, CentralMoment

using Distributions

# TODO revise later on
# struct Compatible end
# struct CompatibilityUnknown end

# iscompatible(method, q) = CompatibilityUnknown()

function adjust_integrand end
function adjust_result end

##

struct Expectation end

##

abstract type Moment end

struct RawMoment{N, C} <: Moment
    c :: C
end

RawMoment(::Val{N}, c::C) where { N, C } = RawMoment{N, C}(c)

function adjust_integrand(moment::RawMoment{N}, f::F) where { N, F }
    return let c = float(moment.c)
        (x) -> x^N * f(x) / c
    end
end

function adjust_integrand(moment::RawMoment{N}, f::LogIntegrand) where { N } 
    return let logf = log(f), logC = float(log(moment.c))
        LogIntegrand((x) -> N * log(Complex(x)) + logf(x) - logC)
    end
end

## 

struct CentralMoment{N, C, M} <: Moment
    c :: C
    μ :: M
end

CentralMoment(::Val{N}, c::C, μ::M) where {N, C, M} = CentralMoment{N, C, M}(c, μ)

function adjust_integrand(moment::CentralMoment{N}, f::F) where {N, F} 
    return let μ = float(moment.μ), c = float(moment.c)
        (x) -> (x - μ)^N * f(x) / c
    end
end

function adjust_integrand(moment::CentralMoment{N}, f::LogIntegrand) where {N} 
    return let μ = float(moment.μ), logf = log(f), logC = float(log(moment.c))
        LogIntegrand((x) -> N * log(Complex(x - μ)) + logf(x) - logC)
    end
end

##

as_integrand(f)                            = f
as_integrand(f::F) where { F <: Function } = f
as_integrand(f::LogIntegrand)              = f
as_integrand(f::Distribution)              = LogIntegrand((x) -> logpdf(f, x))

##

approximate(method, f::F) where F    = approximate(method, nothing, f)
approximate(method, q, f::F) where F = approximate(Expectation(), method, q, f)

approximate(::Expectation, method, q, f::F) where F  = approximate_expectation(method, q, as_integrand(f))
approximate(moment::Moment, method, q, f::F) where F = approximate_expectation(method, q, adjust_integrand(moment, as_integrand(f)))


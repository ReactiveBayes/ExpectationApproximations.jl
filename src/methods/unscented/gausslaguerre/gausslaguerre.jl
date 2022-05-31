export GaussLaguerre

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

    nonzeroweights = map(>(zero(eltype(weights))), weights)

    keepat!(points, nonzeroweights)
    keepat!(weights, nonzeroweights)

    logweights = map(log, weights)

    return GaussLaguerre(Val{ForceLogComputations}(), length, points, weights, logweights)
end

Base.length(method::GaussLaguerre) = method.length

evaluate(rule::GaussLaguerre, q) = rule

function call_approximate(method::GaussLaguerre{ForceLogComputations}, q, f) where { ForceLogComputations }
    return call_approximate(Val{ForceLogComputations}(), method, q, f)
end

# TODO for testing pusposes only
function call_mean_var(method::GaussLaguerre, q::Any, f::LogFn)
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
function call_mean_var(method::GaussLaguerre, q::Any, f)

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

# Utility functions 

_gausslaguerre_force_log(f::LogFn) = log(f)
_gausslaguerre_force_log(f)        = (x) -> log(Complex(f(x)))

_gausslaguerre_is_complex(x::Number)  = false
_gausslaguerre_is_complex(x::Complex) = true

function _gausslaguerre_force_log_process_result(f, logresult)
    # In general case it might be that `logresult` is complex, but imaginary part should be almost zero
    # Otherwise it indicates that something went wrong
    result = exp(logresult)
    if _gausslaguerre_is_complex(result) && !(imag(result) < 1e-8)
        error("GaussLaguerre failed to compute an approximation, result is a complex number with non-zero imaginary part: $(result)")
    end
    return real(result)
end

function _gausslaguerre_force_log_process_result(::LogFn, logresult)
    # In case of LogFn we assume that `logresult` is not complex
    # Otherwise it indicates that something went wrong
    # return `logresult` in a form of the `Logarithmic` number from `LogarithmicNumbers.jl` package
    return exp(Logarithmic, logresult)
end

# Specific implementations
include("gausslaguerre_gamma.jl")
include("gausslaguerre_exponential.jl")
include("gausslaguerre_nothing.jl")
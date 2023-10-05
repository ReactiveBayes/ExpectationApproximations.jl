export LogIntegrand

using LogarithmicNumbers

import Base: iterate, IteratorSize, IteratorEltype, eltype, length, size, in, convert, log, float

"""
    LogIntegrand

# Fields
- `logf` - base callable object that represents "log"-version of integrand `f`, such that `f(x) = exp(logf(x))`. 

By default `LogIntegrand` exponentiates the result, use `log` function to get an access to the "log"-version of the function `f`.
For any callable object it is also possible to just call `convert(LogIntegrand, f)` (the result of the modified function might be a `Complex` number).
`convert` does nothing if `f` is already an instance of `LogIntegrand`.

```jldoctest logintegrand
julia> f = LogIntegrand((x) -> log(x));

julia> f(1.0)
1.0

julia> log(f)(1.0)
0.0
```
"""
struct LogIntegrand{F}
    callable::F
end

(f::LogIntegrand)(x) = exp(f.callable(x))

Base.log(f::LogIntegrand) = f.callable

Base.convert(::Type{LogIntegrand}, input::LogIntegrand) = input
Base.convert(::Type{LogIntegrand}, input::F) where {F}  = LogIntegrand((x) -> _convert_logintegrand_number(input(x)))

_convert_logintegrand_number(r::Number)      = log(Complex(r))
_convert_logintegrand_number(r::Logarithmic) = log(r)

"""
    LogResult

Internal structure for representing the result by storing its log. Should not be used outside of the library. 
For interfacing with outside world the library always returns `Logarithmic` from `LogarithmicNumbers.jl` as it supports broader class of operations.
`LogResult` structure is needed for multiple dispatch and for storing `Complex` numbers as `Logarihtmic` does not support it.

See also: [`LogIntegrand`](@ref)
"""
struct LogResult{R}
    log::R
end

Base.log(logresult::LogResult) = logresult.log

function Base.convert(::Type{Logarithmic}, result::LogResult)
    logresult = log(result)
    expimag   = real(exp(complex(0, imag(logresult)))) # Exponentiate imaginary part only 

    # Check sign of the result, slight deviations are acceptable
    if !(isapprox(abs(expimag), 1, rtol = 0.01, atol = 0.01))
        error("Failed to convert a result of the approximation, result is a complex number with non-zero imaginary part: $(logresult)")
    end

    return sign(expimag) * exp(Logarithmic, real(logresult))
end

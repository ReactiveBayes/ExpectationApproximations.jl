export LogIntegrand

using LogarithmicNumbers

import Base: iterate, IteratorSize, IteratorEltype, eltype, length, size, in, convert, log

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
    callable :: F
end

(f::LogIntegrand)(x) = exp(f.callable(x))

Base.log(f::LogIntegrand) = f.callable

Base.convert(::Type{LogIntegrand}, input::LogIntegrand) = input
Base.convert(::Type{LogIntegrand}, input::F) where F    = LogIntegrand(_convert_logintegrand(input))

_convert_logintegrand(f::LogIntegrand) = log(f)
_convert_logintegrand(f::F) where F    = (x) -> _convert_logintegrand_number(f(x))

_convert_logintegrand_number(r::Number)      = log(Complex(r))
_convert_logintegrand_number(r::Logarithmic) = log(r)

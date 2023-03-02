export RawMoment

struct RawMoment{N, C} <: Kernel
    c::C
end

RawMoment(::Val{N}, c::C) where {N, C} = RawMoment{N, C}(c)

function adjust_integrand(moment::RawMoment{N}, f::F) where {N, F}
    return let c = float(moment.c)
        (x) -> x^N * f(x) / c
    end
end

function adjust_integrand(moment::RawMoment{N}, f::LogIntegrand) where {N}
    return let logf = log(f), logC = float(log(moment.c))
        LogIntegrand((x) -> N * log(Complex(x)) + logf(x) - logC)
    end
end

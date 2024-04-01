export MeasureAdjustedRawMomentGenerator
using TensorOperations

struct MeasureAdjustedRawMomentGenerator{T, N, C} <: Kernel
    c::C
end

MeasureAdjustedRawMomentGenerator(::Val{N}, c::C) where {N, C} = MeasureAdjustedRawMomentGenerator{Univariate, N, C}(c)
MeasureAdjustedRawMomentGenerator(::Type{T}, ::Val{N}, c::C) where {T, N, C} =  MeasureAdjustedRawMomentGenerator{T, N, C}(c)

adjust_integrand(moment::MeasureAdjustedRawMomentGenerator, f::F) where {F} = adjust_integrand(moment, f, Val(false))

function adjust_integrand(moment::MeasureAdjustedRawMomentGenerator{T, N}, f::F, ::Val{V}) where {T <: Univariate, N, F, V}
    return let c = float(moment.c)
        (x) -> x^N * f(x) / c
    end
end

function adjust_integrand(moment::MeasureAdjustedRawMomentGenerator{T, N}, f::LogIntegrand) where {T <: Univariate, N}
    return let logf = log(f), logC = float(log(moment.c))
        LogIntegrand((x) -> N * log(Complex(x)) + logf(x) - logC)
    end
end

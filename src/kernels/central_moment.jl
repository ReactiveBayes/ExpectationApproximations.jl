export CentralMoment

struct CentralMoment{N, C, M} <: Kernel
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
export MeasureAdjustedMomentGenerator

struct MeasureAdjustedMomentGenerator{T, N, C, M} <: Kernel
    c::C
    μ::M
end

MeasureAdjustedMomentGenerator(c::C, μ::M) where {C, M} = MeasureAdjustedMomentGenerator(Val(1), c, μ)
MeasureAdjustedMomentGenerator(::Val{N}, c::C, μ::M) where {N, C, M} = MeasureAdjustedMomentGenerator{Univariate, N, C, M}(c, μ)
MeasureAdjustedMomentGenerator(::Type{T}, ::Val{N}, c::C, μ::M) where {T, N, C, M} = MeasureAdjustedMomentGenerator{T, N, C, M}(c, μ)

adjust_integrand(moment::MeasureAdjustedMomentGenerator, f::F) where {F} = adjust_integrand(moment, f, Val(false))

function adjust_integrand(moment::MeasureAdjustedMomentGenerator{T, N}, f::F, ::Val{V}) where {T <: Univariate, N, F, V}
    return let μ = float(moment.μ), c = float(moment.c)
        (x) -> (x - μ)^N * f(x) / c
    end
end

function adjust_integrand(moment::MeasureAdjustedMomentGenerator{T, 1}, f::F, ::Val{V}) where {T <: Multivariate, F, V}
    if V
        return let μ = float(moment.μ), c = float(moment.c)
            (x...) -> (reduce(vcat, x) - μ) * f(x...) / c
        end
    else
        return let μ = float(moment.μ), c = float(moment.c)
            (x) -> (x - μ) * f(x) / c
        end
    end
end

function adjust_integrand(moment::MeasureAdjustedMomentGenerator{T, 2}, f::F, ::Val{V}) where {T <: Multivariate, F, V}
    if V
        return let μ = float(moment.μ), c = float(moment.c)
            (x...) -> begin
                Σ = zeros(eltype(μ), length(μ), length(μ))
                ξ = (reduce(vcat, x) - μ)
                @tensor begin
                    Σ[i, j] = ξ[i] * ξ[j]
                end
                return Σ * f(x...) / c
            end
        end
    else
        return let μ = float(moment.μ), c = float(moment.c)
            (x) -> begin
                Σ = zeros(eltype(μ), length(μ), length(μ))
                ξ = (x - μ)
                @tensor begin
                    Σ[i, j] = ξ[i] * ξ[j]
                end
                return Σ * f(x) / c
            end
        end
    end
end

function adjust_integrand(moment::MeasureAdjustedMomentGenerator{T, N}, f::LogIntegrand) where {T <: Univariate, N}
    return let μ = float(moment.μ), logf = log(f), logC = float(log(moment.c))
        LogIntegrand((x) -> N * log(Complex(x - μ)) + logf(x) - logC)
    end
end

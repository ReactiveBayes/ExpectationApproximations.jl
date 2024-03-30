export MomentGenerator, DiagonalMomentGenerator

using TensorOperations

#T is the output variate type
#μ should be an array
struct MomentGenerator{T, N, M} <: Kernel
    μ::M
end

MomentGenerator(::Val{N}, μ::M) where {N, M} = MomentGenerator{Univariate, N, M}(μ)
MomentGenerator(::Type{T}, ::Val{N}, μ::M) where {T, N, M} = MomentGenerator{T,N,M}(μ)

adjust_integrand(moment::MomentGenerator{T, N}, f::F) where {N, T, F} = adjust_integrand(moment, f, Val(true))
function adjust_integrand(moment::MomentGenerator{T, N}, f::F, ::Val{V}) where {N, T<:Univariate, F, V}
    if V
        return let μ = float(moment.μ)
            (x...) -> (f(x...) - μ)^N 
        end
    else
        return let μ = float(moment.μ)
            (x) -> (f(x) - μ)^N 
        end
    end
end

function adjust_integrand(moment::MomentGenerator{T, 2}, f::F, ::Val{V}) where {T<:Multivariate, F, V}
    if V
        return let μ = float(moment.μ)
            (x...) -> begin
                Σ = zeros(eltype(μ), length(μ), length(μ))
                ξ = f(x...) - μ
                @tensor begin
                    Σ[i,j] = ξ[i]*ξ[j]
                end
                return Σ
            end
        end
    else
        return let μ = float(moment.μ)
            (x) -> begin
                Σ = zeros(eltype(μ), length(μ), length(μ))
                ξ = f(x) - μ
                @tensor begin
                    Σ[i,j] = ξ[i]*ξ[j]
                end
                return Σ
            end
        end
    end
end


function adjust_integrand(moment::MomentGenerator{T, 3}, f::F, ::Val{V}) where {T<:Multivariate, F, V}
    if V
        return let μ = float(moment.μ)
            (x...) -> begin
                Σ = zeros(eltype(μ), length(μ), length(μ), length(μ))
                ξ = f(x...) - μ
                @tensor begin
                    Σ[i,j,k] = ξ[i]*ξ[j]*ξ[k]
                end
                return Σ
            end
        end
    else
        return let μ = float(moment.μ)
            (x) -> begin
                Σ = zeros(eltype(μ), length(μ), length(μ), length(μ))
                ξ = f(x) - μ
                @tensor begin
                    Σ[i,j,k] = ξ[i]*ξ[j]*ξ[k]
                end
                return Σ
            end
        end
    end
end

struct DiagonalMomentGenerator{N, M} <: Kernel
    μ::M
end

DiagonalMomentGenerator(::Val{N}, μ::M) where {N, M} = DiagonalMomentGenerator{N, M}(μ)

adjust_integrand(moment::DiagonalMomentGenerator{N}, f::F) where {N, F} = adjust_integrand(moment, f, Val(true))
function adjust_integrand(moment::DiagonalMomentGenerator{N}, f::F, ::Val{V}) where {N, F, V}
    if V
        return let μ = float(moment.μ)
            (x...) ->  begin
                Σ = zeros(eltype(μ), length(μ))
                map!(y -> y^N, Σ, f(x...) - μ)
                return Σ
            end
        end
    else
        return let μ = float(moment.μ)
            (x) ->  begin
                Σ = zeros(eltype(μ), length(μ))
                map!(y -> y^N, Σ, f(x) - μ)
                return Σ
            end
        end
    end
end
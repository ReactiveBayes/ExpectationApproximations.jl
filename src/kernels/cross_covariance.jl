export CrossCovariance

using TensorOperations
struct CrossCovariance{T1,T2, M1, M2} <: Kernel
    μ1 :: M1
    μ2 :: M2
end

CrossCovariance(μ1::M1, μ2::M2) where {M1, M2} = CrossCovariance{Univariate, Univariate, M1, M2}(μ1, μ2)
CrossCovariance(::Type{T1}, ::Type{T2}, μ1::M1, μ2::M2) where {T1,T2,M1,M2} = CrossCovariance{T1,T2,M1,M2}(μ1,μ2)

adjust_integrand(moment::CrossCovariance{T1, T2}, f::F) where {T1, T2, F} = adjust_integrand(moment, f, Val(true))

function adjust_integrand(moment::CrossCovariance{T1, T2}, f::F, ::Val{V}) where {T1 <: Univariate, T2 <:Univariate, F, V}
    return let μ1 = float(moment.μ1), μ2 = float(moment.μ2)
        (x) -> (x - μ2) * (f(x) - μ1)
    end
end


function adjust_integrand(moment::CrossCovariance{T1, T2}, f::F, ::Val{V}) where {T1 <: Multivariate, T2 <:Univariate, F, V}
    return let μ1 = float(moment.μ1), μ2 = float(moment.μ2)
        (x) -> (x - μ2) * (f(x) - μ1)
    end
end

function adjust_integrand(moment::CrossCovariance{T1, T2}, f::F, ::Val{V}) where {T1 <: Univariate, T2 <: Multivariate, F, V} 
    if V
        return let μ1 = float(moment.μ1), μ2 = float(moment.μ2)
            (x...) -> (reduce(vcat,x) - μ2) * (f(x...) - μ1)
        end
    else
        return let μ1 = float(moment.μ1), μ2 = float(moment.μ2)
            (x) -> (x - μ2) * (f(x) - μ1)
        end
    end
end

function adjust_integrand(moment::CrossCovariance{T1, T2}, f::F, ::Val{V}) where {T1 <: Multivariate, T2 <: Multivariate, F, V}
    if V
        return let μ1 = float(moment.μ1), μ2 = float(moment.μ2)
            (x...) -> begin
                C = zeros(promote_type(eltype(μ1),eltype(μ2)), length(μ2), length(μ1))
                ξ = reduce(vcat, x) - μ2
                ϕ = f(x...) - μ1
                @tensor begin
                    C[i,j] = ξ[i] * ϕ[j]
                end
                return C
            end
        end
    else
        return let μ1 = float(moment.μ1), μ2 = float(moment.μ2)
            (x) -> begin
                C = zeros(promote_type(eltype(μ1),eltype(μ2)), length(μ2), length(μ1))
                ξ = x - μ2
                ϕ = f(x) - μ1
                @tensor begin
                    C[i,j] = ξ[i] * ϕ[j]
                end
                return C
            end
        end
    end
end




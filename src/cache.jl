struct ApproximationCache
    cache_tensors::Dict{Tuple{Symbol, Tuple{Int, Int, Int}}, Array{Float64, 3}}
    cache_matrices::Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}
    cache_vectors::Dict{Tuple{Symbol, Int}, Vector{Float64}}
end

ApproximationCache() = ApproximationCache(Dict{Tuple{Symbol, Tuple{Int, Int, Int}}, Array{Float64, 3}}(),Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}(), Dict{Tuple{Symbol, Int}, Vector{Float64}}())

function getcache(cache::ApproximationCache, label::Tuple{Symbol, Tuple{Int, Int, Int}})
    return get!(() -> Array{Float64}(undef, first(label[2]), getindex(label[2],2),last(label[2])), cache.cache_tensors, label)
end

function getcache(cache::ApproximationCache, label::Tuple{Symbol, Tuple{Int, Int}})
    return get!(() -> Matrix{Float64}(undef, label[2]), cache.cache_matrices, label)
end

function getcache(cache::ApproximationCache, label::Tuple{Symbol, Int})
    return get!(() -> Vector{Float64}(undef, label[2]), cache.cache_vectors, label)
end

function getpointscachewithzeros(cache::ApproximationCache, label::Tuple{Symbol, Tuple{Int,Int}})
    return get!(() -> Zeros(label[2]), cache.cache_matrices, label)
end

function getweightscacewithzeros(cache::ApproximationCache, label::Tuple{Symbol, Int})
    return get!(() -> Zeros(label[2]), cache.cache_vectors, label)
end


function getunscentedtransformedpoints(::Type{FromTo{Tin, Tout}}, method::AbstractApproximationMethod) where {Tin, Tout}
    getunscentedtransformedpoints(Tout, method.cache, method.inputdim, method.outputdim)
end

getunscentedtransformedpoints(::Type{T}, cache::ApproximationCache, id, _) where {T <: Univariate} = getcache(cache, (:transformedpoints, 2*id+1))
getunscentedtransformedpoints(::Type{T}, cache::ApproximationCache, id, od) where {T <: Multivariate} = getcache(cache, (:transformedpoints, (first(od), 2*id+1)))
getunscentedtransformedpoints(::Type{T}, cache::ApproximationCache, id, od) where {T <: Matrixvariate} = getcache(cache, (:transformedpoints, (first(od), last(od), 2*id+1)))

export GenUnscented, GenUT, GenUnscentedTransform
using BlockArrays, BlockDiagonals
using  KWLinalg,TensorCore
import FillArrays: Eye, Zeros

struct GenUnscentedCache
    cache_tensors::Dict{Tuple{Symbol, Tuple{Int, Int, Int}}, Array{Float64, 3}}
    cache_matrices::Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}
    cache_vectors::Dict{Tuple{Symbol, Int}, Vector{Float64}}
end

GenUnscentedCache() = GenUnscentedCache(Dict{Tuple{Symbol, Tuple{Int, Int, Int}}, Array{Float64, 3}}(),Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}(), Dict{Tuple{Symbol, Int}, Vector{Float64}}())

function getcache(cache::GenUnscentedCache, label::Tuple{Symbol, Tuple{Int, Int, Int}})
    return get!(() -> Array{Float64}(undef, first(label[2]), getindex(label[2],2),last(label[2])), cache.cache_tensors, label)
end

function getcache(cache::GenUnscentedCache, label::Tuple{Symbol, Tuple{Int, Int}})
    return get!(() -> Matrix{Float64}(undef, label[2]), cache.cache_matrices, label)
end

function getcache(cache::GenUnscentedCache, label::Tuple{Symbol, Int})
    return get!(() -> Vector{Float64}(undef, label[2]), cache.cache_vectors, label)
end

function getpointscachewithzeros(cache::GenUnscentedCache, label::Tuple{Symbol, Tuple{Int,Int}})
    return get!(() -> Zeros(label[2]), cache.cache_matrices, label)
end

function getweightscacewithzeros(cache::GenUnscentedCache, label::Tuple{Symbol, Int})
    return get!(() -> Zeros(label[2]), cache.cache_vectors, label)
end

"""
The `GenUnscented` structure defines the approximation method of Generalized Unscented Transform. 
"""
struct GenUnscented{Tout <: VariateForm, Tin <: VariateForm} <: AbstractApproximationMethod 
    inputdim :: Int
    outputdim :: Int
    cache :: GenUnscentedCache
end

function GenUnscented{Tout,Tin}(id::Int,od::Int) where {Tout <: VariateForm, Tin <: VariateForm} 
    GenUnscented{Tout,Tin}(id,od, GenUnscentedCache())
end
GenUnscented() = GenUnscented{Univariate, Unviariate}(1,1)

"""An alias for the [`GenUnscented`](@ref) approximation method."""
const GenUT = GenUnscented

"""An alias for the [`GenUnscented`](@ref) approximation method."""
const GenUnscentedTransform = GenUnscented

function mul_A_B_At!(cache::GenUnscentedCache, A, B)
    AB = getcache(cache, (:mul_A_B_At!_AB, (size(A, 1), size(B, 2))))
    ABAt = getcache(cache, (:mul_A_B_At!_ABAt, (size(AB, 1), size(A, 1))))
    mul!(AB, A, B)
    mul!(ABAt, AB, A')

    return ABAt
end

function copy_and_svd!(A, AC, f)
    AC .= A
    f(AC)
end

function prepare_cache!(cache::GenUnscentedCache, V::AbstractMatrix, S::AbstractVector, K::AbstractVector) 
    Vsvd = getcache(cache, (:Vsvd, size(V)))
    Ssvd = getcache(cache, (:Ssvd, size(V, 1)))
    f  = svd_functor_divconquer(size(V, 1),size(V, 1),Float64)
    svd_object = copy_and_svd!(V, Vsvd, f)
  
    map!(sqrt, Ssvd, svd_object.S)
    mul_A_B_At!(cache, svd_object.U, Diagonal(Ssvd))

    L3 = getcache(cache, (:L3, size(V)))
    hadamard!(L3, getcache(cache, (:mul_A_B_At!_ABAt, size(V))),getcache(cache, (:mul_A_B_At!_ABAt, size(V))))
    hadamard!(L3, getcache(cache, (:mul_A_B_At!_ABAt, size(V))), L3 )

    L4 = getcache(cache, (:L4, size(V)) )
    hadamard!(L4, L3, getcache(cache, (:mul_A_B_At!_ABAt, size(V))))
    
    L3S = getcache(cache, (:L3S, size(S,1)))
    ldiv!(L3S, fastcholesky!(L3), S)

    minK = getcache(cache, (:minK, size(K,1)))
    L3S2 = getcache(cache, (:L3S2, size(S,1)))
    mul!(minK, L4 , hadamard!(L3S2,L3S,L3S))

    any(K .< minK) ? K[K .<= minK] .= 1.001 * (minK[K .< minK] .+ 1e-8) : K

    L4K = getcache(cache, (:L4K, size(K,1)))
    ldiv!(L4K, fastcholesky!(L4), K)

    u = getcache(cache, (:u, size(K,1)))
    v = getcache(cache, (:v, size(K,1)))
    map!(sqrt, L3S2, mul!(L3S2, Eye{eltype(L3S2)}(size(V,1)), L4K, 4, -3))
    map!((x,y) -> (-x+y)/2, u, L3S, L3S2)
    map!((x,y) -> x+y, v, u, L3S)

    return nothing
end

function form_generalized_sigma_points!(cache::GenUnscentedCache, m::AbstractVector)
    d = length(m)
    weights = getcache(cache, (:weights, 2d + 1)) 
    points  = getcache(cache, (:points, (d, 2d + 1)))
    L = getcache(cache, (:mul_A_B_At!_ABAt, (d,d)))
    u = getcache(cache, (:u, d))
    v = getcache(cache, (:v, d))
    
    points[:, 1] = m
    @inbounds for (u_scal,v_scal, L_vec, i) in zip(u, v, eachcol(L), 1:d)
        map!((L_scal, j) -> m[j] - u_scal * L_scal, view(points, :, i+1), L_vec, 1:d)
        map!((L_scal, j) -> m[j] + v_scal * L_scal, view(points, :, d+i+1), L_vec, 1:d)
    end
     
    map!((x,y) -> inv(y)/(y+x), view(weights, d+2:2d+1) ,u,v)
    map!((x,y, z) -> z*x/y, view(weights, 2:d+1),v,u,view(weights, d+2:2d+1) )

    weights[1] = 1 - sum(@views weights[2:end])

    return nothing
end

#Multivariate sigma points and weights
function sigma_points_weights!(approximation::GenUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector)
    gen_unscented_cache = approximation.cache 
    prepare_cache!(gen_unscented_cache, V, S, K)
    form_generalized_sigma_points!(gen_unscented_cache, m)

    return nothing
end

#Univariate sigma points and weights
function sigma_points_weights!(method::GenUnscented, m::Real, V::Real,S::Real,K::Real)
    L     = sqrt(V)
    invL3 = inv(L^3) 
    minK  = L^4*(S*invL3)^2
    K     = (K <= minK) ? 1.001minK : K
    u     = (1/2)*(-S*invL3 + (1/V)*sqrt(4*K - 3*(S^2)/V))
    v     = u + S*invL3
   
    cache = method.cache
    weights = getcache(cache, (:weights, 3)) 
    points  = getcache(cache, (:points, (1, 3)))
    @views begin
        points[1,1] = m
        points[1,2] = m - u*L
        points[1,3] = m + v*L
        weights[3] = inv(v*(u+v))
        weights[2] = (v/u)*weights[3]
        weights[1] = 1 - weights[3] - weights[2]
    end

    return nothing
end

function approximate_expectation(method::GenUnscented, q, f::F) where {F}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    approximate_expectation(out_variate_type, in_variate_type, method, q, f, Val(false))
end

function approximate_expectation(method::GenUnscented, q, f::F, ::Val{C}) where {F, C}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    approximate_expectation(out_variate_type, in_variate_type, method, q, f, Val(C))
end

function approximate_expectation(method::GenUnscented, m, V, S, K, f::F) where {F}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    approximate_expectation(out_variate_type, in_variate_type, method, m, V, S, K, f, Val(false))
end

function approximate_expectation(method::GenUnscented, m, V, S, K, f::F, ::Val{C}) where {F, C}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    approximate_expectation(out_variate_type, in_variate_type, method, m, V, S, K, f, Val(C))
end

##Univariate-Univariate - single input
function approximate_expectation(::Type{T}, ::Type{T}, method::GenUnscented, q, f::F,::Val{C}) where {T <: Univariate, F, C}
    if !C
        m = mean(q)
        V = _cov(q)
        S = _skewness(q)
        K = _kurtosis(q)
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache, (:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, 2method.inputdim+1)) 
    map!((x,y) -> y*f(x), transformedpoints, points, weights)

    return sum(transformedpoints)
end

function approximate_expectation(::Type{T}, ::Type{T}, method::GenUnscented, m::Real, V::Real, S::Real ,K::Real, f::F,::Val{C}) where {T <: Univariate, F, C}
    if !C
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache, (:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, 2method.inputdim+1)) 
    map!((x,y) -> y*f(x), transformedpoints, points, weights)

    return sum(transformedpoints)
end

# Multivariate-Univariate -single input
function approximate_expectation(::Type{T1}, ::Type{T2},method::GenUnscented, q, f::F, ::Val{C}) where {T1 <: Univariate, T2 <: Multivariate, F, C}
    if !C  
        m = mean(q)
        V = _cov(q)
        S = _skewness(q)
        K = _kurtosis(q)
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, 2method.inputdim+1)) 
    map!((x,y) -> y*f(x), transformedpoints, eachcol(points), weights)

    return sum(transformedpoints)
end

function approximate_expectation(::Type{T1}, ::Type{T2},method::GenUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector, f::F, ::Val{C}) where {T1 <: Univariate, T2 <: Multivariate, F, C}
    if !C  
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, 2method.inputdim+1)) 
    map!((x,y) -> y*f(x), transformedpoints, eachcol(points), weights)

    return sum(transformedpoints)
end

# Univariate-Multivariate -single input
function approximate_expectation(::Type{T1}, ::Type{T2},method::GenUnscented, q, f::F, ::Val{C}) where {T1 <: Multivariate, T2<:Univariate, F, C}
    if !C 
        m = mean(q)
        V = _cov(q)
        S = _skewness(q)
        K = _kurtosis(q)
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim,2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(x)), eachcol(transformedpoints), points, weights)
    
    result = zeros(method.outputdim)
    for elem = 1:method.outputdim
        result[elem] = sum(@views transformedpoints[elem, :])
    end
    
    return result
end

function approximate_expectation(::Type{T1}, ::Type{T2},method::GenUnscented, m::Real, V::Real, S::Real, K::Real, f::F, ::Val{C}) where {T1 <: Multivariate, T2<:Univariate, F, C}
    if !C 
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim,2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(x)), eachcol(transformedpoints), points, weights)
    
    result = zeros(method.outputdim)
    for elem = 1:method.outputdim
        result[elem] = sum(@views transformedpoints[elem, :])
    end
    
    return result
end

# Univariate-Matrixvariate -single input
function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenUnscented, q, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2<:Univariate, F, C}
    if !C
        m = mean(q)
        V = _cov(q)
        S = _skewness(q)
        K = _kurtosis(q)
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim, method.outputdim, 2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(x)), eachslice(transformedpoints, dims=3), points, weights)
    
    result = zeros(method.outputdim, method.outputdim)

   for i=1:method.outputdim
        for j=1:method.outputdim
            result[i,j] = sum(@views transformedpoints[i, j,:])
        end
    end

    return result
end

function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenUnscented, m::Real, V::Real, S::Real, K::Real, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2<:Univariate, F, C}
    if !C
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim, method.outputdim, 2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(x)), eachslice(transformedpoints, dims=3), points, weights)
    
    result = zeros(method.outputdim, method.outputdim)

   for i=1:method.outputdim
        for j=1:method.outputdim
            result[i,j] = sum(@views transformedpoints[i, j,:])
        end
    end

    return result
end

# Multivariate-Matrixvariate -single input
function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenUnscented, q, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2 <: Multivariate, F, C}
    if !C 
        m = mean(q)
        V = _cov(q)
        S = _skewness(q)
        K = _kurtosis(q)
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim, method.outputdim,2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(x)), eachslice(transformedpoints,dims=3), eachcol(points), weights)

    result = zeros(method.outputdim, method.outputdim)
    for i=1:method.outputdim
        for j=1:method.outputdim
            result[i,j] = sum(@views transformedpoints[i,j,:])
        end
    end

    return result   
end

function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenUnscented, m::AbstractVector,V::AbstractMatrix,S::AbstractVector,K::AbstractVector, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2 <: Multivariate, F, C}
    if !C 
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim, method.outputdim,2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(x)), eachslice(transformedpoints,dims=3), eachcol(points), weights)

    result = zeros(method.outputdim, method.outputdim)
    for i=1:method.outputdim
        for j=1:method.outputdim
            result[i,j] = sum(@views transformedpoints[i,j,:])
        end
    end

    return result   
end

#Multivariate - Multivariate -single input
function approximate_expectation(::Type{T}, ::Type{T}, method::GenUnscented, q, f::F, ::Val{C}) where {T <: Multivariate, F, C}
    if !C
        m = mean(q)
        V = _cov(q)
        S = _skewness(q)
        K = _kurtosis(q)
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim, 2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(x)), eachcol(transformedpoints), eachcol(points), weights)

    result = zeros(method.outputdim)
    for elem = 1:method.outputdim
        result[elem] = sum(@views transformedpoints[elem, :])
    end
    
    return result
end

function approximate_expectation(::Type{T}, ::Type{T}, method::GenUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector, f::F, ::Val{C}) where {T <: Multivariate, F, C}
    if !C
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getcache(cache, (:weights, 2method.inputdim + 1))
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim, 2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(x)), eachcol(transformedpoints), eachcol(points), weights)

    result = zeros(method.outputdim)
    for elem = 1:method.outputdim
        result[elem] = sum(@views transformedpoints[elem, :])
    end
    
    return result
end

## Univariate-Univariate tuple input
function approximate_expectation(::Type{T}, ::Type{T}, method::GenUnscented, q::Tuple, f::F, ::Val{C}) where {T <: Univariate, F, C}
    ms = map(mean,q)
    ds = size.(ms)
    if !C    
        m = first(ms)
        S = _skewness(first(q))
        K = _kurtosis(first(q))
        V = var(first(q))
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache 
    points = getcache(cache,(:points, 2method.inputdim+1))
    weights = getcache(cache, (:weights, 2method.inputdim+1))
    transformedpoints = getcache(cache, (:transformedpoints, 2method.inputdim+1))
    map!((x,y) -> y*f(__splitjoin(x, ds)...), transformedpoints, points, weights)

    return sum(transformedpoints)
end

function approximate_expectation(::Type{T}, ::Type{T}, method::GenUnscented, m::Tuple, V::Tuple, S::Tuple, K::Tuple, f::F, ::Val{C}) where {T <: Univariate, F, C}
    ds = size.(m)
    if !C    
        sigma_points_weights!(method, first(m), first(V), first(S), first(K))
    end
    cache = method.cache 
    points = getcache(cache,(:points, 2method.inputdim+1))
    weights = getcache(cache, (:weights, 2method.inputdim+1))
    transformedpoints = getcache(cache, (:transformedpoints, 2method.inputdim+1))
    map!((x,y) -> y*f(__splitjoin(x, ds)...), transformedpoints, points, weights)

    return sum(transformedpoints)
end

## Multivariate-Univariate tuple input
function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenUnscented, q::Tuple, f::F, ::Val{C}) where {T1 <: Univariate, T2<: Multivariate, F, C}
    ms = map(mean,q)
    ds = size.(ms)
    if !C    
        m = reduce(vcat, ms)
        S = reduce(vcat, map(_skewness, q))
        K = reduce(vcat, map(_kurtosis, q))
        V = BlockDiagonal(collect(map(x -> typeof(x) <: Matrix ? x : [x;;], map(_cov,q))))
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache 
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim+1)))
    weights = getcache(cache, (:weights, 2method.inputdim+1))
    transformedpoints = getcache(cache, (:transformedpoints, 2method.inputdim+1))
    map!((x,y) -> y*f(__splitjoin(x, ds)...), transformedpoints, eachcol(points), weights)

    return sum(transformedpoints)
end

function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenUnscented, ms::Tuple, Vs::Tuple, Ss::Tuple, Ks::Tuple, f::F, ::Val{C}) where {T1 <: Univariate, T2<: Multivariate, F, C}
    ds = size.(ms)
    if !C    
        m = reduce(vcat, ms)
        S = reduce(vcat, Ss)
        K = reduce(vcat, Ks)
        V = BlockDiagonal(collect(map(x -> typeof(x) <: Matrix ? x : [x;;], Vs)))
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache 
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim+1)))
    weights = getcache(cache, (:weights, 2method.inputdim+1))
    transformedpoints = getcache(cache, (:transformedpoints, 2method.inputdim+1))
    map!((x,y) -> y*f(__splitjoin(x, ds)...), transformedpoints, eachcol(points), weights)

    return sum(transformedpoints)
end

## Multivariate-Multivariate tuple input
function approximate_expectation(::Type{T},::Type{T}, method::GenUnscented, q::Tuple, f::F, ::Val{C}) where {T <: Multivariate, F, C}
    ms = map(mean,q)
    ds = size.(ms)
    if !C
        m = reduce(vcat, ms)
        S = reduce(vcat, map(_skewness, q))
        K = reduce(vcat, map(_kurtosis, q))
        V = BlockDiagonal(collect(map(x -> typeof(x) <: Matrix ? x : [x;;], map(_cov,q))))
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache   
    points = getcache(cache,(:points, (method.inputdim, 2*method.inputdim+1)))
    weights = getcache(cache,(:weights, 2*method.inputdim+1))
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim, 2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(__splitjoin(x, ds)...)), eachcol(transformedpoints), eachcol(points), weights)
    result = zeros(method.outputdim)
    for elem = 1:method.outputdim
        result[elem] = sum(@views transformedpoints[elem, :])
    end

    return result
end

function approximate_expectation(::Type{T},::Type{T}, method::GenUnscented, ms::Tuple, Vs::Tuple, Ss::Tuple, Ks::Tuple, f::F, ::Val{C}) where {T <: Multivariate, F, C}
    ds = size.(ms)
    if !C
        m = reduce(vcat, ms)
        S = reduce(vcat, Ss)
        K = reduce(vcat, Ks)
        V = BlockDiagonal(collect(map(x -> typeof(x) <: Matrix ? x : [x;;], Vs)))
        sigma_points_weights!(method, m, V, S, K)
    end
    cache = method.cache   
    points = getcache(cache,(:points, (method.inputdim, 2*method.inputdim+1)))
    weights = getcache(cache,(:weights, 2*method.inputdim+1))
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim, 2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(__splitjoin(x, ds)...)), eachcol(transformedpoints), eachcol(points), weights)
    result = zeros(method.outputdim)
    for elem = 1:method.outputdim
        result[elem] = sum(@views transformedpoints[elem, :])
    end

    return result
end

## Multivariate-Matrixvariate tuple input
function approximate_expectation(::Type{T1},::Type{T2}, method::GenUnscented, q::Tuple, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2 <: Multivariate, F, C}
    ms = map(mean,q)
    ds = size.(ms)

    if !C
        m = reduce(vcat, ms)
        S = reduce(vcat, map(_skewness, q))
        K = reduce(vcat, map(_kurtosis, q))
        V = BlockDiagonal(collect(map(x -> typeof(x) <: Matrix ? x : [x;;], map(_cov,q))))
        sigma_points_weights!(method, m, V, S, K)
    end

    cache = method.cache 
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim+1)))
    weights = getcache(cache, (:weights, 2method.inputdim+1))
  
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim, method.outputdim, 2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(__splitjoin(x, ds)...)), eachslice(transformedpoints, dims=3),eachcol(points), weights)
    result = zeros(method.outputdim,method.outputdim)
    for i=1:method.outputdim
        for j=1:method.outputdim
            result[i,j] = sum(@views transformedpoints[i,j,:])
        end
    end

    return result   
end

function approximate_expectation(::Type{T1},::Type{T2}, method::GenUnscented, ms::Tuple, Vs::Tuple, Ss::Tuple, Ks::Tuple, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2 <: Multivariate, F, C}
    ds = size.(ms)
    if !C
        m = reduce(vcat, ms)
        S = reduce(vcat, Ss)
        K = reduce(vcat, Ks)
        V = BlockDiagonal(collect(map(x -> typeof(x) <: Matrix ? x : [x;;], Vs)))
        sigma_points_weights!(method, m, V, S, K)
    end

    cache = method.cache 
    points = getcache(cache,(:points, (method.inputdim, 2method.inputdim+1)))
    weights = getcache(cache, (:weights, 2method.inputdim+1))
  
    transformedpoints = getcache(cache, (:transformedpoints, (method.outputdim, method.outputdim, 2method.inputdim+1)))
    map!((x,y) -> lmul!(y, f(__splitjoin(x, ds)...)), eachslice(transformedpoints, dims=3),eachcol(points), weights)
    result = zeros(method.outputdim,method.outputdim)
    for i=1:method.outputdim
        for j=1:method.outputdim
            result[i,j] = sum(@views transformedpoints[i,j,:])
        end
    end

    return result   
end

# Unscented_approximate functions 
function Unscented_approximate(method::GenUnscented, q, g::G) where {G}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    unscented_statistics(out_variate_type, in_variate_type, method, Val(false), q, g)
end

function Unscented_approximate(method::GenUnscented, m, V, S, K, g::G) where {G}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    unscented_statistics(out_variate_type, in_variate_type, method,Val(false), m, V, S, K, g)
end

# unscented_statistics functions
function unscented_statistics(method::GenUnscented, q, g::G) where {G}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)
    return unscented_statistics(out_variate_type, in_variate_type, method, Val(true), q, g)
end

function unscented_statistics(method::GenUnscented, m,V,S,K, g::G) where {G}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)
    return unscented_statistics(out_variate_type, in_variate_type, method,m, V, S, K, Val(true), g)
end

# Univariate in Univariate out single input
function unscented_statistics(out_variate::Type{T},in_variate::Type{T}, method::GenUnscented, ::Val{C}, q, g::G) where {T<:Univariate, C, G}
    mresult = approximate_expectation(method, q, g)
    Vresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(mresult, mean(q)), g), Val(true)) : nothing
    return mresult,Vresult,Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T},in_variate::Type{T}, method::GenUnscented, ::Val{C}, m::Real,V::Real,S::Real,K::Real, g::G) where {T<:Univariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(mresult, m), g), Val(true)) : nothing
    return mresult,Vresult,Sresult, Kresult, Cresult
end

# Univariate in Multivariate out single input 
function unscented_statistics(out_variate::Type{T1},in_variate::Type{T2}, method::GenUnscented, ::Val{C}, q, g::G) where {T1<:Multivariate, T2<:Univariate, C, G}
    mresult = approximate_expectation(method, q, g)
    Vresult = approximate_expectation(Matrixvariate, Univariate, method, q, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true)) 
    Sresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(Multivariate, Univariate, mresult, mean(q)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T1},in_variate::Type{T2}, method::GenUnscented, ::Val{C}, m::Real, V::Real, S::Real, K::Real, g::G) where {T1<:Multivariate, T2<:Univariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(Matrixvariate, Univariate, method, m, V, S, K, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true)) 
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Multivariate, Univariate, mresult, m), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

# Multivariate in Univariate out - single input
function unscented_statistics(out_variate::Type{T1},in_variate::Type{T2}, method::GenUnscented, ::Val{C}, q, g::G) where {T1<:Univariate, T2<:Multivariate, C, G}
    mresult = approximate_expectation(method, q, g)   
    Vresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(Univariate, Multivariate, mresult, mean(q)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T1},in_variate::Type{T2}, method::GenUnscented, ::Val{C}, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector, g::G) where {T1<:Univariate, T2<:Multivariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)   
    Vresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Univariate, Multivariate, mresult, m), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

#Multivariate in Multivariate out - single input
function unscented_statistics(out_variate::Type{T},in_variate::Type{T}, method::GenUnscented, ::Val{C}, q, g::G) where {T <: Multivariate, C, G}
    mresult = approximate_expectation(method, q, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, q, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))
    
    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, mean(q)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T},in_variate::Type{T}, method::GenUnscented, ::Val{C}, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector, g::G) where {T <: Multivariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, m, V, S, K, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))
    
    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, m), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

#Univariate in Univariate out with q tuple
function unscented_statistics(out_variate::Type{T}, in_variate::Type{T}, method::GenUnscented, ::Val{C}, q::Tuple, g::G) where { T <: Univariate, C, G}
    mresult = approximate_expectation(method, first(q), g)
    Vresult = approximate_expectation(method, first(q), adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, first(q), adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, first(q), adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, first(q), adjust_integrand(CrossCovariance(mresult, mean(first(q))), g), Val(true)) : nothing
    return mresult,Vresult,Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T}, in_variate::Type{T}, method::GenUnscented, ::Val{C}, m::Tuple, V::Tuple, S::Tuple, K::Tuple, g::G) where { T <: Univariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, first(q), adjust_integrand(CrossCovariance(mresult, m), g), Val(true)) : nothing
    return mresult,Vresult,Sresult, Kresult, Cresult
end

#Univariate in Multivariate out with q tuple
function unscented_statistics(out_variate::Type{T1}, in_variate::Type{T2}, method::GenUnscented, ::Val{C}, q::Tuple, g::G) where { T1 <: Multivariate, T2 <: Univariate, C, G}
    mresult = approximate_expectation(method, first(q), g)
    Vresult = approximate_expectation(Matrixvariate, Univariate, method, first(q), adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, first(q), adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, first(q), adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, first(q), adjust_integrand(CrossCovariance(Multivariate, Univariate, mresult, mean(first(q))), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T1}, in_variate::Type{T2}, method::GenUnscented, ::Val{C}, m::Tuple, V::Tuple, S::Tuple, K::Tuple, g::G) where { T1 <: Multivariate, T2 <: Univariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(Matrixvariate, Univariate, method, m, V, S, K, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Multivariate, Univariate, mresult, first(m)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

#Multivariate in Univariate out with q tuple
function unscented_statistics(out_variate::Type{T1}, in_variate::Type{T2}, method::GenUnscented, ::Val{C}, q::Tuple, g::G) where { T1 <: Univariate, T2 <: Multivariate, C, G}
    mresult = approximate_expectation(method, q, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, q, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, mapreduce(mean, vcat, q)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T1}, in_variate::Type{T2}, method::GenUnscented, ::Val{C}, m::Tuple, V::Tuple, S::Tuple, K::Tuple, g::G) where { T1 <: Univariate, T2 <: Multivariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, m, V, S, K, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, reduce(vcat, m)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end
#Multivariate in Multivariate out with q tuple
function unscented_statistics(out_variate::Type{T}, in_variate::Type{T}, method::GenUnscented, ::Val{C}, q::Tuple, g::G) where { T <: Multivariate, C, G}
    mresult = approximate_expectation(method, q, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, q, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, mapreduce(mean, vcat, q)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T}, in_variate::Type{T}, method::GenUnscented, ::Val{C}, m::Tuple, V::Tuple, S::Tuple, K::Tuple, g::G) where { T <: Multivariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, m, V, S, K, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, reduce(vcat, m)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

#approximate function if every distribution is Gaussian
function approximate(method::GenUnscented, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}

    μ_tilde, Σ_tilde,_,_,_ = Unscented_approximate(method, distributions, f)
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    return convert(ExponentialFamily.promote_variate_type( out_variate_type, NormalMeanVariance), μ_tilde, Σ_tilde)
end

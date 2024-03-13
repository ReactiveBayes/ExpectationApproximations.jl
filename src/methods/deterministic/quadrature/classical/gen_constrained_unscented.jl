export GenConstrainedUnscented, GenCUT, GenConstrainedUnscentedTransform
using BlockArrays, BlockDiagonals
using  KWLinalg,TensorCore

const DEFAULT_Θ = 0.5
"""
The `GenConstrainedUnscented` structure defines the approximation method of Generalized Constrained Unscented Transform.
# Arguments
- `lb`: lower bound on the sigma points #1
- `ub`: upper bound on the sigma points #2
lb = (1,) will imply univariate 
ub = (3,)
lb = ([2,1], 3, 4) will imply first variable is 2 two dimensional, 2nd and 3rd variable are univariate
ub = ([3,5], 5, 8)

The `GenConstrainedUnscented` structure with default parameters can be constructed as `GenConstrainedUnscented()`.
"""
struct GenConstrainedUnscented{Tout<:VariateForm, Tin <:VariateForm, T1,T2, R} <: AbstractApproximationMethod
    lb::T1
    ub::T2
    inputdim::Int
    outputdim::Int
    θ::R
    cache::GenUnscentedCache
end

function GenConstrainedUnscented{Tout, Tin}(lb::T1, ub::T2, id::Int, od::Int, θ::R = DEFAULT_Θ)  where {Tout <:VariateForm, Tin<:VariateForm, T1,T2, R}
    @assert size.(lb) == size.(ub) "Upper bound $(ub) and lower bound $(lb) are not of same size"
    @assert sum(length.(lb)) == id "Sum of the length of upper bound $(ub) and lower bound $(lb) elemets should add up to inputdim"
    @assert all(map(d -> all(first(d) .< getindex(d,2)), zip(lb,ub)))  "Upper bound elements $(ub) should be larger than lower bound elements $(lb) "
    @assert 0 < θ < 1 "Free parameter θ=$(θ) should be in between 0 and 1"
  
    return GenConstrainedUnscented{Tout, Tin, T1, T2, R}(lb, ub, id, od, θ, GenUnscentedCache())
end

GenConstrainedUnscented(lb::T1,ub::T2,θ::R = DEFAULT_Θ) where {T1, T2, R} = GenConstrainedUnscented{Univariate,Univariate}(lb,ub,1,1, θ)

"""An alias for the [`GenConstrainedUnscented`](@ref) approximation method."""
const GenCUT = GenConstrainedUnscented

"""An alias for the [`GenConstrainedUnscented`](@ref) approximation method."""
const GenConstrainedUnscentedTransform = GenConstrainedUnscented

getlowerbound(approximation::GenConstrainedUnscented) = approximation.lb
getupperbound(approximation::GenConstrainedUnscented) = approximation.ub
getθ(approximation::GenConstrainedUnscented)          = approximation.θ


## Univariate sigma points and weights
function sigma_points_weights!(approximation::GenConstrainedUnscented, m::Real, V::Real, S::Real, K::Real)
    L     = sqrt(V)
    invL3 = inv(L^3) 
    minK = L^4*(S*invL3)^2
    K = (K <= minK) ? 1.001minK : K
    u     = (1/2)*(-S*invL3 + (1/V)*sqrt(4*K - 3*(S^2)/V))
    v     = u + S*invL3 

    lb = first(getlowerbound(approximation))
    ub = first(getupperbound(approximation))
    θ  = getθ(approximation)
   
    cache   = approximation.cache
    weights = getcache(cache, (:weights, 3)) 
    points  = getcache(cache, (:points, (1, 3)))
    @views begin
        points[1,1] = m
        points[1,2] = m - u*L
        points[1, 3] = m + v*L
        weights[3] = inv(v*(u+v))
        weights[2] = (v/u)*weights[3]
        weights[1] = 1 - weights[3] - weights[2]
    end
    ##lowerbound
    u = @views points[1,2] < lb ? θ*abs((m - lb)/L) : u
    vnew = @views points[1,3] < lb ? θ*abs((lb - m)/L) : v
    vnew = vnew == v ? u + S*invL3  : vnew
    @views begin
        points[1,1] = m
        points[1,2] = m - u*L
        points[1, 3] = m + vnew*L
    end
    ##upperbound
    u = @views points[1,2] > ub ? θ*abs((m - ub)/L) : u
    vnew2 = @views points[1,3] > ub ? θ*abs((ub - m)/L) : vnew
    vnew2 = vnew2 == vnew ? u + S*invL3 : vnew2
    @views begin
        points[1,1] = m
        points[1,2] = m - u*L
        points[1,3] = m + vnew2*L
    end
    
    auxiliary = inv(vnew2*(u+vnew2))
    @views begin
        weights[3] = auxiliary
        weights[2] = (vnew2/u)*auxiliary
        weights[1] = 1-auxiliary*(vnew2/u +1)
    end
    return nothing
end

#Multivariate sigma points and weights
function form_generalized_constrained_sigma_points!(cache::GenUnscentedCache, m::AbstractVector, lb::T1, ub::T2, θ::R = DEFAULT_Θ) where {T1, T2, R}
    dim = length(m)
    form_generalized_sigma_points!(cache, m)

    lowerbound = reduce(vcat, lb)
    upperbound = reduce(vcat, ub)

    points = getcache(cache, (:points, (dim, 2*dim+1)))
  
    u = getcache(cache, (:u, dim))
    v = getcache(cache, (:v, dim))
    L = getcache(cache, (:mul_A_B_At!_ABAt, (dim,dim)))
    vnew = getcache(cache, (:vnew, dim))
    vnew2 = getcache(cache, (:vnew2, dim))
    tmp = getcache(cache, (:tmp, dim))
    comparator = Vector{Bool}(undef, 2dim)

    ## lower bound
    @inbounds for i=1:dim
        @views u[i] = any(map!((x,y) -> isless(x , y), comparator[1:dim], points[:, i+1], lowerbound) ) ?  minimum(map!((x,y,z) -> θ*abs((x - y) / z), tmp, m , lowerbound, @views L[:,i])) : u[i]
        @views vnew[i] = any(map!((x,y) -> isless(x, y),comparator[dim+1:end], points[:, dim+i+1] ,lowerbound) ) ?  minimum(map!((x,y,z) -> θ*abs(x - y)/z, tmp, lowerbound, m, @views L[:,i])) : v[i]
    end
    L3S = getcache(cache, (:L3S, dim))
    vnew = vnew == v ? vnew = u + L3S : vnew

    @inbounds for (u_scal,v_scal, L_vec, i) in zip(u, vnew, eachcol(L), 1:dim)
        map!((L_scal, j) -> m[j] - u_scal * L_scal, view(points, :, i+1), L_vec, 1:dim)
        map!((L_scal, j) -> m[j] + v_scal * L_scal, view(points, :, dim+i+1), L_vec, 1:dim)
    end
    
    ## upper bound
    @inbounds for i=1:dim
        @views u[i] = any(map!((x,y) -> x  > y,comparator[1:dim], points[:, i+1] , upperbound )) ? minimum(map!((x,y,z) ->  θ*abs((x -y)/z), tmp, m, upperbound, @views L[:,i])) : u[i]
        @views vnew2[i] = any(map!((x,y) -> x > y,comparator[dim+1:end],  points[:, dim+i+1] , upperbound)) ? minimum(map!((x,y,z) -> θ*abs((y -x)/z),tmp, m, upperbound, @views L[:,i])) : vnew[i]
    end
    vnew2 = vnew2 == vnew ? vnew2 = u + L3S : vnew2

    @inbounds for (u_scal,v_scal, L_vec, i) in zip(u, vnew2, eachcol(L), 1:dim)
        map!((L_scal, j) -> m[j] - u_scal * L_scal, view(points, :, i+1), L_vec, 1:dim)
        map!((L_scal, j) -> m[j] + v_scal * L_scal, view(points, :, dim+i+1), L_vec, 1:dim)
    end
    
    weights = getcache(cache, (:weights, 2*dim+1))
    
    map!((x,y) -> inv(y)/(y+x), view(weights, dim+2:2dim+1) ,u, vnew2)
    map!((x,y, z) -> z*x/y, view(weights, 2:dim+1),vnew2, u, view(weights, dim+2:2dim+1))
    @views weights[1] = 1 - sum(@views weights[2:end])

    return nothing
end

function sigma_points_weights!(approximation::GenConstrainedUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector)
    gen_unscented_cache = approximation.cache 
    prepare_cache!(gen_unscented_cache, V, S, K)
    form_generalized_constrained_sigma_points!(gen_unscented_cache, m, getlowerbound(approximation),getupperbound(approximation), getθ(approximation))

    return nothing
end


function approximate_expectation(method::GenConstrainedUnscented, q, f::F) where {F}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    approximate_expectation(out_variate_type, in_variate_type, method, q, f, Val(false))
end

function approximate_expectation(method::GenConstrainedUnscented, q, f::F, ::Val{C}) where {F, C}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    approximate_expectation(out_variate_type, in_variate_type, method, q, f, Val(C))
end

function approximate_expectation(method::GenConstrainedUnscented, m, V, S, K, f::F) where {F}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    approximate_expectation(out_variate_type, in_variate_type, method, m, V, S, K, f, Val(false))
end

function approximate_expectation(method::GenConstrainedUnscented, m, V, S, K, f::F, ::Val{C}) where {F, C}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    approximate_expectation(out_variate_type, in_variate_type, method, m, V, S, K, f, Val(C))
end


function approximate_expectation(::Type{T}, ::Type{T}, method::GenConstrainedUnscented, q, f::F,::Val{C}) where {T <: Univariate, F, C}
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

function approximate_expectation(::Type{T}, ::Type{T}, method::GenConstrainedUnscented, m::Real, V::Real, S::Real ,K::Real, f::F,::Val{C}) where {T <: Univariate, F, C}
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


function approximate_expectation(::Type{T1}, ::Type{T2},method::GenConstrainedUnscented, q, f::F, ::Val{C}) where {T1 <: Univariate, T2 <: Multivariate, F, C}
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

function approximate_expectation(::Type{T1}, ::Type{T2},method::GenConstrainedUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector, f::F, ::Val{C}) where {T1 <: Univariate, T2 <: Multivariate, F, C}
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

function approximate_expectation(::Type{T1}, ::Type{T2},method::GenConstrainedUnscented, q, f::F, ::Val{C}) where {T1 <: Multivariate, T2<:Univariate, F, C}
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

function approximate_expectation(::Type{T1}, ::Type{T2},method::GenConstrainedUnscented, m::Real, V::Real, S::Real, K::Real, f::F, ::Val{C}) where {T1 <: Multivariate, T2<:Univariate, F, C}
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


function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenConstrainedUnscented, q, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2<:Univariate, F, C}
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

function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenConstrainedUnscented, m::Real, V::Real, S::Real, K::Real, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2<:Univariate, F, C}
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

function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenConstrainedUnscented, q, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2 <: Multivariate, F, C}
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

function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenConstrainedUnscented, m::AbstractVector,V::AbstractMatrix,S::AbstractVector,K::AbstractVector, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2 <: Multivariate, F, C}
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

function approximate_expectation(::Type{T}, ::Type{T}, method::GenConstrainedUnscented, q::Tuple, f::F, ::Val{C}) where {T <: Univariate, F, C}
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

function approximate_expectation(::Type{T}, ::Type{T}, method::GenConstrainedUnscented, m::Tuple, V::Tuple, S::Tuple, K::Tuple, f::F, ::Val{C}) where {T <: Univariate, F, C}
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

function approximate_expectation(::Type{T}, ::Type{T},method::GenConstrainedUnscented, q, f::F, ::Val{C}) where {T <: Multivariate, F, C}
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

function approximate_expectation(::Type{T}, ::Type{T}, method::GenConstrainedUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector, f::F, ::Val{C}) where {T <: Multivariate, F, C}
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

function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenConstrainedUnscented, q::Tuple, f::F, ::Val{C}) where {T1 <: Univariate, T2<: Multivariate, F, C}
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

function approximate_expectation(::Type{T1}, ::Type{T2}, method::GenConstrainedUnscented, ms::Tuple, Vs::Tuple, Ss::Tuple, Ks::Tuple, f::F, ::Val{C}) where {T1 <: Univariate, T2<: Multivariate, F, C}
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

function approximate_expectation(::Type{T},::Type{T}, method::GenConstrainedUnscented, q::Tuple, f::F, ::Val{C}) where {T <: Multivariate, F, C}
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

function approximate_expectation(::Type{T},::Type{T}, method::GenConstrainedUnscented, ms::Tuple, Vs::Tuple, Ss::Tuple, Ks::Tuple, f::F, ::Val{C}) where {T <: Multivariate, F, C}
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

function approximate_expectation(::Type{T1},::Type{T2}, method::GenConstrainedUnscented, q::Tuple, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2 <: Multivariate, F, C}
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

function approximate_expectation(::Type{T1},::Type{T2}, method::GenConstrainedUnscented, ms::Tuple, Vs::Tuple, Ss::Tuple, Ks::Tuple, f::F, ::Val{C}) where {T1 <: Matrixvariate, T2 <: Multivariate, F, C}
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

function Unscented_approximate(method::GenConstrainedUnscented, q, g::G) where {G}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    unscented_statistics(out_variate_type, in_variate_type, method, Val(false), q, g)
end

function Unscented_approximate(method::GenConstrainedUnscented, m, V, S, K, g::G) where {G}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)

    unscented_statistics(out_variate_type, in_variate_type, method,Val(false), m, V, S, K, g)
end

function unscented_statistics(method::GenConstrainedUnscented, q, g::G) where {G}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)
    return unscented_statistics(out_variate_type, in_variate_type, method, Val(true), q, g)
end

function unscented_statistics(method::GenConstrainedUnscented, m,V,S,K, g::G) where {G}
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    in_variate_type = getindex(type_parameters, 2)
    return unscented_statistics(out_variate_type, in_variate_type, method,m, V, S, K, Val(true), g)
end

# Univariate in Univariate out
function unscented_statistics(out_variate::Type{T},in_variate::Type{T}, method::GenConstrainedUnscented, ::Val{C}, q, g::G) where {T<:Univariate, C, G}
    mresult = approximate_expectation(method, q, g)
    Vresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(mresult, mean(q)), g), Val(true)) : nothing
    return mresult,Vresult,Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T},in_variate::Type{T}, method::GenConstrainedUnscented, ::Val{C}, m::Real,V::Real,S::Real,K::Real, g::G) where {T<:Univariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(mresult, m), g), Val(true)) : nothing
    return mresult,Vresult,Sresult, Kresult, Cresult
end

# Univariate in Multivariate out
function unscented_statistics(out_variate::Type{T1},in_variate::Type{T2}, method::GenConstrainedUnscented, ::Val{C}, q, g::G) where {T1<:Multivariate, T2<:Univariate, C, G}
    mresult = approximate_expectation(method, q, g)
    Vresult = approximate_expectation(Matrixvariate, Univariate, method, q, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true)) 
    Sresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(Multivariate, Univariate, mresult, mean(q)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T1},in_variate::Type{T2}, method::GenConstrainedUnscented, ::Val{C}, m::Real, V::Real, S::Real, K::Real, g::G) where {T1<:Multivariate, T2<:Univariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(Matrixvariate, Univariate, method, m, V, S, K, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true)) 
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Multivariate, Univariate, mresult, m), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

# Multivariate in Univariate out
function unscented_statistics(out_variate::Type{T1},in_variate::Type{T2}, method::GenConstrainedUnscented, ::Val{C}, q, g::G) where {T1<:Univariate, T2<:Multivariate, C, G}
    mresult = approximate_expectation(method, q, g)   
    Vresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(Univariate, Multivariate, mresult, mean(q)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T1},in_variate::Type{T2}, method::GenConstrainedUnscented, ::Val{C}, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector, g::G) where {T1<:Univariate, T2<:Multivariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)   
    Vresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Univariate, Multivariate, mresult, m), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

#Multivariate in Multivariate out
function unscented_statistics(out_variate::Type{T},in_variate::Type{T}, method::GenConstrainedUnscented, ::Val{C}, q, g::G) where {T <: Multivariate, C, G}
    mresult = approximate_expectation(method, q, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, q, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))
    
    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, mean(q)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T},in_variate::Type{T}, method::GenConstrainedUnscented, ::Val{C}, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector, g::G) where {T <: Multivariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, m, V, S, K, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))
    
    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, m), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

#Univariate in Univariate out with q tuple
function unscented_statistics(out_variate::Type{T}, in_variate::Type{T}, method::GenConstrainedUnscented, ::Val{C}, q::Tuple, g::G) where { T <: Univariate, C, G}
    mresult = approximate_expectation(method, first(q), g)
    Vresult = approximate_expectation(method, first(q), adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, first(q), adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, first(q), adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(mresult, first(mean.(q))), g), Val(true)) : nothing
    return mresult,Vresult,Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T}, in_variate::Type{T}, method::GenConstrainedUnscented, ::Val{C}, m::Tuple, V::Tuple, S::Tuple, K::Tuple, g::G) where { T <: Univariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(MomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(mresult, m), g), Val(true)) : nothing
    return mresult,Vresult,Sresult, Kresult, Cresult
end

#Univariate in Multivariate out with q tuple
function unscented_statistics(out_variate::Type{T1}, in_variate::Type{T2}, method::GenConstrainedUnscented, ::Val{C}, q::Tuple, g::G) where { T1 <: Multivariate,T2 <: Univariate, C, G}
    mresult = approximate_expectation(method, first(q), g)
    Vresult = approximate_expectation(Matrixvariate, Univariate, method, first(q), adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, first(q), adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, first(q), adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, first(q), adjust_integrand(CrossCovariance(Multivariate, Univariate, mresult, mean(first(q))), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T1}, in_variate::Type{T2}, method::GenConstrainedUnscented, ::Val{C}, m::Tuple, V::Tuple, S::Tuple, K::Tuple, g::G) where { T1 <: Multivariate, T2 <: Univariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(Matrixvariate, Univariate, method, m, V, S, K, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Multivariate, Univariate, mresult, first(m)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

#Multivariate in Univariate out with q tuple
function unscented_statistics(out_variate::Type{T1}, in_variate::Type{T2}, method::GenConstrainedUnscented, ::Val{C}, q::Tuple, g::G) where { T1 <: Univariate, T2 <: Multivariate, C, G}
    mresult = approximate_expectation(method, q, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, q, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, mapreduce(mean, vcat, q)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T1}, in_variate::Type{T2}, method::GenConstrainedUnscented, ::Val{C}, m::Tuple, V::Tuple, S::Tuple, K::Tuple, g::G) where { T1 <: Univariate, T2 <: Multivariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, m, V, S, K, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, reduce(vcat, m)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

#Multivariate in Multivariate out with q tuple
function unscented_statistics(out_variate::Type{T}, in_variate::Type{T}, method::GenConstrainedUnscented, ::Val{C}, q::Tuple, g::G) where { T <: Multivariate, C, G}
    mresult = approximate_expectation(method, q, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, q, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, q, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, q, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, mapreduce(mean,vcat,q)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(out_variate::Type{T}, in_variate::Type{T}, method::GenConstrainedUnscented, ::Val{C}, m::Tuple, V::Tuple, S::Tuple, K::Tuple, g::G) where { T <: Multivariate, C, G}
    mresult = approximate_expectation(method, m, V, S, K, g)
    Vresult = approximate_expectation(Matrixvariate, Multivariate, method, m, V, S, K, adjust_integrand(MomentGenerator(Multivariate, Val(2), mresult), g), Val(true))
    Sresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), g), Val(true))
    Kresult = approximate_expectation(method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), g), Val(true))

    Cresult = C ? approximate_expectation(method, m, V, S, K, adjust_integrand(CrossCovariance(Multivariate, Multivariate, mresult, reduce(vcat, m)), g), Val(true)) : nothing
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function approximate(method::GenConstrainedUnscented, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}

    μ_tilde, Σ_tilde,_,_,_ = Unscented_approximate(method, distributions, f)
    type_parameters = typeof(method).parameters
    out_variate_type = first(type_parameters)
    return convert(ExponentialFamily.promote_variate_type( out_variate_type, NormalMeanVariance), μ_tilde, Σ_tilde)
end

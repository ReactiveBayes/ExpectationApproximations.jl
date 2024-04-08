export GenConstrainedUnscented, GenCUT, GenConstrainedUnscentedTransform
using BlockArrays, BlockDiagonals
using  KWLinalg,TensorCore

const DEFAULT_Θ = 0.5
"""
The `GenConstrainedUnscented` structure defines the approximation method of Generalized Constrained Unscented Transform.
# Arguments
- `lb`: lower bound on the sigma points #1
- `ub`: upper bound on the sigma points #2
- `inputdim`: Dimension of inputs. Tuple is not allowed #3
- `outputdim`: Dimensions of outputs. Should be a tuple. i.e outputdim = (2,) or (3,4) #4
- `θ`: internal parameter controlling resampling of sigma points. Between 0 and 1 #5
- `cache`: cache holds all temporary variables #6
lb = (1,) will imply univariate 
ub = (3,)
lb = ([2,1], 3, 4) will imply first variable is 2 two dimensional, 2nd and 3rd variable are univariate
ub = ([3,5], 5, 8)

"""
struct GenConstrainedUnscented{T <: FromTo, T1, T2,I, O, R} <: AbstractApproximationMethod
    lb::T1
    ub::T2
    inputdim::I
    outputdim::O
    θ::R
    cache::ApproximationCache
end

function GenConstrainedUnscented{T}(lb::T1, ub::T2, id::I, od::O, θ::R = DEFAULT_Θ)  where {T <: FromTo, T1,T2,I,O, R}
    @assert size.(lb) == size.(ub) "Upper bound $(ub) and lower bound $(lb) are not of same size"
    @assert sum(length.(lb)) == id "Sum of the length of upper bound $(ub) and lower bound $(lb) elemets should add up to inputdim"
    @assert all(map(d -> all(first(d) .< getindex(d,2)), zip(lb,ub)))  "Upper bound elements $(ub) should be larger than lower bound elements $(lb) "
    @assert 0 < θ < 1 "Free parameter θ=$(θ) should be in between 0 and 1"
  
    return GenConstrainedUnscented{T, T1, T2, I, O, R}(lb, ub, id, od, θ, ApproximationCache())
end

GenConstrainedUnscented(lb::T1,ub::T2,θ::R = DEFAULT_Θ) where {T1, T2, R} = GenConstrainedUnscented{FromTo(Univariate,Univariate)}(lb,ub,1,1, θ)

"""An alias for the [`GenConstrainedUnscented`](@ref) approximation method."""
const GenCUT = GenConstrainedUnscented

"""An alias for the [`GenConstrainedUnscented`](@ref) approximation method."""
const GenConstrainedUnscentedTransform = GenConstrainedUnscented

getlowerbound(approximation::GenConstrainedUnscented) = approximation.lb
getupperbound(approximation::GenConstrainedUnscented) = approximation.ub
getθ(approximation::GenConstrainedUnscented)          = approximation.θ


getpoints(method::GenConstrainedUnscented) = getcache(method.cache,(:points, (method.inputdim, 2method.inputdim + 1)))
getweights(method::GenConstrainedUnscented) = getcache(method.cache, (:weights, 2method.inputdim+1))

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
function form_generalized_constrained_sigma_points!(cache::ApproximationCache, m::AbstractVector, lb::T1, ub::T2, θ::R = DEFAULT_Θ) where {T1, T2, R}
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
    T = first(typeof(method).parameters)
    approximate_expectation(T, method, q, f, Val(false))
end

function approximate_expectation(method::GenConstrainedUnscented, q, f::F, ::Val{C}) where {F, C}
    T = first(typeof(method).parameters)
    return approximate_expectation(T, method, q, f, Val(C))
end

function approximate_expectation(FT::Type{T}, method::GenConstrainedUnscented, q::Tuple, f::F, ::Val{C}) where {F, C, T <: FromTo}
    m = map(mean, q)
    V = map(_cov, q)
    S = map(_skewness, q)
    K = map(_kurtosis, q)
    return approximate_expectation(FT, method, m, V, S, K, f, Val(C))
end

function approximate_expectation(FT::Type{T}, method::GenConstrainedUnscented, q, f::F, ::Val{C}) where {F, C, T <: FromTo}

    return approximate_expectation(FT, method, mean(q), _cov(q), _skewness(q), _kurtosis(q), f, Val(C))
end

function approximate_expectation(method::GenConstrainedUnscented, m, V, S, K, f::F) where {F}
    T = first(typeof(method).parameters)
    return approximate_expectation(T, method, m, V, S, K, f, Val(false))
end

function approximate_expectation(method::GenConstrainedUnscented, m, V, S, K, f::F, ::Val{C}) where {F, C}
    T = first(typeof(method).parameters)
    return approximate_expectation(T, method, m, V, S, K, f, Val(C))
end

function approximate_expectation(FT::Type{T}, method::GenConstrainedUnscented, m, V, S, K, f::F, ::Val{C}) where {F, C, T <: FromTo}
    !C ? sigma_points_weights!(method, m, V, S, K) : nothing
    points = getpoints(method)
    weights = getweights(method)
    transformedpoints = getunscentedtransformedpoints(FT, method)
    return weighted_sum(FT, method, points, weights, transformedpoints, f, nothing)
end

function approximate_expectation(FT::Type{T}, method::GenConstrainedUnscented, ms::Tuple, Vs::Tuple, Ss::Tuple, Ks::Tuple, f::F, ::Val{C}) where {F, C, T <: FromTo}
    if !C
        m = reduce(vcat, ms)
        S = reduce(vcat, Ss)
        K = reduce(vcat, Ks)
        V = BlockDiagonal(collect(map(x -> typeof(x) <: Matrix ? x : [x;;], Vs)))
        sigma_points_weights!(method, m, V, S, K) 
    end
    points = getpoints(method)
    weights = getweights(method)
    transformedpoints = getunscentedtransformedpoints(FT, method)

    return weighted_sum(FT, method, points, weights, transformedpoints, f, size.(ms))
end

function unscented_statistics(method::GenConstrainedUnscented, q, f::F) where {F}
    T = first(typeof(method).parameters)
    unscented_statistics(T, method, q, f, Val(false))
end

function unscented_statistics(method::GenConstrainedUnscented, q, f::F, ::Val{C}) where {F, C}
    T = first(typeof(method).parameters)
    return unscented_statistics(T, method, q, f, Val(C), Val(true))
end

function unscented_statistics(FT::Type{T}, method::GenConstrainedUnscented, q::Tuple, f::F, ::Val{C}) where {F, C, T <: FromTo}
    m = map(mean, q)
    V = map(_cov, q)
    S = map(_skewness, q)
    K = map(_kurtosis, q)
    return unscented_statistics(FT, method, m, V, S, K, f, Val(C), Val(false))
end

function unscented_statistics(FT::Type{T}, method::GenConstrainedUnscented, q, f::F, ::Val{C}) where {F, C, T <: FromTo}
    return unscented_statistics(FT, method, mean(q), _cov(q), _skewness(q), _kurtosis(q), f, Val(C), Val(false))
end

function unscented_statistics(method::GenConstrainedUnscented, m, V, S, K, f::F,::Val{Tu}) where {F, Tu}
    T = first(typeof(method).parameters)
    return unscented_statistics(T, method, m, V, S, K, f, Val(false), Val(Tu))
end

function unscented_statistics(method::GenConstrainedUnscented, m, V, S, K, f::F, ::Val{C}, ::Val{Tu}) where {F, C, Tu}
    T = first(typeof(method).parameters)
    return unscented_statistics(T, method, m, V, S, K, f, Val(C), Val(Tu))
end

function unscented_statistics(FT::Type{FromTo{Tin, Tout}}, method::GenConstrainedUnscented, m, V, S, K, f::F, ::Val{C}, ::Val{Tu}) where {F, C, Tin <: Union{Univariate, Multivariate}, Tout <: Multivariate, Tu}
    T = FromTo(Tin, Matrixvariate)
    method2 = GenUnscented{FromTo(Tin, Matrixvariate)}(method.inputdim,(first(method.outputdim),first(method.outputdim)))
    
    mresult = approximate_expectation(FT, method, m, V, S, K, f, Val(false))
    Vresult = approximate_expectation(T, method2, m, V, S, K, adjust_integrand(MomentGenerator(Tout, Val(2), mresult), f, Val(Tu)), Val(false)) 
    Sresult = approximate_expectation(FT, method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), f, Val(Tu)), Val(true)) 
    Kresult = approximate_expectation(FT, method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), f, Val(Tu)), Val(true)) 

    if C
        method3 = GenUnscented{FromTo(Tin, Matrixvariate)}(method.inputdim,(first(method.inputdim),first(method.outputdim)))
        Cresult = approximate_expectation(T, method3, m, V, S, K, adjust_integrand(CrossCovariance(Tout, Tin, mresult, reduce(vcat,m)), f, Val(Tu)), Val(false))
    else
        Cresult = nothing
    end
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(FT::Type{FromTo{Tin, Tout}}, method::GenConstrainedUnscented, m, V, S, K, f::F, ::Val{C}, ::Val{Tu}) where {F, C, Tin <: Union{Multivariate, Univariate}, Tout <: Univariate, Tu}
    mresult = approximate_expectation(FT, method, m, V, S, K, f, Val(false))
    Vresult = approximate_expectation(FT, method, m, V, S, K, adjust_integrand(MomentGenerator(Val(2), mresult), f, Val(Tu)), Val(true)) 
    Sresult = approximate_expectation(FT, method, m, V, S, K, adjust_integrand(MomentGenerator(Val(3), mresult), f, Val(Tu)), Val(true)) 
    Kresult = approximate_expectation(FT, method, m, V, S, K, adjust_integrand(MomentGenerator(Val(4), mresult), f, Val(Tu)), Val(true)) 
    if C
        T = FromTo(Tin, Tin)
        method2 = GenUnscented{T}(method.inputdim, method.inputdim)
        Cresult = approximate_expectation(T, method2, m, V, S, K, adjust_integrand(CrossCovariance(Tout, Tin, mresult, reduce(vcat,m)), f, Val(Tu)), Val(false))
    else
        Cresult = nothing
    end
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function approximate(method::GenConstrainedUnscented, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}
    ## Not optimal as it computes higher order statitstics
    μ_tilde, Σ_tilde, _, _, _ = unscented_statistics(method, distributions, f, Val(false))
    type_parameters = first(typeof(method).parameters)
    out_variate_type = getoutputvariate(type_parameters)
    return convert(ExponentialFamily.promote_variate_type( out_variate_type, NormalMeanVariance), μ_tilde, Σ_tilde)
end

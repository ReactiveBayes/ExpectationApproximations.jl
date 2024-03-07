export GenUnscented, GenUT, GenUnscentedTransform
using BlockArrays, BlockDiagonals
import ExponentialFamily: mean_cov
using  KWLinalg,TensorCore
import FillArrays: Eye, Ones, Zeros
import LoopVectorization: @turbo


struct GenUnscentedCache
    cache_matrices::Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}
    cache_vectors::Dict{Tuple{Symbol, Int}, Vector{Float64}}
    points::Dict{Tuple{Symbol, Tuple{Int,Int}}, Matrix{Float64}}
    weights::Dict{Tuple{Symbol, Int}, Vector{Float64}}
end

GenUnscentedCache() = GenUnscentedCache(Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}(), Dict{Tuple{Symbol, Int}, Vector{Float64}}(),Dict{Tuple{Symbol, Tuple{Int,Int}}, Matrix{Float64}}() ,Dict{Tuple{Symbol, Int}, Vector{Float64}}())

function getcache(cache::GenUnscentedCache, label::Tuple{Symbol, Tuple{Int, Int}})
    return get!(() -> Matrix{Float64}(undef, label[2]), cache.cache_matrices, label)
end

function getcache(cache::GenUnscentedCache, label::Tuple{Symbol, Int})
    return get!(() -> Vector{Float64}(undef, label[2]), cache.cache_vectors, label)
end


function getpointscache(cache::GenUnscentedCache, label::Tuple{Symbol, Tuple{Int,Int}})
    return get!(() -> Zeros(first(label[2]),last(label[2])), cache.points, label)
end

function getweightscache(cache::GenUnscentedCache, label::Tuple{Symbol, Int})
    return get!(() -> Zeros(label[2]), cache.weights, label)
end

"""
The `GenUnscented` structure defines the approximation method of Generalized Unscented Transform. 
"""
struct GenUnscented{F} <: AbstractApproximationMethod 
    inputdim :: Int
    outputdim :: Int
    cache :: GenUnscentedCache
    svd_functor :: F
end

GenUnscented(id,od) = GenUnscented(id,od, GenUnscentedCache(),svd_functor_divconquer(id,id,Float64))


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

function prepare!(cache::GenUnscentedCache, V::AbstractMatrix, S::AbstractVector, K::AbstractVector, f::F) where {F}
    Vsvd = getcache(cache, (:Vsvd, size(V)))
    Ssvd = getcache(cache, (:Ssvd, size(V, 1)))
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

    any(K .< minK) ? K[K .< minK] .= 1.001 * (minK[K .< minK] .+ 1e-8) : K

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
    weights = getweightscache(cache, (:weights, 2d + 1)) 
    points  = getpointscache(cache, (:points, (d,2d + 1)))
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

function sigma_points_weights!(approximation::GenUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector)
    gen_unscented_cache = approximation.cache 
    prepare!(gen_unscented_cache, V, S, K, approximation.svd_functor)
    form_generalized_sigma_points!(gen_unscented_cache, m)

    return nothing
end

#Univariate sigma points and weights
function sigma_points_weights!(method::GenUnscented, m::Real, V::Real,S::Real,K::Real)
    L     = sqrt(V)
    invL3 = inv(L^3) 
    u     = (1/2)*(-S*invL3 + (1/V)*sqrt(4*K - 3*(S^2)/V))
    v     = u + S*invL3
   
    cache = method.cache
    weights = getweightscache(cache, (:weights, 3)) 
    points  = getpointscache(cache, (:points, (1, 3)))
    @views begin
        points[1] = m
        points[2] = m - u*L
        points[3] = m + v*L
        weights[3] = inv(v*(u+v))
        weights[2] = (v/u)*weights[3]
        weights[1] = 1 - weights[3] - weights[2]
    end

    return nothing
end


function approximate_expectation(method::GenUnscented, q, f::F) where {F}
    m = mean(q)
    V = _cov(q)
    S = skewness(q)
    K = kurtosis(q, false)
    sigma_points_weights!(method, m, V, S, K)
    cache = method.cache
    points = getpointscache(cache,(:points, (method.inputdim, 2method.inputdim + 1)))
    weights = getweightscache(cache, (:weights, 2method.inputdim + 1))
    
    if method.outputdim == 1
        transformedpoints = getcache(cache, (:transformedpoints, 2method.inputdim+1))

        if method.inputdim == 1
            map!((x,y) -> y*f(x), transformedpoints, points, weights)
        else
            map!((x,y) -> y*f(x), transformedpoints, eachcol(points), weights)
        end

        return sum(transformedpoints)
    else
        transformedpoints = getpointscache(cache, (:transformedpoints, (method.outputdim,2method.inputdim+1)))
        
        if method.inputdim == 1
            map!((x,y) -> lmul!(y, f(x)), eachcol(transformedpoints), points, weights)
        else
            map!((x,y) -> lmul!(y, f(x)), eachcol(transformedpoints), eachcol(points), weights)
        end

        result = getcache(cache, (:result, method.outputdim))
        for elem = 1:method.outputdim
            result[elem] = sum(@views transformedpoints[elem, :])
        end
        
        return result

    end

end

function approximate_expectation(method::GenUnscented, q::Tuple, f::F) where {F}
    ms = map(mean,q)
    ds = size.(ms)

    cache = method.cache   
    m = reduce(vcat, ms)
    S = reduce(vcat, map(skewness, q))
    K = reduce(vcat, map(x -> kurtosis(x,false), q))
    V = BlockDiagonal(collect(map(x -> typeof(x) <: Matrix ? x : [x;;], map(_cov,q))))
    sigma_points_weights!(method, m, V, S, K)
    points = getpointscache(cache,(:points, (method.inputdim, 2method.inputdim+1)))
    weights = getweightscache(cache, (:weights, 2method.inputdim+1))
  
    if method.outputdim == 1
        transformedpoints = getcache(cache, (:transformedpoints, 2method.inputdim+1))
        map!((x,y) -> y*f(__splitjoin(x, ds)...), transformedpoints, eachcol(points), weights)

        return sum(transformedpoints)
    else
        transformedpoints = getpointscache(cache, (:transformedpoints, (method.outputdim,2method.inputdim+1)))
        map!((x,y) -> lmul!(y, f(__splitjoin(x, ds)...)), eachcol(transformedpoints),eachcol(points), weights)
        result = getcache(cache, (:result, method.outputdim))
        for elem = 1:method.outputdim
            result[elem] = sum(@views transformedpoints[elem, :])
        end

        return result
    end
end


function Unscented_approximate(method::GenUnscented, q::Tuple, g::G) where {G}
    unscented_statistics(method, Val(false), q, g)

    # return (m, V, S, K)
end

function unscented_statistics(method::GenUnscented, q::Tuple, g::G) where {G}
    # By default we compute the `C` component, thus `Val(true)`
    return unscented_statistics(method, Val(true), q, g)
end

# Single univariate variable
function unscented_statistics(method::GenUnscented, ::Val{C}, q::Tuple, g::G) where {C, G}
    cache = method.cache
    if method.outputdim == 1
        mresult = approximate_expectation(method, first(q), g)
        Vresult = approximate_expectation(method, first(q), adjust_integrand(CentralMoment(Val(2), 1, mresult), g))
        Sresult = approximate_expectation(method, first(q), adjust_integrand(CentralMoment(Val(3), 1, mresult), g))
        Kresult = approximate_expectation(method, first(q), adjust_integrand(CentralMoment(Val(4), 1, mresult), g))

        # Cresult = C ? approximate_expectation(method, first(q), (x) -> (g(x) - mresult)*(x - first(mean.(q)))) : nothing

        # return mresult, Vresult, Sresult, Kresult, Cresult
    else
        mresult = getweightscache(cache, (:mresult, method.outputdim))
        Vresult = getpointscache(cache, (:Vresult, (method.outputdim, method.outputdim)))
        Sresult = getweightscache(cache, (:Sresult, method.outputdim))
        Kresult = getweightscache(cache, (:Kresult, method.outputdim))
    end


    # mresult = approximate_expectation(method, first(q), g)
    # Vresult = approximate_expectation(method, first(q), (x) -> kron(g(x)-mresult, g(x)-mresult))
    # Sresult = approximate_expectation(method, first(q), (x) -> (g(x) - mresult).^3)
    # Kresult = approximate_expectation(method, first(q), (x) -> (g(x) - mresult).^4)

    # return nothing
    # varf = FunctionWithMemory((collection, memory) -> map!(e -> e ^ 2, memory, collection), ones(outputdim))
end

# Single multivariate variable
function unscented_statistics(method::GenUnscented, ::Val{C}, g::G, means::Tuple{AbstractVector}, covs::Tuple{AbstractMatrix},skewnesses::Tuple{AbstractVector}, kurtosises::Tuple{AbstractVector} ) where {C, G}
    m = first(means)
    V = first(covs)
    S = first(skewnesses)
    K = first(kurtosises)

    (sigma_points, weights) = sigma_points_weights(method, m, V, S, K)

    d = length(m)

    g_sigma = g.(sigma_points)
    @inbounds m_tilde = sum(weights[k + 1] * g_sigma[k + 1] for k in 0:(2d))
    @inbounds V_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d))
    @inbounds S_tilde = sum(weights[k + 1]*(g_sigma[k + 1] - m_tilde).^3 for k in 0:2d)
    @inbounds K_tilde = sum(weights[k + 1]*(g_sigma[k + 1] - m_tilde).^4 for k in 0:2d)
    # Compute `C_tilde` only if `C === true`
    @inbounds C_tilde = C ? sum(weights[k + 1] * (sigma_points[k + 1] - m) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) : nothing

    return (m_tilde, V_tilde, S_tilde, K_tilde, C_tilde)
end

# Multiple inbounds of possibly mixed variate type
function unscented_statistics(method::GenUnscented, ::Val{C}, g::G, ms::Tuple, Vs::Tuple,Ss::Tuple, Ks::Tuple) where {C, G}
    ds     = size.(ms)
    m      = vcat(ms...)
    S      = vcat(Ss...)
    K      = vcat(Ks...)
    V      = vmapreduce(d -> typeof(d) <: Matrix ? sparse(d) : spdiagm([d]),blockdiag ,Vs)
    (sigma_points, weights ) = sigma_points_weights(method, m, V, S, K)
    g_sigma = [g(__splitjoin(sp, ds)...) for sp in sigma_points] # Unpack each sigma point in g

    d = sum(prod.(ds)) # Dimensionality of joint
    @inbounds m_tilde = sum(weights[k + 1] * g_sigma[k + 1] for k in 0:(2d)) # Vector
    @inbounds V_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) # Matrix
    @inbounds S_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde).^3 for k in 0:(2d))
    @inbounds K_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde).^4 for k in 0:(2d))
    # # Compute `C_tilde` only if `C === true`
    @inbounds C_tilde = C ? sum(weights[k + 1] * (sigma_points[k + 1] - m) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) : nothing

    return (m_tilde, V_tilde, S_tilde, K_tilde, C_tilde)
end


function approximate(method::GenUnscented, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}
    statistics = mean_cov.(distributions)
    means      = first.(statistics)
    covs       = last.(statistics)
    skewnesses = skewness.(distributions)
    kurtosises = kurtosis.(distributions, false)

    μ_tilde, Σ_tilde,_,_,_ = Unscented_approximate(method, f, means, covs,skewnesses,kurtosises)

    return convert(promote_variate_type(variate_form(μ_tilde), NormalMeanVariance), μ_tilde, Σ_tilde)
end

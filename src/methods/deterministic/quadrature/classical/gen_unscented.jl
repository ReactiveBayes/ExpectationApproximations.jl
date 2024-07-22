export GenUnscented, GenUT, GenUnscentedTransform, weighted_sum
using BlockArrays, BlockDiagonals
using KWLinalg, TensorCore
import FillArrays: Eye, Zeros

"""
The `GenUnscented` structure defines the approximation method of Generalized Unscented Transform.
# Arguments
- `inputdim`: Dimension of inputs. Tuple is not allowed #1
- `outputdim`: Dimensions of outputs. Should be a tuple. i.e outputdim = (2,) or (3,4) #2
- `cache`: cache holds all temporary variables #3
"""
struct GenUnscented{T <: FromTo, I, O} <: AbstractApproximationMethod
    inputdim::I
    outputdim::O
    cache::ApproximationCache
end

function GenUnscented{T}(id::I, od::O) where {T <: FromTo, I, O}
    GenUnscented{T, I, O}(id, od, ApproximationCache())
end

GenUnscented() = GenUnscented{FromTo(Univariate, Univariate), Int, Int}(1, 1)

getpoints(method::GenUnscented) = getcache(method.cache, (:points, (method.inputdim, 2method.inputdim + 1)))
getweights(method::GenUnscented) = getcache(method.cache, (:weights, 2method.inputdim + 1))

"""An alias for the [`GenUnscented`](@ref) approximation method."""
const GenUT = GenUnscented

"""An alias for the [`GenUnscented`](@ref) approximation method."""
const GenUnscentedTransform = GenUnscented

function mul_A_B_At!(cache::ApproximationCache, A, B)
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

function prepare_cache!(cache::ApproximationCache, V::AbstractMatrix, S::AbstractVector, K::AbstractVector)
    Vsvd = getcache(cache, (:Vsvd, size(V)))
    Ssvd = getcache(cache, (:Ssvd, size(V, 1)))
    f = svd_functor_divconquer(size(V, 1), size(V, 1), Float64)
    svd_object = copy_and_svd!(V, Vsvd, f)

    map!(sqrt, Ssvd, svd_object.S)
    mul_A_B_At!(cache, svd_object.U, Diagonal(Ssvd))

    L3 = getcache(cache, (:L3, size(V)))
    hadamard!(L3, getcache(cache, (:mul_A_B_At!_ABAt, size(V))), getcache(cache, (:mul_A_B_At!_ABAt, size(V))))
    hadamard!(L3, getcache(cache, (:mul_A_B_At!_ABAt, size(V))), L3)

    L4 = getcache(cache, (:L4, size(V)))
    hadamard!(L4, L3, getcache(cache, (:mul_A_B_At!_ABAt, size(V))))

    L3S = getcache(cache, (:L3S, size(S, 1)))
    ldiv!(L3S, fastcholesky!(L3), S)

    minK = getcache(cache, (:minK, size(K, 1)))
    L3S2 = getcache(cache, (:L3S2, size(S, 1)))
    mul!(minK, L4, hadamard!(L3S2, L3S, L3S))

    any(K .< minK) ? K[K .<= minK] .= 1.001 * (minK[K .< minK] .+ 1e-8) : K

    L4K = getcache(cache, (:L4K, size(K, 1)))
    ldiv!(L4K, fastcholesky!(L4), K)

    u = getcache(cache, (:u, size(K, 1)))
    v = getcache(cache, (:v, size(K, 1)))
    map!(sqrt, L3S2, mul!(L3S2, Eye{eltype(L3S2)}(size(V, 1)), L4K, 4, -3))
    map!((x, y) -> (-x + y) / 2, u, L3S, L3S2)
    map!((x, y) -> x + y, v, u, L3S)

    return nothing
end

function form_generalized_sigma_points!(cache::ApproximationCache, m::AbstractVector)
    d       = length(m)
    weights = getcache(cache, (:weights, 2d + 1))
    points  = getcache(cache, (:points, (d, 2d + 1)))
    L       = getcache(cache, (:mul_A_B_At!_ABAt, (d, d)))
    u       = getcache(cache, (:u, d))
    v       = getcache(cache, (:v, d))

    points[:, 1] = m
    @inbounds for (u_scal, v_scal, L_vec, i) in zip(u, v, eachcol(L), 1:d)
        map!((L_scal, j) -> m[j] - u_scal * L_scal, view(points, :, i + 1), L_vec, 1:d)
        map!((L_scal, j) -> m[j] + v_scal * L_scal, view(points, :, d + i + 1), L_vec, 1:d)
    end

    map!((x, y) -> inv(y) / (y + x), view(weights, (d + 2):(2d + 1)), u, v)
    map!((x, y, z) -> z * x / y, view(weights, 2:(d + 1)), v, u, view(weights, (d + 2):(2d + 1)))

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
function sigma_points_weights!(method::GenUnscented, m::Real, V::Real, S::Real, K::Real)
    L     = sqrt(V)
    invL3 = inv(L^3)
    minK  = L^4 * (S * invL3)^2
    K     = (K <= minK) ? 1.001minK : K
    u     = (1 / 2) * (-S * invL3 + (1 / V) * sqrt(4 * K - 3 * (S^2) / V))
    v     = u + S * invL3

    weights = getweights(method)
    points  = getpoints(method)
    @views begin
        points[1, 1] = m
        points[1, 2] = m - u * L
        points[1, 3] = m + v * L
        weights[3] = inv(v * (u + v))
        weights[2] = (v / u) * weights[3]
        weights[1] = 1 - weights[3] - weights[2]
    end

    return nothing
end

function approximate_expectation(method::GenUnscented, q, f::F) where {F}
    T = first(typeof(method).parameters)
    approximate_expectation(T, method, q, f, Val(false))
end

function approximate_expectation(method::GenUnscented, q, f::F, ::Val{C}) where {F, C}
    T = first(typeof(method).parameters)
    return approximate_expectation(T, method, q, f, Val(C))
end

function approximate_expectation(FT::Type{T}, method::GenUnscented, q::Tuple, f::F, ::Val{C}) where {F, C, T <: FromTo}
    m = map(mean, q)
    V = map(_cov, q)
    S = map(_skewness, q)
    K = map(_kurtosis, q)
    return approximate_expectation(FT, method, m, V, S, K, f, Val(C))
end

function approximate_expectation(FT::Type{T}, method::GenUnscented, q, f::F, ::Val{C}) where {F, C, T <: FromTo}
    return approximate_expectation(FT, method, mean(q), _cov(q), _skewness(q), _kurtosis(q), f, Val(C))
end

function approximate_expectation(method::GenUnscented, m, V, S, K, f::F) where {F}
    T = first(typeof(method).parameters)
    return approximate_expectation(T, method, m, V, S, K, f, Val(false))
end

function approximate_expectation(method::GenUnscented, m, V, S, K, f::F, ::Val{C}) where {F, C}
    T = first(typeof(method).parameters)
    return approximate_expectation(T, method, m, V, S, K, f, Val(C))
end

function approximate_expectation(FT::Type{T}, method::GenUnscented, m, V, S, K, f::F, ::Val{C}) where {F, C, T <: FromTo}
    !C ? sigma_points_weights!(method, m, V, S, K) : nothing
    points = getpoints(method)
    weights = getweights(method)
    transformedpoints = getunscentedtransformedpoints(FT, method)
    return weighted_sum(FT, method, points, weights, transformedpoints, f, nothing)
end

function approximate_expectation(FT::Type{T}, method::GenUnscented, ms::Tuple, Vs::Tuple, Ss::Tuple, Ks::Tuple, f::F, ::Val{C}) where {F, C, T <: FromTo}
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

function unscented_statistics(method::GenUnscented, q, f::F) where {F}
    T = first(typeof(method).parameters)
    unscented_statistics(T, method, q, f, Val(false))
end

function unscented_statistics(method::GenUnscented, q, f::F, ::Val{C}) where {F, C}
    T = first(typeof(method).parameters)
    return unscented_statistics(T, method, q, f, Val(C))
end

function unscented_statistics(FT::Type{T}, method::GenUnscented, q::Tuple, f::F, ::Val{C}) where {F, C, T <: FromTo}
    m = map(mean, q)
    V = map(_cov, q)
    S = map(_skewness, q)
    K = map(_kurtosis, q)
    return unscented_statistics(FT, method, m, V, S, K, f, Val(C), Val(true))
end

function unscented_statistics(FT::Type{T}, method::GenUnscented, q, f::F, ::Val{C}) where {F, C, T <: FromTo}
    return unscented_statistics(FT, method, mean(q), _cov(q), _skewness(q), _kurtosis(q), f, Val(C), Val(false))
end

function unscented_statistics(method::GenUnscented, m, V, S, K, f::F, ::Val{Tu}) where {F, Tu}
    T = first(typeof(method).parameters)
    return unscented_statistics(T, method, m, V, S, K, f, Val(false), Val(Tu))
end

function unscented_statistics(method::GenUnscented, m, V, S, K, f::F, ::Val{C}, ::Val{Tu}) where {F, C, Tu}
    T = first(typeof(method).parameters)
    return unscented_statistics(T, method, m, V, S, K, f, Val(C), Val(Tu))
end

function unscented_statistics(
    FT::Type{FromTo{Tin, Tout}}, method::GenUnscented, m, V, S, K, f::F, ::Val{C}, ::Val{Tu}
) where {F, C, Tin <: Union{Univariate, Multivariate}, Tout <: Multivariate, Tu}
    T = FromTo(Tin, Matrixvariate)
    method2 = GenUnscented{FromTo(Tin, Matrixvariate)}(method.inputdim, (first(method.outputdim), first(method.outputdim)))

    mresult = approximate_expectation(FT, method, m, V, S, K, f, Val(false))
    Vresult = approximate_expectation(T, method2, m, V, S, K, adjust_integrand(MomentGenerator(Tout, Val(2), mresult), f, Val(Tu)), Val(false))
    Sresult = approximate_expectation(FT, method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(3), mresult), f, Val(Tu)), Val(true))
    Kresult = approximate_expectation(FT, method, m, V, S, K, adjust_integrand(DiagonalMomentGenerator(Val(4), mresult), f, Val(Tu)), Val(true))
    if C
        method3 = GenUnscented{T}(method.inputdim, (first(method.inputdim), first(method.outputdim)))
        Cresult = approximate_expectation(T, method3, m, V, S, K, adjust_integrand(CrossCovariance(Tout, Tin, mresult, reduce(vcat, m)), f, Val(Tu)), Val(false))
    else
        Cresult = nothing
    end
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function unscented_statistics(
    FT::Type{FromTo{Tin, Tout}}, method::GenUnscented, m, V, S, K, f::F, ::Val{C}, ::Val{Tu}
) where {F, C, Tin <: Union{Multivariate, Univariate}, Tout <: Univariate, Tu}
    mresult = approximate_expectation(FT, method, m, V, S, K, f, Val(false))
    Vresult = approximate_expectation(FT, method, m, V, S, K, adjust_integrand(MomentGenerator(Val(2), mresult), f, Val(Tu)), Val(true))
    Sresult = approximate_expectation(FT, method, m, V, S, K, adjust_integrand(MomentGenerator(Val(3), mresult), f, Val(Tu)), Val(true))
    Kresult = approximate_expectation(FT, method, m, V, S, K, adjust_integrand(MomentGenerator(Val(4), mresult), f, Val(Tu)), Val(true))
    if C
        T = FromTo(Tin, Tin)
        method2 = GenUnscented{T}(method.inputdim, method.inputdim)
        Cresult = approximate_expectation(T, method2, m, V, S, K, adjust_integrand(CrossCovariance(Tout, Tin, mresult, reduce(vcat, m)), f, Val(Tu)), Val(false))
    else
        Cresult = nothing
    end
    return mresult, Vresult, Sresult, Kresult, Cresult
end

function approximate(method::GenUnscented, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}
    ## Not optimal as it computes higher order statitstics
    μ_tilde, Σ_tilde, _, _, _ = unscented_statistics(method, distributions, f, Val(false))
    type_parameters = first(typeof(method).parameters)
    out_variate_type = getoutputvariate(type_parameters)
    return convert(ExponentialFamily.promote_variate_type(out_variate_type, NormalMeanVariance), μ_tilde, Σ_tilde)
end

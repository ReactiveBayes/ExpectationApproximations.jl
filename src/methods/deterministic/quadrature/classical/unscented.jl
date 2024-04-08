export Unscented, UT, UnscentedTransform

const default_alpha = 1e-3 # Default value for the spread parameter
const default_beta = 2.0
const default_kappa = 0.0

struct UnscentedExtra{R, M, C}
    λ  :: R
    Wm :: M
    Wc :: C
end

function UnscentedExtra(dim::Int64; alpha::Real = default_alpha, beta::Real = default_beta, kappa::Real = default_kappa)
    λ = alpha^2 * (dim + kappa) - dim
    Wm = ones(2 * dim + 1)
    Wc = ones(2 * dim + 1)
    Wm ./= (2 * (dim + λ))
    Wc ./= (2 * (dim + λ))
    Wm[1] = λ / (dim + λ)
    Wc[1] = λ / (dim + λ) + (1 - alpha^2 + beta)
    return UnscentedExtra(λ, Wm, Wc)
end

UnscentedExtra() = UnscentedExtra(1)

struct UnscentedParameters{A, B, K} 
    α::A
    β::B
    κ::K
    cov::Bool
end
UnscentedParameters(α::A, β::B, κ::K) where {A, B, K} = UnscentedParameters(α, β, κ, false)
UnscentedParameters() = UnscentedParameters(default_alpha,default_beta, default_kappa,false)

struct Unscented{T <: FromTo, I, O} <: AbstractApproximationMethod 
    inputdim :: I
    outputdim :: O
    parameters :: UnscentedParameters
    cache :: ApproximationCache
    extra :: UnscentedExtra
end


Unscented{T}(id::I,od::O, parameters::UnscentedParameters) where { T <: FromTo, I, O} = Unscented{T, I, O}(id, od, parameters, ApproximationCache(), UnscentedExtra(id; alpha=parameters.α, beta = parameters.β, kappa = parameters.κ))
Unscented{T}(id::I,od::O) where {T <: FromTo, I, O}  = Unscented{T, I, O}(id, od, UnscentedParameters(), ApproximationCache(), UnscentedExtra(id))
Unscented() = Unscented{FromTo(Univariate, Univariate)}(1, 1)

getpoints(method::Unscented) = getcache(method.cache,(:points, (method.inputdim, 2method.inputdim + 1)))
getweights(method::Unscented) = getcache(method.cache, (:weights, 2method.inputdim+1))

"""An alias for the [`Unscented`](@ref) approximation method."""
const UT = Unscented

"""An alias for the [`Unscented`](@ref) approximation method."""
const UnscentedTransform = Unscented

# get-functions for the Unscented structure

getα(approximation::Unscented) = approximation.α
getβ(approximation::Unscented) = approximation.β
getκ(approximation::Unscented) = approximation.κ

getextra(approximation::Unscented) = approximation.extra

getλ(approximation::Unscented)  = getλ(getextra(approximation))
getWm(approximation::Unscented) = getWm(getextra(approximation))
getWc(approximation::Unscented) = getWc(getextra(approximation))

getλ(extra::UnscentedExtra)  = extra.λ
getWm(extra::UnscentedExtra) = extra.Wm
getWc(extra::UnscentedExtra) = extra.Wc

"""Return the sigma points and weights for a Gaussian distribution"""
function sigma_points_weights!(method::Unscented, m::Real, V::Real)
    lambda = getλ(method)
    l = sqrt((1 + lambda) * V)

    points  = getpoints(method)
    @inbounds begin
        @views points[1,1] = m
        @views points[1,2] = m + l
        @views points[1,3] = m - l
    end


    return nothing
end

function sigma_points_weights!(method::Unscented, m::AbstractVector, V::AbstractMatrix)
    d = method.inputdim
    cache = method.cache

    lambda = getλ(method)

    Vsvd = getcache(cache, (:Vsvd, size(V)))
    Ssvd = getcache(cache, (:Ssvd, size(V, 1)))
    f  = svd_functor_divconquer(size(V, 1),size(V, 1), Float64)
    svd_object = copy_and_svd!((d+lambda)*V, Vsvd, f)
  
    map!(sqrt, Ssvd, svd_object.S)
    mul_A_B_At!(cache, svd_object.U, Diagonal(Ssvd))

    L = getcache(cache, (:mul_A_B_At!_ABAt, size(V)))

    sigma_points = getpoints(method)  
    @views sigma_points[:, 1] = m

    @inbounds for (L_vec, i) in zip(eachcol(L), 1:d)
        map!((L_scal, j) -> m[j] + L_scal, view(sigma_points, :, i+1), L_vec, 1:d)
        map!((L_scal, j) -> m[j] - L_scal, view(sigma_points, :, d+i+1), L_vec, 1:d)
    end
     
    return nothing
end

approximate_expectation(method::Unscented, q, f::F) where {F} = approximate_expectation(first(typeof(method).parameters), method, q, f, Val(false))
approximate_expectation(method::Unscented, q, f::F, ::Val{C}) where {F, C} = approximate_expectation(first(typeof(method).parameters), method, q, f, Val(C))
approximate_expectation(FT::Type{T}, method::Unscented, q::Tuple, f::F, ::Val{C}) where {F, C, T <: FromTo} = approximate_expectation(FT, method, map(mean, q), map(_cov, q), f, Val(C))
approximate_expectation(FT::Type{T}, method::Unscented, q, f::F, ::Val{C}) where {F, C, T <: FromTo} = approximate_expectation(FT, method, mean(q), _cov(q), f, Val(C))
approximate_expectation(method::Unscented, m, V, f::F) where {F} = approximate_expectation(first(typeof(method).parameters), method, m, V, f, Val(false))
approximate_expectation(method::Unscented, m, V, f::F, ::Val{C}) where {F, C} = approximate_expectation(first(typeof(method).parameters), method, m, V, f, Val(C))

function approximate_expectation(FT::Type{T}, method::Unscented, m, V, f::F, ::Val{C}; covariance_weights::Bool = false) where {F, C, T <: FromTo}
    !C ? sigma_points_weights!(method, m, V) : nothing
    points = getpoints(method)
    weights = covariance_weights ?  getWc(method) : getWm(method)
    transformedpoints = getunscentedtransformedpoints(FT, method)
    return weighted_sum(FT, method, points, weights, transformedpoints, f, nothing)
end

function approximate_expectation(FT::Type{T}, method::Unscented, ms::Tuple, Vs::Tuple, f::F, ::Val{C}; covariance_weights::Bool = false) where {F, C, T <: FromTo}
    if !C
        m = reduce(vcat, ms)
        V = BlockDiagonal(collect(map(x -> typeof(x) <: Matrix ? x : [x;;], Vs)))
        sigma_points_weights!(method, m, V) 
    end
    points = getpoints(method)
    weights = covariance_weights ? getWc(method) : getWm(method)
    transformedpoints = getunscentedtransformedpoints(FT, method)

    return weighted_sum(FT, method, points, weights, transformedpoints, f, size.(ms))
end

unscented_statistics(method::Unscented, q, f::F) where {F} = unscented_statistics(first(typeof(method).parameters), method, q, f, Val(false))
unscented_statistics(method::Unscented, q, f::F, ::Val{C}) where {F, C} = unscented_statistics(first(typeof(method).parameters), method, q, f, Val(C))
unscented_statistics(FT::Type{T}, method::Unscented, q::Tuple, f::F, ::Val{C}) where {F, C, T <: FromTo} = unscented_statistics(FT, method, map(mean, q), map(_cov, q), f, Val(C), Val(true))
unscented_statistics(FT::Type{T}, method::Unscented, q, f::F, ::Val{C}) where {F, C, T <: FromTo} = unscented_statistics(FT, method, mean(q), _cov(q), f, Val(C),Val(false))
unscented_statistics(method::Unscented, m, V, f::F,::Val{Tu}) where {F,Tu} = unscented_statistics(first(typeof(method).parameters), method, m, V, f, Val(false), Val(Tu))
unscented_statistics(method::Unscented, m, V, f::F, ::Val{C},::Val{Tu}) where {F, C,Tu} = unscented_statistics(first(typeof(method).parameters), method, m, V, f, Val(C), Val(Tu))


function unscented_statistics(FT::Type{FromTo{Tin, Tout}}, method::Unscented, m, V, f::F, ::Val{C}, ::Val{Tu}) where {F, C, Tin <: Union{Univariate, Multivariate}, Tout <: Multivariate, Tu}
    T = FromTo(Tin, Matrixvariate)
    method2 = Unscented{T}(method.inputdim,(first(method.outputdim),first(method.outputdim)), method.parameters)

    mresult = approximate_expectation(FT, method, m, V, f, Val(false))
    Vresult = approximate_expectation(T, method2, m, V, adjust_integrand(MomentGenerator(Tout, Val(2), mresult), f, Val(Tu)), Val(false); covariance_weights = true) 
    if C
        method3 = Unscented{T}(method.inputdim,(first(method.inputdim),first(method.outputdim)), method.parameters)
        Cresult = approximate_expectation(T, method3, m, V, adjust_integrand(CrossCovariance(Tout, Tin, mresult, reduce(vcat,m)), f, Val(Tu)), Val(false); covariance_weights = true)
    else
        Cresult = nothing
    end
    
    return mresult, Vresult, Cresult
end

function unscented_statistics(FT::Type{FromTo{Tin, Tout}}, method::Unscented, m, V, f::F, ::Val{C}, ::Val{Tu}) where {F, C, Tin <: Union{Multivariate, Univariate}, Tout <: Univariate, Tu}
    mresult = approximate_expectation(FT, method, m, V, f, Val(false))
    Vresult = approximate_expectation(FT, method, m, V, adjust_integrand(MomentGenerator(Val(2), mresult), f, Val(Tu)), Val(true); covariance_weights = true) 
    if C
        T = FromTo(Tin, Tin)
        method2 = Unscented{T}(method.inputdim, method.inputdim, method.parameters)
        Cresult = approximate_expectation(T, method2, m, V, adjust_integrand(CrossCovariance(Tout, Tin, mresult, reduce(vcat,m)), f, Val(Tu)), Val(false); covariance_weights = true)
    else
        Cresult = nothing
    end
    return mresult, Vresult, Cresult
end

function approximate(method::Unscented, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}
    μ_tilde, Σ_tilde = unscented_statistics(method, distributions, f, Val(false))
    type_parameters = first(typeof(method).parameters)
    out_variate_type = getoutputvariate(type_parameters)
    return convert(ExponentialFamily.promote_variate_type( out_variate_type, NormalMeanVariance), μ_tilde, Σ_tilde)
end

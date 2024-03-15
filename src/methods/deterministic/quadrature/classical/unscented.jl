export Unscented, UT, UnscentedTransform

const default_alpha = 1e-3 # Default value for the spread parameter
const default_beta = 2.0
const default_kappa = 0.0

struct UnscentedExtra{T, R, M, C}
    L  :: T
    λ  :: R
    Wm :: M
    Wc :: C
end

function UnscentedExtra(dim::Int64; alpha::Real = default_alpha, beta::Real = default_beta, kappa::Real = default_kappa)
    α = alpha
    β = beta
    κ = kappa
    λ = α^2 * (dim + κ) - dim
    Wm = ones(2 * dim + 1)
    Wc = ones(2 * dim + 1)
    Wm ./= (2 * (dim + λ))
    Wc ./= (2 * (dim + λ))
    Wm[1] = λ / (dim + λ)
    Wc[1] = λ / (dim + λ) + (1 - α^2 + β)
    return UnscentedExtra(dim, λ, Wm, Wc)
end

UnscentedExtra() = UnscentedExtra(1)

struct UnscentedParameters{A, B, K} <: AbstractApproximationMethod
    α::A
    β::B
    κ::K
    e::UnscentedExtra
    cov::Bool
end

UnscentedParameters(dim::Int) = UnscentedParameters(default_alpha,default_beta, default_kappa, UnscentedExtra(dim),false)
UnscentedParameters() = UnscentedParameters(1)
"""
The `Unscented` structure defines the approximation method of the `Delta` and `Flow` factor nodes. 
More specifically, it contains the hyperparameters used for sigma points computation.
# Arguments
- `α`: Spread parameter for unscented transform #1
- `β`: Algorithm parameter for incorporating prior information on the (non-Gaussian) distribution of Delta node input
- `κ`: Spread parameter for unscented transform #2
- `e`: Internal cache
The `Unscented` structure with default parameters can be constructed as `Unscented()`.
"""

struct Unscented{T <: FromTo} <: AbstractApproximationMethod 
    inputdim :: Int
    outputdim :: Tuple
    parameters :: UnscentedParameters
    cache :: ApproximationCache
end

Unscented{T}(id::Int,od::Tuple) where {T <: FromTo}  = Unscented{T}(id, od, UnscentedParameters(id), ApproximationCache())
Unscented() = Unscented{FromTo(Univariate, Univariate)}(1,(1,))

"""An alias for the [`Unscented`](@ref) approximation method."""
const UT = Unscented

"""An alias for the [`Unscented`](@ref) approximation method."""
const UnscentedTransform = Unscented

# get-functions for the Unscented structure

getα(approximation::Unscented) = approximation.α
getβ(approximation::Unscented) = approximation.β
getκ(approximation::Unscented) = approximation.κ

getextra(approximation::Unscented) = approximation.e

getL(approximation::Unscented)  = getL(getextra(approximation))
getλ(approximation::Unscented)  = getλ(getextra(approximation))
getWm(approximation::Unscented) = getWm(getextra(approximation))
getWc(approximation::Unscented) = getWc(getextra(approximation))

getL(extra::UnscentedExtra)  = extra.L
getλ(extra::UnscentedExtra)  = extra.λ
getWm(extra::UnscentedExtra) = extra.Wm
getWc(extra::UnscentedExtra) = extra.Wc

"""Return the sigma points and weights for a Gaussian distribution"""
function sigma_points_weights!(method::Unscented, m::Real, V::Real)
    alpha  = getα(method)
    beta   = getβ(method)
    kappa  = getκ(method)
    lambda = (1 + kappa) * alpha^2 - 1

    if (1 + lambda) < 0
        @warn "`(1 + lambda)` in the sigma points computation routine is negative. This may lead to the incorrect results. Adjust the `alpha`, `kappa` and `beta` parameters."
    end

    l = sqrt((1 + lambda) * V)

    weights = getweights(method)
    points  = getpoints(method)

    @views begin
        points[1,1] = m
        points[1,2] = m + l
        points[1,3] = m - l
        weights[1] = lambda / (1 + lambda)
        weights[2] = 1 / (2 * (1 + lambda))
        weights[3] = 1 / (2 * (1 + lambda))
    end

    # sigma_points = (m, m + l, m - l)
    # weights_m    = (lambda / (1 + lambda), 1 / (2 * (1 + lambda)), 1 / (2 * (1 + lambda)))
    # weights_c    = (weights_m[1] + (1 - alpha^2 + beta), 1 / (2 * (1 + lambda)), 1 / (2 * (1 + lambda)))

    # return (sigma_points, weights_m, weights_c)
    return nothing
end











function approximate_expectation(method::Unscented, q, f::F) where {F}
    res = Unscented_approximate(method, f, (mean(q),), (var(q),))
    return_covariance = method.cov
    if return_covariance
        return res
    else
        return first(res)
    end
end

function approximate_expectation(method::Unscented, q::Tuple, f::F) where {F}
    res = Unscented_approximate(method, f, mean.(q), var.(q))
    return_covariance = method.cov
    if return_covariance
        return res
    else
        return first(res)
    end
end

function Unscented_approximate(method::Unscented, f::F, means::Tuple, covs::Tuple) where {F}
    # `Val(false)` indicates that we do not compute the `C` component
    (m, V, _) = unscented_statistics(method, Val(false), f, means, covs)
    return (m, V)
end
#############

function unscented_statistics(method::Unscented, g::G, means::Tuple, covs::Tuple) where {G}
    # By default we compute the `C` component, thus `Val(true)`
    return unscented_statistics(method, Val(true), g, means, covs)
end

# Single univariate variable
function unscented_statistics(method::Unscented, ::Val{C}, g::G, means::Tuple{Real}, covs::Tuple{Real}) where {C, G}
    m = first(means)
    V = first(covs)

    (sigma_points, weights_m, weights_c) = sigma_points_weights(method, m, V)

    g_sigma = g.(sigma_points)
    m_tilde = sum(weights_m .* g_sigma)
    V_tilde = sum(weights_c .* (g_sigma .- m_tilde) .^ 2)

    # Compute `C_tilde` only if `C === true`
    C_tilde = C ? sum(weights_c .* (sigma_points .- m) .* (g_sigma .- m_tilde)) : nothing

    return (m_tilde, V_tilde, C_tilde)
end

# Single multivariate inbound
function unscented_statistics(method::Unscented, ::Val{C}, g::G, means::Tuple{AbstractVector}, covs::Tuple{AbstractMatrix}) where {C, G}
    m = first(means)
    V = first(covs)

    (sigma_points, weights_m, weights_c) = sigma_points_weights(method, m, V)

    d = length(m)

    g_sigma = g.(sigma_points)
    @inbounds m_tilde = sum(weights_m[k + 1] * g_sigma[k + 1] for k in 0:(2d))
    @inbounds V_tilde = sum(weights_c[k + 1] * (g_sigma[k + 1] - m_tilde) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d))

    # Compute `C_tilde` only if `C === true`
    @inbounds C_tilde = C ? sum(weights_c[k + 1] * (sigma_points[k + 1] - m) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) : nothing

    return (m_tilde, V_tilde, C_tilde)
end

# Multiple inbounds of possibly mixed variate type
function unscented_statistics(method::Unscented, ::Val{C}, g::G, ms::Tuple, Vs::Tuple) where {C, G}
    joint = convert(JointNormal, ms, Vs)

    (m, V) = mean_cov(joint)
    ds     = dimensionalities(joint)

    (sigma_points, weights_m, weights_c) = sigma_points_weights(method, m, V)

    g_sigma = [g(__splitjoin(sp, ds)...) for sp in sigma_points] # Unpack each sigma point in g

    d = sum(prod.(ds)) # Dimensionality of joint
    @inbounds m_tilde = sum(weights_m[k + 1] * g_sigma[k + 1] for k in 0:(2d)) # Vector
    @inbounds V_tilde = sum(weights_c[k + 1] * (g_sigma[k + 1] - m_tilde) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) # Matrix

    # Compute `C_tilde` only if `C === true`
    @inbounds C_tilde = C ? sum(weights_c[k + 1] * (sigma_points[k + 1] - m) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) : nothing

    return (m_tilde, V_tilde, C_tilde)
end



function sigma_points_weights(method::Unscented, m::AbstractVector, V::AbstractMatrix)
    d      = length(m)
    alpha  = getα(method)
    beta   = getβ(method)
    kappa  = getκ(method)
    lambda = (d + kappa) * alpha^2 - d

    if (d + lambda) < 0
        @warn "`(d + lambda)` in the sigma points computation routine is negative. This may lead to the incorrect results. Adjust the `alpha`, `kappa` and `beta` parameters."
    end

    T = promote_type(eltype(m), eltype(V))

    sigma_points = Vector{Vector{T}}(undef, 2 * d + 1)
    weights_m    = Vector{T}(undef, 2 * d + 1)
    weights_c    = Vector{T}(undef, 2 * d + 1)

    L = cholsqrt((d + lambda) * V)

    sigma_points[1] = m
    weights_m[1]    = lambda / (d + lambda)
    weights_c[1]    = weights_m[1] + (1 - alpha^2 + beta)

    @inbounds for i in 1:d
        @views sigma_points[2 * i] = m + L[:, i]
        @views sigma_points[2 * i + 1] = m - L[:, i]
    end

    @inbounds weights_m[2:end] .= 1 / (2 * (d + lambda))
    @inbounds weights_c[2:end] .= 1 / (2 * (d + lambda))

    return (sigma_points, weights_m, weights_c)
end

# This function extends the `Unscented` approximation method in case if all inputs are from the `NormalDistributionsFamily`
function approximate(method::Unscented, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}
    statistics = mean_cov.(distributions)
    means      = first.(statistics)
    covs       = last.(statistics)

    μ_tilde, Σ_tilde = approximate(method, f, means, covs)

    return convert(promote_variate_type(variate_form(μ_tilde), NormalMeanVariance), μ_tilde, Σ_tilde)
end

"""
    Unscented(; α = 1e-3, β = 2.0, κ = 0.0)

The unscented transform for Gaussian distributions: `2d + 1` sigma points, the mean and the mean
plus and minus the columns of `√(d + λ) L`, with `λ = α²(d + κ) - d` and `L` the Cholesky factor
of the covariance. Expectations are exact for polynomials up to degree 3.

The centre has a covariance weight that differs from its mean weight by `1 - α² + β`, which
[`mean_cov`](@ref) and [`mean_cov_cross`](@ref) apply; `β = 2` is optimal for Gaussians. With
the default `α` the centre weight is large and negative, so the transform does not support
products of distributions, and [`tilted_mean_cov`](@ref) refuses it.
"""
struct Unscented
    α::Float64
    β::Float64
    κ::Float64
end

Unscented(; α::Real = 1.0e-3, β::Real = 2.0, κ::Real = 0.0) = Unscented(α, β, κ)

"""
    SigmaPointSet(points, weights, centre)

A prepared expectation over sigma points: a weighted sum with the mean weights `weights`. The
first point, the centre, has a covariance weight that is larger by `centre`.

`BayesBase.prepare_expectation(Unscented(), q)` returns a `SigmaPointSet` instead of a
`BayesBase.ExpectationPointSet`, because the unscented transform weights the centre differently
for means and for covariances, and one set of weights cannot express that.
`BayesBase.expectation` uses the mean weights, and [`mean_cov`](@ref) and
[`mean_cov_cross`](@ref) add the correction.
"""
struct SigmaPointSet{P, W, T}
    points::P
    weights::W
    centre::T
end

Base.length(set::SigmaPointSet) = length(set.weights)

BayesBase.expectation(set::SigmaPointSet, f::F) where {F} = expectation(ExpectationPointSet(set.points, set.weights), f)

function mean_cov(set::SigmaPointSet, f::F) where {F}
    m, V = mean_cov(ExpectationPointSet(set.points, set.weights), f)
    return m, V + set.centre * outer(f(first(set.points)) - m)
end

# The centre is the mean, so its correction leaves the cross-covariance unchanged
function mean_cov_cross(set::SigmaPointSet, f::F) where {F}
    m, V = mean_cov(set, f)
    mx = expectation(set, identity)
    C = expectation(set, x -> (x - mx) * (f(x) - m)')
    return m, V, C
end

function tilted_mean_cov(::SigmaPointSet, logf)
    throw(ArgumentError("sigma points can have negative weights, so they cannot tilt a distribution; use a quadrature backend"))
end

BayesBase.supports_expectation(::Unscented, ::Type, ::Type{<:Union{Normal, AbstractMvNormal}}) = true
BayesBase.supports_expectation(::Unscented, ::Type, ::Type{FactorizedJoint{T}}) where {T <: Tuple} = false

BayesBase.prepare_expectation(rule::Unscented, q::Normal) = unscented_points(rule, mean(q), std(q))
BayesBase.prepare_expectation(rule::Unscented, q::AbstractMvNormal) = unscented_points(rule, mean(q), cov(q))

function sigma_weights(rule::Unscented, d)
    λ = rule.α^2 * (d + rule.κ) - d
    return λ, λ / (d + λ), 1 / (2 * (d + λ)), 1 - rule.α^2 + rule.β
end

function unscented_points(rule::Unscented, m::Real, s::Real)
    λ, centre, side, correction = sigma_weights(rule, 1)
    spread = sqrt(1 + λ) * s
    return SigmaPointSet((m, m - spread, m + spread), (centre, side, side), correction)
end

function unscented_points(rule::Unscented, m::AbstractVector, Σ::AbstractMatrix)
    d = length(m)
    λ, centre, side, correction = sigma_weights(rule, d)
    S = sqrt(d + λ) * cholesky(Symmetric(Σ)).L
    return SigmaPointSet(axis_points(m, S), [centre; fill(side, 2d)], correction)
end

# The mean, then the mean plus and minus each column of S
axis_points(m, S) = vcat([m], [m + S[:, i] for i in axes(S, 2)], [m - S[:, i] for i in axes(S, 2)])

"""
    GenUnscented()

The generalised unscented transform: three points per dimension, with weights that match the
mean, variance, skewness and kurtosis of the distribution, so that expectations are exact for
polynomials up to degree 4 of a univariate variable. It supports the univariate families listed
below, whose four moments are finite and computed by Distributions, and Gaussian vectors, for
which the points lie along the columns of the Cholesky factor of the covariance. For a Gaussian
vector the rule matches the moments along each axis, so it is exact for polynomials up to degree 3
and for the fourth powers of single coordinates, but not for products such as `x₁² x₂²`; for `d ≥ 4`
its central weight `1 - d / 3` is not positive.

The univariate families are `Normal`, `Gamma`, `Exponential`, `Chisq`, `Erlang`, `Beta`,
`LogNormal`, `Weibull`, `Uniform`, `Laplace`, `Logistic`, `Gumbel`, `Poisson`, `Bernoulli`,
`Binomial`, `Geometric` and `NegativeBinomial`. Others, such as a heavy-tailed distribution with
an infinite kurtosis, are not supported, although `prepare_expectation` can still be called for
any univariate distribution whose moments are finite.
"""
struct GenUnscented end

const GenUnscentedFamily = Union{
    Normal, Gamma, Exponential, Chisq, Erlang, Beta, LogNormal, Weibull, Uniform, Laplace, Logistic, Gumbel,
    Poisson, Bernoulli, Binomial, Geometric, NegativeBinomial,
}

BayesBase.supports_expectation(::GenUnscented, ::Type, ::Type{<:Union{GenUnscentedFamily, AbstractMvNormal}}) = true

function BayesBase.prepare_expectation(::GenUnscented, q::UnivariateDistribution)
    return generalised_points(mean(q), std(q), skewness(q), kurtosis(q) + 3)
end

BayesBase.prepare_expectation(::GenUnscented, q::Normal) = generalised_points(mean(q), std(q), 0.0, 3.0)

function BayesBase.prepare_expectation(::GenUnscented, q::AbstractMvNormal)
    m = mean(q)
    d = length(m)
    S = sqrt(3) * cholesky(Symmetric(cov(q))).L
    return ExpectationPointSet(axis_points(m, S), [1 - d / 3; fill(1 / 6, 2d)])
end

# For the standardised variable, the points -u and v with u = (-S + √(4K - 3S²)) / 2 and v = u + S
# match the first four moments
function generalised_points(μ, σ, skew, kurt)
    u = (-skew + sqrt(4kurt - 3skew^2)) / 2
    v = u + skew
    w2 = 1 / (v * (u + v))
    w1 = w2 * v / u
    return ExpectationPointSet((μ, μ - u * σ, μ + v * σ), (1 - w1 - w2, w1, w2))
end

"""
    SphericalRadialCubature()

The third-degree spherical-radial cubature rule for Gaussian distributions: `2d` points, the
mean plus and minus the columns of `√d L`, with equal positive weights. It is exact for
polynomials up to degree 3.
"""
struct SphericalRadialCubature end

BayesBase.supports_expectation(::SphericalRadialCubature, ::Type, ::Type{<:Union{Normal, AbstractMvNormal}}) = true

BayesBase.prepare_expectation(::SphericalRadialCubature, q::Normal) = cubature_points(mean(q), std(q))
BayesBase.prepare_expectation(::SphericalRadialCubature, q::AbstractMvNormal) = cubature_points(mean(q), cov(q))

cubature_points(m::Real, s::Real) = ExpectationPointSet((m - s, m + s), (0.5, 0.5))

function cubature_points(m::AbstractVector, Σ::AbstractMatrix)
    d = length(m)
    S = sqrt(d) * cholesky(Symmetric(Σ)).L
    return ExpectationPointSet(axis_points(m, S)[2:end], fill(1 / (2d), 2d))
end

"""
    Domain

The set of values of a random variable, with a smooth bijection `forward` from unconstrained
coordinates onto it, its `inverse`, and the log-Jacobian `logjacobian` of `forward`. A Gaussian
in the unconstrained coordinates, pushed forward, approximates a density on the domain.
"""
abstract type Domain end

"The whole real line, or `ℝᵈ` for a vector; the transform is the identity."
struct RealLine <: Domain end

"The positive half-line, through `x = exp(u)`."
struct Positive <: Domain end

"The open unit interval, through `x = logistic(u)`."
struct UnitInterval <: Domain end

"The half-line above `lower`, through `x = lower + exp(u)`."
struct LowerBounded{T <: Real} <: Domain
    lower::T
end

"The half-line below `upper`, through `x = upper - exp(u)`."
struct UpperBounded{T <: Real} <: Domain
    upper::T
end

"The interval between `lower` and `upper`, through `x = lower + (upper - lower) logistic(u)`."
struct Bounded{T <: Real} <: Domain
    lower::T
    upper::T
end

Bounded(lower::Real, upper::Real) = Bounded(promote(lower, upper)...)

"""
    Simplex(K)

The probability vectors of length `K`, through the additive log-ratio transform from `K - 1`
unconstrained coordinates: `xₖ = exp(uₖ) / (1 + Σⱼ exp(uⱼ))` for `k < K`.
"""
struct Simplex <: Domain
    K::Int
end

"""
    Componentwise(domains...)

A vector whose components lie in different domains, each transformed on its own: a block of
correlated parameters such as a probability and a rate.
"""
struct Componentwise{D <: Tuple} <: Domain
    domains::D
end

Componentwise(domains::Domain...) = Componentwise(domains)

"""
    forward(domain, u)

The value in `domain` of the unconstrained coordinates `u`.
"""
function forward end

"""
    inverse(domain, x)

The unconstrained coordinates of the value `x` in `domain`.
"""
function inverse end

"""
    logjacobian(domain, u)

`log |det ∂forward(domain, u) / ∂u|`, the term a log-density gains in unconstrained coordinates.
For a [`Simplex`](@ref), the Jacobian of the first `K - 1` components, which are free.
"""
function logjacobian end

forward(::RealLine, u) = u
inverse(::RealLine, x) = x
logjacobian(::RealLine, u) = zero(eltype(u))

forward(::Positive, u::Real) = exp(u)
inverse(::Positive, x::Real) = log(x)
logjacobian(::Positive, u::Real) = u

forward(::UnitInterval, u::Real) = logistic(u)
inverse(::UnitInterval, x::Real) = logit(x)
logjacobian(::UnitInterval, u::Real) = -log1pexp(-u) - log1pexp(u)

forward(D::LowerBounded, u::Real) = D.lower + exp(u)
inverse(D::LowerBounded, x::Real) = log(x - D.lower)
logjacobian(::LowerBounded, u::Real) = u

forward(D::UpperBounded, u::Real) = D.upper - exp(u)
inverse(D::UpperBounded, x::Real) = log(D.upper - x)
logjacobian(::UpperBounded, u::Real) = u

forward(D::Bounded, u::Real) = D.lower + (D.upper - D.lower) * logistic(u)
inverse(D::Bounded, x::Real) = logit((x - D.lower) / (D.upper - D.lower))
logjacobian(D::Bounded, u::Real) = log(D.upper - D.lower) - log1pexp(-u) - log1pexp(u)

function forward(::Simplex, u::AbstractVector)
    shift = max(zero(eltype(u)), maximum(u))
    e = exp.(u .- shift)
    last = exp(-shift)
    total = sum(e) + last
    return vcat(e ./ total, last / total)
end
inverse(::Simplex, x::AbstractVector) = log.(x[1:(end - 1)] ./ x[end])
# the determinant of the Jacobian of the first K - 1 components is the product of all K
logjacobian(S::Simplex, u::AbstractVector) = sum(log, forward(S, u))

forward(D::Componentwise, u::AbstractVector) = [forward(d, ui) for (d, ui) in zip(D.domains, u)]
inverse(D::Componentwise, x::AbstractVector) = [inverse(d, xi) for (d, xi) in zip(D.domains, x)]
logjacobian(D::Componentwise, u::AbstractVector) = sum(logjacobian(d, ui) for (d, ui) in zip(D.domains, u))

"""
    domain(q)

The [`Domain`](@ref) of the distribution `q`, read from its support, or `nothing` when it has
none that a transform covers (a discrete distribution, for one). For BayesBase's log-densities,
`ContinuousUnivariateLogPdf` and `ContinuousMultivariateLogPdf`, it is read from their domain;
an unspecified one says nothing. For a product of densities, as the product of a prior and
messages, the domain of the first factor that has one.
"""
domain(::AbstractMvNormal) = RealLine()
domain(q::ContinuousUnivariateDistribution) = interval(minimum(q), maximum(q))
domain(q::Dirichlet) = Simplex(length(q))
domain(q::ContinuousUnivariateLogPdf) = setdomain(support(q))
domain(q::ContinuousMultivariateLogPdf) = setdomain(support(q))
domain(q::ProductOf) = firstdomain((getleft(q), getright(q)))
domain(q::LinearizedProductOf) = firstdomain(view(q.vector, 1:min(q.length, length(q.vector))))
domain(q) = nothing

# the domain of a DomainSets set: an interval, or the whole space in any dimension
setdomain(::DomainSets.FullSpace) = RealLine()
setdomain(set::DomainSets.AbstractInterval) = interval(DomainSets.leftendpoint(set), DomainSets.rightendpoint(set))
setdomain(set::DomainSets.VcatDomain) = all(d -> d isa DomainSets.FullSpace, DomainSets.components(set)) ? RealLine() : nothing
setdomain(set) = nothing

function interval(lower::Real, upper::Real)
    isinf(lower) && isinf(upper) && return RealLine()
    iszero(lower) && isinf(upper) && return Positive()
    iszero(lower) && isone(upper) && return UnitInterval()
    isinf(upper) && return LowerBounded(lower)
    isinf(lower) && return UpperBounded(upper)
    return Bounded(lower, upper)
end

function firstdomain(factors)
    for factor in factors
        D = domain(factor)
        isnothing(D) || return D
    end
    return nothing
end

"""
    pushforward(domain, gaussian)

The distribution of `forward(domain, u)` for `u ~ gaussian`: the `gaussian` itself on the real
line, and a [`TransformedGaussian`](@ref) otherwise, a log-normal distribution on the positive
half-line and a logit-normal one on the unit interval among them. Distributions' `LogNormal` and
`LogitNormal` have no `mean(log, q)` in the ecosystem, which conjugate rules take, and
`LogitNormal` no mean, variance or entropy either; a `TransformedGaussian` has them all.
"""
pushforward(::RealLine, gaussian) = gaussian
pushforward(D::Domain, gaussian) = TransformedGaussian(gaussian, D)

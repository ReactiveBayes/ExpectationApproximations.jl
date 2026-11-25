"""
    Domain

The set of values of a random variable, with a smooth bijection `forward` from unconstrained
coordinates onto it, its `inverse`, and the log-Jacobian `logjacobian` of `forward`. A Gaussian
in the unconstrained coordinates, pushed forward through `forward`, approximates a density on the
domain.
"""
abstract type Domain end

"""
    RealLine()

The whole real line, or `ℝᵈ` for a vector: the domain of a Gaussian. The transform is the
identity, so an approximation in this domain is the same as one without a domain.
"""
struct RealLine <: Domain end

"""
    Positive()

The positive half-line, `x > 0`, through `x = exp(u)`: the domain of a Gamma, an inverse Gamma or
a log-normal variable, such as a rate, a precision or a scale. A Gaussian in `u` is a log-normal
distribution in `x`.
"""
struct Positive <: Domain end

"""
    UnitInterval()

The open unit interval, `0 < x < 1`, through `x = logistic(u)`: the domain of a Beta variable,
such as a probability. A Gaussian in `u` is a logit-normal distribution in `x`.
"""
struct UnitInterval <: Domain end

"""
    LowerBounded(lower)

The half-line `x > lower`, through `x = lower + exp(u)`.
"""
struct LowerBounded{T <: Real} <: Domain
    lower::T
end

"""
    UpperBounded(upper)

The half-line `x < upper`, through `x = upper - exp(u)`.
"""
struct UpperBounded{T <: Real} <: Domain
    upper::T
end

"""
    Bounded(lower, upper)

The interval `lower < x < upper`, through `x = lower + (upper - lower) logistic(u)`: a
[`UnitInterval`](@ref) scaled and shifted.
"""
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

A vector whose components lie in different domains, each transformed separately, such as a
block of correlated parameters made of a probability and a rate.
"""
struct Componentwise{D <: Tuple} <: Domain
    domains::D
end

Componentwise(domains::Domain...) = Componentwise(domains)

"""
    forward(domain, u)

The value in `domain` of the unconstrained coordinates `u`: `exp(u)` for [`Positive`](@ref),
`logistic(u)` for [`UnitInterval`](@ref), and so on. It is a smooth bijection from the real line
(or `ℝᵈ`) onto the domain, so `forward(domain, u)` is in the domain for every finite `u`.

```jldoctest
julia> EA.forward(EA.Positive(), 0.0)
1.0

julia> EA.forward(EA.Simplex(3), [0.0, 0.0]) ≈ [1, 1, 1] / 3
true
```
"""
function forward end

"""
    inverse(domain, x)

The unconstrained coordinates of the value `x` in `domain`, so that
`forward(domain, inverse(domain, x)) ≈ x`: `log(x)` for [`Positive`](@ref), `logit(x)` for
[`UnitInterval`](@ref), and so on. For a [`Simplex`](@ref) of `K` components, a vector of
`K - 1`.

```jldoctest
julia> EA.inverse(EA.UnitInterval(), 0.5)
0.0
```
"""
function inverse end

"""
    logjacobian(domain, u)

`log |det ∂forward(domain, u) / ∂u|`, the term that a log-density gains in unconstrained
coordinates: a density `p(x)` on the domain is the density `p(forward(domain, u)) |det ∂forward/∂u|`
of `u`. For [`Positive`](@ref), `logjacobian(Positive(), u) = u`, since `d exp(u) / du = exp(u)`.
For a [`Simplex`](@ref), it is the log-Jacobian of the first `K - 1` components, which are free.

```jldoctest
julia> EA.logjacobian(EA.Positive(), 0.7)
0.7
```
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

The [`Domain`](@ref) of the distribution `q`, read from its support, or `nothing` when no
domain of this package matches the support, as for a discrete distribution. For the BayesBase
log-densities `ContinuousUnivariateLogPdf` and `ContinuousMultivariateLogPdf`, the domain is read
from their own domain, and is `nothing` when that is unspecified. For a product of densities, such
as the product of a prior and messages, it is the narrowest of its factors' domains: the
intersection of their intervals, where the real line of a Gaussian factor, such as a message
moment-matched through a function, does not hide the support of another factor.
"""
domain(::AbstractMvNormal) = RealLine()
function domain(q::ContinuousUnivariateDistribution)
    # the support is information a distribution may not give: Distributions has no default
    lower, upper = try
        minimum(q), maximum(q)
    catch error
        error isa MethodError || rethrow()
        return nothing
    end
    return interval(lower, upper)
end
domain(q::Dirichlet) = Simplex(length(q))
domain(q::ContinuousUnivariateLogPdf) = setdomain(support(q))
domain(q::ContinuousMultivariateLogPdf) = setdomain(support(q))
domain(q::ProductOf) = productdomain((getleft(q), getright(q)))
domain(q::LinearizedProductOf) = productdomain(view(q.vector, 1:min(q.length, length(q.vector))))
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

# The narrowest domain of the factors of a product, or `nothing` where none has one
function productdomain(factors)
    found = nothing
    for factor in factors
        D = domain(factor)
        isnothing(D) && continue
        found = isnothing(found) ? D : narrower(found, D)
    end
    return found
end

# The intersection of two domains: of two intervals, the interval between the larger lower
# bound and the smaller upper one; where one is not an interval, as a simplex, the first
narrower(a::Domain, b::Domain) = a == b ? a : narrower(endpoints(a), endpoints(b), a)
narrower(::RealLine, b::Domain) = b
narrower(a::Domain, ::RealLine) = a
narrower(a::RealLine, ::RealLine) = a
narrower(a::Tuple, b::Tuple, first) = interval(max(a[1], b[1]), min(a[2], b[2]))
narrower(a, b, first) = first

endpoints(::RealLine) = (-Inf, Inf)
endpoints(::Positive) = (0.0, Inf)
endpoints(::UnitInterval) = (0.0, 1.0)
endpoints(D::LowerBounded) = (D.lower, Inf)
endpoints(D::UpperBounded) = (-Inf, D.upper)
endpoints(D::Bounded) = (D.lower, D.upper)
endpoints(::Domain) = nothing

"""
    pushforward(domain, gaussian)

The distribution of `forward(domain, u)` for `u ~ gaussian`. On the real line this is `gaussian`
itself. On any other domain it is a [`TransformedGaussian`](@ref), including the log-normal
distribution on the positive half-line and the logit-normal distribution on the unit interval.
Distributions' `LogNormal` and `LogitNormal` are not used, because they have no `mean(log, q)` in
the ecosystem, which conjugate rules use, and `LogitNormal` also has no mean, variance or entropy.
A `TransformedGaussian` has all of them.
"""
pushforward(::RealLine, gaussian) = gaussian
pushforward(D::Domain, gaussian) = TransformedGaussian(gaussian, D)

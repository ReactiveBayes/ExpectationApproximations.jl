"""
    TransformedGaussian(gaussian, domain)

The distribution of `forward(domain, u)` for `u ~ gaussian`: a Gaussian in unconstrained
coordinates, pushed forward onto a [`Domain`](@ref) such as the positive half-line, an interval, a
simplex or a block of components in different domains. It has a `logpdf` and can be sampled.
[`GaussHermite`](@ref) computes its expectations through its Gaussian.
"""
struct TransformedGaussian{V <: VariateForm, G <: Distribution, D <: Domain} <: Distribution{V, Continuous}
    gaussian::G
    domain::D
end

function TransformedGaussian(gaussian::Distribution{Univariate}, domain::Domain)
    return TransformedGaussian{Univariate, typeof(gaussian), typeof(domain)}(gaussian, domain)
end

function TransformedGaussian(gaussian::Distribution{Multivariate}, domain::Domain)
    return TransformedGaussian{Multivariate, typeof(gaussian), typeof(domain)}(gaussian, domain)
end

domain(t::TransformedGaussian) = t.domain

const UnivariateTransformed = TransformedGaussian{Univariate}
const MultivariateTransformed = TransformedGaussian{Multivariate}

# `forward` decreases for an upper-bounded domain, so the ends are the smaller and larger limit
Base.minimum(t::UnivariateTransformed) = min(forward(t.domain, -Inf), forward(t.domain, Inf))
Base.maximum(t::UnivariateTransformed) = max(forward(t.domain, -Inf), forward(t.domain, Inf))
Distributions.insupport(t::UnivariateTransformed, x::Real) = minimum(t) < x < maximum(t)

function Distributions.logpdf(t::UnivariateTransformed, x::Real)
    insupport(t, x) || return -Inf
    u = inverse(t.domain, x)
    return logpdf(t.gaussian, u) - logjacobian(t.domain, u)
end

Base.rand(rng::AbstractRNG, t::UnivariateTransformed) = forward(t.domain, rand(rng, t.gaussian))

Distributions.mean(t::UnivariateTransformed) = first(mean_cov(GaussHermite(32), identity, t))
Distributions.var(t::UnivariateTransformed) = last(mean_cov(GaussHermite(32), identity, t))

# `E[f(x)]`, as conjugate rules and energies take it: `mean(log, q)`, `mean(inv, q)`, …
Distributions.mean(f::F, t::TransformedGaussian) where {F <: Function} = expectation(GaussHermite(32), f, t)

# The logarithms of `x` and `1 - x` in the Gaussian's coordinate, where `log(1 - logistic(u))`
# would round to `log(0)` far in a broad Gaussian's tail
Distributions.mean(::typeof(log), t::UnivariateTransformed{<:Distribution, Positive}) = mean(t.gaussian)
Distributions.mean(::typeof(log), t::UnivariateTransformed{<:Distribution, UnitInterval}) = expectation(GaussHermite(32), u -> -log1pexp(-u), t.gaussian)
Distributions.mean(::typeof(mirrorlog), t::UnivariateTransformed{<:Distribution, UnitInterval}) = expectation(GaussHermite(32), u -> -log1pexp(u), t.gaussian)

# H[t] = H[g] + E_g[log |J|]: the Gaussian's entropy and the expected log-Jacobian of `forward`
Distributions.entropy(t::TransformedGaussian) = entropy(t.gaussian) + expectation(GaussHermite(32), u -> logjacobian(t.domain, u), t.gaussian)

Base.length(t::MultivariateTransformed{<:Distribution, Simplex}) = t.domain.K
Base.length(t::MultivariateTransformed) = length(t.gaussian)

function Distributions.insupport(t::MultivariateTransformed{<:Distribution, Simplex}, x::AbstractVector)
    return length(x) == length(t) && all(>(0), x) && isapprox(sum(x), 1)
end
Distributions.insupport(t::MultivariateTransformed, x::AbstractVector) = length(x) == length(t)

function Distributions._logpdf(t::MultivariateTransformed, x::AbstractArray)
    u = inverse(t.domain, x)
    return logpdf(t.gaussian, u) - logjacobian(t.domain, u)
end

function Distributions._rand!(rng::AbstractRNG, t::MultivariateTransformed, x::AbstractVector{<:Real})
    x .= forward(t.domain, rand(rng, t.gaussian))
    return x
end

BayesBase.supports_expectation(::GaussHermite, ::Type, ::Type{<:TransformedGaussian}) = true
BayesBase.supports_expectation(::QuantileLegendre, ::Type, ::Type{<:TransformedGaussian}) = false

function BayesBase.prepare_expectation(rule::GaussHermite, t::TransformedGaussian)
    set = prepare_expectation(rule, t.gaussian)
    return ExpectationPointSet([forward(t.domain, u) for u in set.points], set.weights)
end

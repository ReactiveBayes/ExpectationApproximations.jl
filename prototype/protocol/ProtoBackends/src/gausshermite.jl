"""
    GaussHermite(n)

Gauss–Hermite quadrature with `n` points per dimension: for Gaussians, a tensor grid for
multivariate ones, and for log-normal and logit-normal distributions through their Gaussian in
unconstrained coordinates. Point masses and categoricals get exact sums, labelled exact.

Preparations are [`PointSet`](@ref BayesBase.PointSet)s whose points are typed from the
parameters of the distribution, so dual numbers pass through both the parameters and the
integrand.
"""
struct GaussHermite
    nodes::Vector{Float64}   # of the standard normal
    weights::Vector{Float64} # summing to one

    function GaussHermite(n::Integer)
        n > 0 || throw(ArgumentError("Gauss–Hermite needs at least one point, got $n"))
        ξ, w = gausshermite(n)
        return new(sqrt(2) .* ξ, w ./ sqrt(π))
    end
end

BayesBase.supports(::GaussHermite, ::Type, ::Type{<:Union{Gaussian, MvGaussian, LogNormal, LogitNormal}}) = true
BayesBase.supports(::GaussHermite, ::Type, ::Type{<:Union{PointMass, Categorical}}) = true
BayesBase.haslabel(::GaussHermite, ::Type, ::Type{<:Union{PointMass, Categorical}}, ::Exact) = true

function BayesBase.prepare_expectation(rule::GaussHermite, q::Gaussian)
    return PointSet(mean(q) .+ std(q) .* rule.nodes, rule.weights)
end

function BayesBase.prepare_expectation(rule::GaussHermite, q::LogNormal)
    μ, σ = params(q)
    return PointSet(exp.(μ .+ σ .* rule.nodes), rule.weights)
end

function BayesBase.prepare_expectation(rule::GaussHermite, q::LogitNormal)
    μ, σ = params(q)
    return PointSet(logistic.(μ .+ σ .* rule.nodes), rule.weights)
end

function BayesBase.prepare_expectation(rule::GaussHermite, q::MvGaussian)
    m, L = mean(q), cholesky(Symmetric(cov(q))).L
    grid = vec(CartesianIndices(ntuple(_ -> length(rule.nodes), length(m))))
    points = [m + L * [rule.nodes[i] for i in Tuple(index)] for index in grid]
    weights = [prod(rule.weights[i] for i in Tuple(index)) for index in grid]
    return PointSet(points, weights)
end

BayesBase.prepare_expectation(::GaussHermite, q::PointMass) = PointSet((mean(q),), (1.0,))
BayesBase.prepare_expectation(::GaussHermite, q::Categorical) = PointSet(1:ncategories(q), probs(q))

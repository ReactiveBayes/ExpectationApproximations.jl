"""
    GaussHermite(n)

Gauss–Hermite quadrature with `n` points per dimension, for Gaussian distributions: a `Normal`,
a tensor grid of `nᵈ` points for a `d`-dimensional `MvNormal`, and a `LogNormal` or
`LogitNormal` through its Gaussian in unconstrained coordinates. The result is exact for
polynomials up to degree `2n - 1` of the Gaussian variable; since integrands are not
polynomials in general, the backend is not labelled `ExpectationExact`.

The points are typed from the distribution's parameters, so dual numbers pass through the
parameters as well as the integrand. The points of a Gaussian whose mean is a static vector are
static vectors.
"""
struct GaussHermite
    nodes::Vector{Float64}   # of the standard normal
    weights::Vector{Float64} # summing to one

    function GaussHermite(n::Integer)
        ξ, w = gausshermite(n)
        return new(sqrt(2) .* ξ, w ./ sqrt(π))
    end
end

Base.length(rule::GaussHermite) = length(rule.nodes)
Base.show(io::IO, rule::GaussHermite) = print(io, "GaussHermite(", length(rule), ")")
# Two rules of the same size are the same rule, so registering one again changes nothing
Base.:(==)(a::GaussHermite, b::GaussHermite) = a.nodes == b.nodes && a.weights == b.weights
Base.hash(rule::GaussHermite, h::UInt) = hash(rule.nodes, hash(:GaussHermite, h))

BayesBase.supports_expectation(::GaussHermite, ::Type, ::Type{<:Union{Normal, AbstractMvNormal, LogNormal, LogitNormal}}) = true

BayesBase.prepare_expectation(rule::GaussHermite, q::Normal) = scalar_points(rule, mean(q), std(q))
BayesBase.prepare_expectation(rule::GaussHermite, q::AbstractMvNormal) = tensor_points(rule, mean(q), cov(q))

function BayesBase.prepare_expectation(rule::GaussHermite, q::LogNormal)
    μ, σ = params(q)
    return ExpectationPointSet(exp.(μ .+ σ .* rule.nodes), rule.weights)
end

function BayesBase.prepare_expectation(rule::GaussHermite, q::LogitNormal)
    μ, σ = params(q)
    return ExpectationPointSet(logistic.(μ .+ σ .* rule.nodes), rule.weights)
end

# The rule for a Gaussian with mean `m` and standard deviation `s`, or covariance `Σ`; the
# ExponentialFamily extension uses them for its Gaussian types too
scalar_points(rule::GaussHermite, m, s) = ExpectationPointSet(m .+ s .* rule.nodes, rule.weights)

function tensor_points(rule::GaussHermite, m::AbstractVector, Σ::AbstractMatrix)
    L = cholesky(Symmetric(Σ)).L
    grid = vec(CartesianIndices(ntuple(_ -> length(rule.nodes), dimension(m))))
    points = [m + L * unit(m, rule.nodes, index) for index in grid]
    weights = [prod(i -> rule.weights[i], Tuple(index)) for index in grid]
    return ExpectationPointSet(points, weights)
end

# A static mean has its dimension in its type, so the grid's indices and points are static too
dimension(m::StaticVector) = Val(length(m))
dimension(m::AbstractVector) = length(m)

unit(::StaticVector, nodes, index) = SVector(map(i -> nodes[i], Tuple(index)))
unit(::AbstractVector, nodes, index) = [nodes[i] for i in Tuple(index)]

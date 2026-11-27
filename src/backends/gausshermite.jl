"""
    GaussHermite(n; maxpoints = 10_000)

Gauss–Hermite quadrature with `n` points per dimension, for Gaussian distributions: a `Normal`,
a tensor grid of `nᵈ` points for a `d`-dimensional `MvNormal`, and a `LogNormal` or
`LogitNormal` through its Gaussian in unconstrained coordinates. The result is exact for
polynomials up to degree `2n - 1` of the Gaussian variable; since integrands are not
polynomials in general, the backend is not labelled `ExpectationExact`.

A tensor grid grows exponentially with the dimension, so it has at most `maxpoints` points. Where
`nᵈ` is larger, the grid has `k` points per dimension, the largest `k` with `kᵈ ≤ maxpoints`, and
where that `k` is below 3, the rule is the `2d` points of [`SphericalRadialCubature`](@ref). With
the default, a grid of 16 points per dimension is kept up to `d = 3`, and the spherical-radial
rule takes over from `d = 9`.

The points are typed from the distribution's parameters, so dual numbers pass through the
parameters as well as the integrand. The points of a Gaussian whose mean is a static vector are
static vectors.
"""
struct GaussHermite
    nodes::Vector{Float64}   # of the standard normal
    weights::Vector{Float64} # summing to one
    maxpoints::Int

    function GaussHermite(n::Integer; maxpoints::Integer = 10_000)
        maxpoints >= 1 || throw(ArgumentError("a grid needs at least one point, got maxpoints = $maxpoints"))
        ξ, w = gausshermite(n)
        return new(sqrt(2) .* ξ, w ./ sqrt(π), maxpoints)
    end
end

Base.length(rule::GaussHermite) = length(rule.nodes)
function Base.show(io::IO, rule::GaussHermite)
    print(io, "GaussHermite(", length(rule))
    rule.maxpoints == 10_000 || print(io, "; maxpoints = ", rule.maxpoints)
    return print(io, ")")
end
# Two rules of the same size are the same rule, so registering one again changes nothing
Base.:(==)(a::GaussHermite, b::GaussHermite) = a.nodes == b.nodes && a.weights == b.weights && a.maxpoints == b.maxpoints
Base.hash(rule::GaussHermite, h::UInt) = hash(rule.nodes, hash(rule.maxpoints, hash(:GaussHermite, h)))

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
    d = length(m)
    if Float64(length(rule.nodes))^d > rule.maxpoints
        k = grid_size(rule.maxpoints, d)
        k < 3 && return cubature_points(m, Σ)
        return tensor_points(GaussHermite(k; maxpoints = rule.maxpoints), m, Σ)
    end
    L = cholesky(Symmetric(Σ)).L
    grid = vec(CartesianIndices(ntuple(_ -> length(rule.nodes), dimension(m))))
    points = [m + L * unit(m, rule.nodes, index) for index in grid]
    weights = [prod(i -> rule.weights[i], Tuple(index)) for index in grid]
    return ExpectationPointSet(points, weights)
end

# The largest number of points per dimension whose grid has at most `maxpoints` points
function grid_size(maxpoints, d)
    k = floor(Int, maxpoints^(1 / d))
    while Float64(k + 1)^d <= maxpoints
        k += 1
    end
    while k > 1 && Float64(k)^d > maxpoints
        k -= 1
    end
    return k
end

# A static mean has its dimension in its type, so the grid's indices and points are static too
dimension(m::StaticVector) = Val(length(m))
dimension(m::AbstractVector) = length(m)

unit(::StaticVector, nodes, index) = SVector(map(i -> nodes[i], Tuple(index)))
unit(::AbstractVector, nodes, index) = [nodes[i] for i in Tuple(index)]

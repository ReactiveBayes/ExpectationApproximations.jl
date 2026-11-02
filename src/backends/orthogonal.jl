"""
    GaussLaguerre(n)

Generalised Gauss–Laguerre quadrature with `n` points for a `Gamma` or an `Exponential`
distribution: for a shape `a`, the rule of weight `t^(a - 1) exp(-t)`, so the result is exact
for polynomials up to degree `2n - 1` whatever the shape, below one included. The weights are
normalised to one, so very large shapes work too.

The rule depends on the shape, so it is computed when the expectation is prepared, and dual
numbers pass through the scale and the integrand but not through the shape.
"""
struct GaussLaguerre
    n::Int

    function GaussLaguerre(n::Integer)
        checkpoints(n)
        return new(n)
    end
end

BayesBase.supports(::GaussLaguerre, ::Type, ::Type{<:Union{Gamma, Exponential}}) = true

function BayesBase.prepare_expectation(rule::GaussLaguerre, q::Gamma)
    a, θ = params(q)
    nodes, weights = probability_laguerre(rule.n, a - 1)
    return PointSet(θ .* nodes, weights)
end

function BayesBase.prepare_expectation(rule::GaussLaguerre, q::Exponential)
    nodes, weights = probability_laguerre(rule.n, 0.0)
    return PointSet(scale(q) .* nodes, weights)
end

"""
    GaussJacobi(n)

Gauss–Jacobi quadrature with `n` points for a `Beta` distribution: for shapes `a` and `b`, the
rule of weight `(1 - t)^(b - 1) (1 + t)^(a - 1)` mapped onto the unit interval, so the result is
exact for polynomials up to degree `2n - 1`. The weights are normalised to one.

The rule depends on the shapes, so it is computed when the expectation is prepared, and dual
numbers pass through the integrand but not through the shapes.
"""
struct GaussJacobi
    n::Int

    function GaussJacobi(n::Integer)
        checkpoints(n)
        return new(n)
    end
end

BayesBase.supports(::GaussJacobi, ::Type, ::Type{<:Beta}) = true

function BayesBase.prepare_expectation(rule::GaussJacobi, q::Beta)
    a, b = params(q)
    nodes, weights = probability_jacobi(rule.n, b - 1, a - 1)
    return PointSet((1 .+ nodes) ./ 2, weights)
end

"""
    QuantileLegendre(n)

Gauss–Legendre quadrature with `n` points in probability space, `E_q[f] = ∫₀¹ f(Q(u)) du` with
`Q` the quantile function of `q`, for any continuous univariate distribution. A fallback for
the distributions no other rule covers: a quantile function is rarely smooth at both ends, so
the error decreases algebraically with `n` rather than exponentially.
"""
struct QuantileLegendre
    nodes::Vector{Float64}   # in (0, 1)
    weights::Vector{Float64} # summing to one

    function QuantileLegendre(n::Integer)
        u, w = gausslegendre(n)
        return new((u .+ 1) ./ 2, w ./ 2)
    end
end

BayesBase.supports(::QuantileLegendre, ::Type, ::Type{<:ContinuousUnivariateDistribution}) = true

function BayesBase.prepare_expectation(rule::QuantileLegendre, q::ContinuousUnivariateDistribution)
    return PointSet(quantile.(q, rule.nodes), rule.weights)
end

# Rules with weights normalised to one, whose total mass would overflow for large parameters
probability_laguerre(n, α) = golubwelsch(laguerre_recurrence(n, shapevalue(α))..., 1.0)
probability_jacobi(n, α, β) = golubwelsch(jacobi_recurrence(n, shapevalue(α), shapevalue(β))..., 1.0)

# A shape parameter as a Float64. A distribution with a dual scale has dual shapes too, with zero
# derivatives, which are reduced to their values; the nodes depend on the shape, so a shape whose
# derivative is not zero is an error rather than a derivative silently lost.
shapevalue(x::Real) = Float64(x)
function shapevalue(x::ForwardDiff.Dual)
    iszero(ForwardDiff.partials(x)) || throw(
        ArgumentError(
            "the nodes of the rule depend on this shape parameter, so its derivative cannot be taken by the rule"
        )
    )
    return shapevalue(ForwardDiff.value(x))
end

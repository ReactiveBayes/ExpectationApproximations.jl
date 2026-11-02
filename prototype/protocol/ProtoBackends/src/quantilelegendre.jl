"""
    QuantileLegendre(n)

Gauss–Legendre quadrature with `n` points in probability space: `E_q[f] = ∫₀¹ f(Q(u)) du`,
with `Q` the quantile function of `q`. Works for any continuous univariate distribution with a
quantile function, such as the Beta and Gamma priors that are the marginals of the first
iteration of a variational schedule. Converges slowly for heavy tails.
"""
struct QuantileLegendre
    nodes::Vector{Float64}   # in (0, 1)
    weights::Vector{Float64} # summing to one

    function QuantileLegendre(n::Integer)
        n > 0 || throw(ArgumentError("Gauss–Legendre needs at least one point, got $n"))
        u, w = gausslegendre(n)
        return new((u .+ 1) ./ 2, w ./ 2)
    end
end

ProtoBase.supports(::QuantileLegendre, ::Type, ::Type{<:ContinuousUnivariateDistribution}) = true

function ProtoBase.prepare_expectation(rule::QuantileLegendre, q::ContinuousUnivariateDistribution)
    return PointSet(quantile.(q, rule.nodes), rule.weights)
end

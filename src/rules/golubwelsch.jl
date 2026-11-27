# Gaussian quadrature rules by the Golub–Welsch algorithm: the nodes are the eigenvalues of the
# Jacobi matrix of the three-term recurrence of the orthogonal polynomials, polished by Newton's
# method on that recurrence, and the weights are the Christoffel numbers 1 / Σₖ pₖ(xᵢ)² of the
# orthonormal polynomials, which stay accurate where the first eigenvector components lose
# relative precision. Rules are computed once, in Float64; the points that an expectation
# evaluates are typed from the distribution's parameters.

"""
    gausshermite(n) -> (nodes, weights)

The Gauss–Hermite rule with `n` points for `∫ f(x) exp(-x²) dx`, exact for polynomials up to
degree `2n - 1`.
"""
function gausshermite(n::Integer)
    checkpoints(n)
    a = zeros(n)
    b = [sqrt(k / 2) for k in 1:(n - 1)]
    return golubwelsch(a, b, sqrt(π))
end

"""
    gausslaguerre(n, α = 0) -> (nodes, weights)

The generalised Gauss–Laguerre rule with `n` points for `∫₀^∞ f(x) x^α exp(-x) dx`, `α > -1`,
exact for polynomials up to degree `2n - 1`.
"""
function gausslaguerre(n::Integer, α::Real = 0.0)
    checkpoints(n)
    α > -1 || throw(ArgumentError("Gauss–Laguerre needs α > -1, got $α"))
    α = Float64(α)
    return golubwelsch(laguerre_recurrence(n, α)..., gamma(α + 1))
end

laguerre_recurrence(n, α) = ([2k + α + 1 for k in 0:(n - 1)], [sqrt(k * (k + α)) for k in 1:(n - 1)])

"""
    gaussjacobi(n, α, β) -> (nodes, weights)

The Gauss–Jacobi rule with `n` points for `∫₋₁¹ f(x) (1 - x)^α (1 + x)^β dx`, `α, β > -1`,
exact for polynomials up to degree `2n - 1`.
"""
function gaussjacobi(n::Integer, α::Real, β::Real)
    checkpoints(n)
    (α > -1 && β > -1) || throw(ArgumentError("Gauss–Jacobi needs α, β > -1, got α = $α, β = $β"))
    α, β = Float64(α), Float64(β)
    s = α + β
    μ0 = 2^(s + 1) * exp(loggamma(α + 1) + loggamma(β + 1) - loggamma(s + 2))
    return golubwelsch(jacobi_recurrence(n, α, β)..., μ0)
end

function jacobi_recurrence(n, α, β)
    s = α + β
    # the k = 0 diagonal entry and the k = 1 off-diagonal entry are the general formulas with a
    # common factor cancelled, which would otherwise be 0 / 0 when α + β is 0 or -1
    a = [k == 0 ? (β - α) / (s + 2) : (β^2 - α^2) / ((2k + s) * (2k + s + 2)) for k in 0:(n - 1)]
    b = map(1:(n - 1)) do k
        k == 1 && return sqrt(4 * (1 + α) * (1 + β) / ((2 + s)^2 * (3 + s)))
        return sqrt(4k * (k + α) * (k + β) * (k + s) / ((2k + s)^2 * (2k + s + 1) * (2k + s - 1)))
    end
    return a, b
end

"""
    gausslegendre(n) -> (nodes, weights)

The Gauss–Legendre rule with `n` points for `∫₋₁¹ f(x) dx`, exact for polynomials up to degree
`2n - 1`.
"""
gausslegendre(n::Integer) = gaussjacobi(n, 0.0, 0.0)

function checkpoints(n)
    n >= 1 || throw(ArgumentError("a quadrature rule needs at least one point, got $n"))
    return nothing
end

# `a` holds the diagonal and `b` the off-diagonal of the Jacobi matrix, `μ0` the total mass of the
# weight function
function golubwelsch(a::Vector{Float64}, b::Vector{Float64}, μ0::Float64)
    n = length(a)
    n == 1 && return ([a[1]], [μ0])
    nodes = eigvals(SymTridiagonal(a, b))
    weights = similar(nodes)
    for i in eachindex(nodes)
        x = nodes[i]
        for _ in 1:2
            value, derivative = recurrence(x, a, b, μ0)
            x -= value / derivative
        end
        nodes[i] = x
        weights[i] = christoffel(x, a, b, μ0)
    end
    return nodes, weights
end

# The degree-n polynomial of the recurrence, up to a constant factor, and its derivative at x
function recurrence(x, a, b, μ0)
    n = length(a)
    previous, current = 0.0, 1 / sqrt(μ0)
    dprevious, dcurrent = 0.0, 0.0
    for k in 1:n
        bprevious = k == 1 ? 0.0 : b[k - 1]
        next = (x - a[k]) * current - bprevious * previous
        dnext = current + (x - a[k]) * dcurrent - bprevious * dprevious
        if k < n # orthonormal up to degree n - 1
            next, dnext = next / b[k], dnext / b[k]
        end
        previous, current = current, next
        dprevious, dcurrent = dcurrent, dnext
    end
    return current, dcurrent
end

# 1 / Σₖ pₖ(x)² over the orthonormal polynomials of degree 0 to n - 1
function christoffel(x, a, b, μ0)
    n = length(a)
    previous, current = 0.0, 1 / sqrt(μ0)
    total = current^2
    for k in 1:(n - 1)
        bprevious = k == 1 ? 0.0 : b[k - 1]
        next = ((x - a[k]) * current - bprevious * previous) / b[k]
        previous, current = current, next
        total += current^2
    end
    return 1 / total
end

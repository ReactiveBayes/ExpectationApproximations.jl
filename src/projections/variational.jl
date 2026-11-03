"""
    gaussian_variational(logp, x0; backend = EADefault(), tol = 1e-10, maxiters = 100)
        -> (; distribution, logZ, mean, iterations, converged)

The Gaussian closest to the density proportional to `exp(logp(x))` in the sense of variational
inference: the `q` that minimises `KL(q ‖ p)`, or maximises the evidence lower bound
`E_q[logp] + H[q]`. `x0` is a number or a vector.

The bound is maximised over the Gaussian's mean `m` and the Cholesky factor `L` of its covariance,
the diagonal of `L` on a log scale. The expectation is a quadrature from `backend` in standard
coordinates: the points `zᵢ` and weights `wᵢ` of a standard Gaussian, so that

    E_q[logp] + H[q] = Σᵢ wᵢ logp(m + L zᵢ) + log det L + d/2 log(2πe)

is a smooth function of `(m, L)`. Newton's method, as in [`laplace`](@ref), with gradients and
Hessians from ForwardDiff, finds its maximum, starting at Laplace's approximation of `logp`. At the
maximum, `E_q[∇ logp] = 0` and `E_q[∇² logp] = -Σ⁻¹`, up to the quadrature's error: the fixed point
of conjugate-computation variational inference (Khan and Rue), the natural-gradient iteration in
the Gaussian's mean parameters.

- `logZ` is the bound at `q`, a lower bound on `log ∫ exp(logp(x)) dx`, equal to it where the
  density is Gaussian;
- `mean` is the Gaussian's mean;
- `converged` is false when `maxiters` Newton steps did not settle to `tol`, or no step length
  raised the bound.

`distribution` is a `Normal` or an `MvNormal`. `logp` must accept ForwardDiff's numbers, as for
[`laplace`](@ref).
"""
function gaussian_variational(logp::F, x0::Real; backend = EADefault(), tol::Real = 1.0e-10, maxiters::Integer = 100) where {F}
    m0, s0 = variational_start(logp, x0)
    points = standard_points(backend, 1)
    bound(θ) = sum(w * logp(θ[1] + exp(θ[2]) * z) for (z, w) in points) + θ[2] + (log2π + 1) / 2
    result = maximise_bound(bound, [m0, log(s0)], tol, maxiters)
    m, s = result.mode[1], exp(result.mode[2])
    return (; distribution = Normal(m, s), logZ = result.value, mean = m, result.iterations, result.converged)
end

function gaussian_variational(logp::F, x0::AbstractVector; backend = EADefault(), tol::Real = 1.0e-10, maxiters::Integer = 100) where {F}
    d = length(x0)
    m0, L0 = variational_start(logp, x0)
    points = standard_points(backend, d)
    pairs = lower_pairs(d)
    function bound(θ)
        m, L = unpack(θ, d, pairs)
        return sum(w * logp(m + L * z) for (z, w) in points) + sum(θ[d + k] for k in 1:d) + d * (log2π + 1) / 2
    end
    result = maximise_bound(bound, pack(m0, L0, pairs), tol, maxiters)
    m, L = unpack(result.mode, d, pairs)
    covariance = Matrix(Symmetric(L * L'))
    return (; distribution = MvNormal(collect(m), covariance), logZ = result.value, mean = m, result.iterations, result.converged)
end

"""
    gaussian_variational(logp, x0, domain::Domain; kwargs...)

The variational Gaussian in the unconstrained coordinates of `domain`: of
`u ↦ logp(forward(domain, u)) + logjacobian(domain, u)`, started at `inverse(domain, x0)`, and
pushed forward onto the domain, a [`TransformedGaussian`](@ref) on any but the real line. `logZ` is
the same lower bound as in the original coordinates, and `mean` is the Gaussian's mean pushed
forward, a point in them.
"""
function gaussian_variational(logp::F, x0, domain::Domain; kwargs...) where {F}
    result = gaussian_variational(u -> logp(forward(domain, u)) + logjacobian(domain, u), inverse(domain, x0); kwargs...)
    return (;
        distribution = pushforward(domain, result.distribution),
        result.logZ,
        mean = forward(domain, result.mean),
        result.iterations,
        result.converged,
    )
end

# Where the maximisation starts: Laplace's approximation where it converges, `x0` with a unit
# variance otherwise
function variational_start(logp, x0::Real)
    result = laplace(logp, x0)
    result.converged && return result.mode, std(result.distribution)
    return float(x0), 1.0
end

function variational_start(logp, x0::AbstractVector)
    result = laplace(logp, x0)
    d = length(x0)
    result.converged && return collect(float.(result.mode)), Matrix(cholesky(Symmetric(Matrix(cov(result.distribution)))).L)
    return float.(collect(x0)), Matrix{Float64}(I, d, d)
end

# The quadrature of a standard Gaussian in `d` dimensions, as (point, weight) pairs
function standard_points(backend, d)
    q = d == 1 ? Normal(0.0, 1.0) : MvNormal(zeros(d), Matrix{Float64}(I, d, d))
    # a router's choice, so that its point set is reached
    chosen = which_expectation_backend(backend, Any, typeof(q))
    isnothing(chosen) && throw(UnsupportedExpectationError(backend, identity, q))
    prepared = prepare_expectation(chosen, q)
    prepared isa ExpectationPointSet || throw(ArgumentError("`gaussian_variational` needs a point set for a Gaussian from its backend, got a $(typeof(prepared))"))
    return collect(zip(prepared.points, prepared.weights))
end

# θ = (m, log diag L, the rest of the lower triangle of L, column by column)
lower_pairs(d) = [(i, j) for j in 1:d for i in (j + 1):d]

pack(m, L, pairs) = vcat(m, [log(L[k, k]) for k in 1:length(m)], [L[i, j] for (i, j) in pairs])

function unpack(θ, d, pairs)
    m = θ[1:d]
    L = zeros(eltype(θ), d, d)
    for k in 1:d
        L[k, k] = exp(θ[d + k])
    end
    for (k, (i, j)) in enumerate(pairs)
        L[i, j] = θ[2d + k]
    end
    return m, L
end

# Newton's method on the bound, `laplace`'s
function maximise_bound(bound, θ0, tol, maxiters)
    isfinite(bound(θ0)) || throw(ArgumentError("the evidence lower bound is not finite at the starting Gaussian"))
    result = laplace(bound, θ0; tol, maxiters)
    return (; result.mode, value = bound(result.mode), result.iterations, result.converged)
end

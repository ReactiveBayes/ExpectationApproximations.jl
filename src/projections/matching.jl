"""
    moment_matching(logp, x0; backend = GaussHermite(20), tol = 1e-10, maxiters = 100)
        -> (; distribution, logZ, mean, iterations, converged)

The Gaussian with the mean and covariance of the density proportional to `exp(logp(x))`: the
projection of expectation propagation, by quadrature. `x0` is a number or a vector. With `q` the
density's Laplace approximation and `xᵢ`, `wᵢ` the points and weights of `backend` for `q`,

```math
Z \\approx \\sum_i w_i \\frac{p(x_i)}{q(x_i)}, \\qquad
m \\approx \\frac{1}{Z} \\sum_i w_i \\frac{p(x_i)}{q(x_i)}\\, x_i, \\qquad
\\Sigma \\approx \\frac{1}{Z} \\sum_i w_i \\frac{p(x_i)}{q(x_i)}\\, (x_i - m)(x_i - m)^\\top,
```

computed in log space.

The points are put where the density is, not where a start or a cavity is: on the density's
Laplace approximation, from [`laplace`](@ref) started at `x0`, and weighted by the density
against it, `exp(logp(x)) / q(x)`. A factor much sharper than its cavity is then resolved, where
points on the cavity, as in [`tilted_mean_cov`](@ref), all fall where it is negligible. Where
Laplace's method does not converge, the points are on a unit Gaussian at `x0`.

- `logZ` approximates `log ∫ exp(logp(x)) dx`, exactly for a Gaussian density;
- `mean` is the Gaussian's mean;
- `iterations` and `converged` are those of the Laplace approximation that places the points,
  and `converged` is also false where the weights are not finite, or the moments are not a
  Gaussian's, as when the weights fall on a single point: `distribution` is then the Gaussian
  the points were placed on.

`backend` must prepare a point set for a Gaussian with positive weights, a quadrature such as
`GaussHermite` or a router that chooses one. `distribution` is a `Normal` or an `MvNormal`.
"""
function moment_matching(logp::F, x0::Real; backend = GaussHermite(20), tol::Real = 1.0e-10, maxiters::Integer = 100) where {F}
    start = laplace(logp, x0; tol, maxiters)
    base = start.converged ? start.distribution : Normal(float(x0), 1.0)
    logZ, m, v = tilted_mean_cov(backend, x -> logp(x) - logpdf(base, x), base)
    proper = isfinite(m) && isfinite(v) && v > 0
    distribution = proper ? Normal(m, sqrt(v)) : base
    return (; distribution, logZ, mean = mean(distribution), start.iterations, converged = start.converged && isfinite(logZ) && proper)
end

function moment_matching(logp::F, x0::AbstractVector; backend = GaussHermite(20), tol::Real = 1.0e-10, maxiters::Integer = 100) where {F}
    start = laplace(logp, x0; tol, maxiters)
    d = length(x0)
    base = start.converged ? start.distribution : MvNormal(float.(collect(x0)), Matrix{Float64}(I, d, d))
    logZ, m, V = tilted_mean_cov(backend, x -> logp(x) - logpdf(base, x), base)
    V = Matrix(Symmetric(V))
    proper = all(isfinite, m) && all(isfinite, V) && isposdef(V)
    distribution = proper ? MvNormal(collect(m), V) : base
    return (; distribution, logZ, mean = mean(distribution), start.iterations, converged = start.converged && isfinite(logZ) && proper)
end

"""
    moment_matching(logp, x0, domain::Domain; kwargs...)

The moment-matched Gaussian in the unconstrained coordinates of `domain`: of
`u ↦ logp(forward(domain, u)) + logjacobian(domain, u)`, started at `inverse(domain, x0)`, and
pushed forward onto the domain, a [`TransformedGaussian`](@ref) on any but the real line. `logZ`
approximates the same integral as in the original coordinates, and `mean` is
`forward(domain, m)`, the image of the Gaussian's mean `m`: a point in the domain, not the mean of
`distribution`, which a nonlinear transform moves.
"""
function moment_matching(logp::F, x0, domain::Domain; kwargs...) where {F}
    result = moment_matching(u -> logp(forward(domain, u)) + logjacobian(domain, u), inverse(domain, x0); kwargs...)
    return (;
        distribution = pushforward(domain, result.distribution),
        result.logZ,
        mean = forward(domain, result.mean),
        result.iterations,
        result.converged,
    )
end

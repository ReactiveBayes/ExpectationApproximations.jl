"""
    laplace(logp, x0; tol = 1e-10, maxiters = 100)
        -> (; distribution, logZ, mode, iterations, converged)

The Laplace approximation of the density proportional to `exp(logp(x))`: a Gaussian at its mode,
with the inverse of the negative Hessian there as covariance. `x0` is a number or a vector,
including a static vector. Gradients and Hessians come from ForwardDiff, so `logp` can be any
Julia function that accepts dual numbers, including an expectation of a log-density.

- The mode is found by Newton's method with a backtracking line search. A step is shortened when
  it leaves the domain of `logp`, where `logp` is `-Inf`, or does not increase `logp` enough.
- Where the negative Hessian is not positive definite, for example at a starting point where
  `logp` is not concave, its eigenvalues are reflected and floored, so that the Newton step still
  points uphill.
- `logZ` is Laplace's approximation of the log-normaliser, with `x̂` the mode and `H` the
  Hessian of `logp` there:

  ```math
  \\log \\int e^{\\log p(x)}\\,dx \\approx \\log p(\\hat x) + \\frac{d}{2}\\log 2\\pi - \\frac{1}{2}\\log\\det(-H),
  ```

  zero for a normalised Gaussian.

`distribution` is a `Normal` or an `MvNormal`. `converged` is false when the iteration did not
settle within `maxiters` Newton steps, or the line search could not increase `logp`.

```jldoctest
julia> result = EA.laplace(x -> logpdf(Gamma(3.0, 0.5), x), 1.0);

julia> result.mode ≈ 1.0  # (shape - 1) * scale
true

julia> result.converged
true
```
"""
function laplace(logp::L, x0::Real; tol::Real = 1.0e-10, maxiters::Integer = 100) where {L}
    x = float(x0)
    fx = start(logp, x)
    converged, iterations = false, 0
    for iteration in 1:maxiters
        iterations = iteration
        g, h = derivatives(logp, x)
        step = g / curvature(-h)
        # at the mode already: a step below the tolerance cannot raise `logp` measurably
        if abs(step) <= tol * (1 + abs(x))
            converged = true
            break
        end
        x, fx, moved = backtrack(logp, x, fx, step, g * step)
        if isnothing(moved)
            break
        elseif abs(moved) <= tol * (1 + abs(x))
            converged = true
            break
        end
    end
    _, h = derivatives(logp, x)
    precision = curvature(-h)
    logZ = fx + (log2π - log(precision)) / 2
    return (; distribution = Normal(x, sqrt(inv(precision))), logZ, mode = x, iterations, converged)
end

function laplace(logp::L, x0::AbstractVector; tol::Real = 1.0e-10, maxiters::Integer = 100) where {L}
    x = float.(x0)
    fx = start(logp, x)
    converged, iterations = false, 0
    for iteration in 1:maxiters
        iterations = iteration
        g, H = ForwardDiff.gradient(logp, x), ForwardDiff.hessian(logp, x)
        step = curvature(-H) \ g
        if norm(step) <= tol * (1 + norm(x))
            converged = true
            break
        end
        x, fx, moved = backtrack(logp, x, fx, step, dot(g, step))
        if isnothing(moved)
            break
        elseif norm(moved) <= tol * (1 + norm(x))
            converged = true
            break
        end
    end
    precision = curvature(-ForwardDiff.hessian(logp, x))
    covariance = Matrix(Symmetric(inv(precision)))
    logZ = fx + (length(x) * log2π - logdet(precision)) / 2
    return (; distribution = MvNormal(collect(x), covariance), logZ, mode = x, iterations, converged)
end

"""
    laplace(logp, x0, domain::Domain; tol = 1e-10, maxiters = 100)

The Laplace approximation in the unconstrained coordinates of `domain`. It approximates the
log-density `u ↦ logp(forward(domain, u)) + logjacobian(domain, u)`, starting at
`inverse(domain, x0)`, and pushes the Gaussian forward onto the domain. The result is a
[`TransformedGaussian`](@ref) on every domain except the real line.

- `logZ` approximates the same integral as in the original coordinates, because the log-Jacobian
  accounts for the change of variables.
- `mode` is `forward(domain, û)`, the image of the mode in unconstrained coordinates. It is not
  the mode of `distribution` or of `logp`, because a change of variables moves a mode.
"""
function laplace(logp::L, x0, domain::Domain; kwargs...) where {L}
    result = laplace(u -> logp(forward(domain, u)) + logjacobian(domain, u), inverse(domain, x0); kwargs...)
    return (;
        distribution = pushforward(domain, result.distribution),
        result.logZ,
        mode = forward(domain, result.mode),
        result.iterations,
        result.converged,
    )
end

function start(logp, x)
    fx = logp(x)
    isfinite(fx) || throw(ArgumentError("the log-density is not finite at the starting point $x"))
    return fx
end

function derivatives(logp, x::Real)
    derivative(y) = ForwardDiff.derivative(logp, y)
    return derivative(x), ForwardDiff.derivative(derivative, x)
end

# The curvature Newton's method divides by: the negative Hessian where it is positive definite,
# otherwise with its eigenvalues reflected and floored, so that the step still ascends
curvature(c::Real) = max(abs(c), 1.0e-8 * max(1, abs(c)))

function curvature(C::AbstractMatrix)
    S = Symmetric(C)
    isposdef(S) && return S
    λ, V = eigen(S)
    floor = 1.0e-8 * max(1, maximum(abs, λ))
    return Symmetric(V * Diagonal(max.(abs.(λ), floor)) * V')
end

# Backtracking from `x` along `step` until `logp` increases by a fraction of the first-order
# prediction `slope`; returns the new point, its log-density and the move, or `nothing` for the
# move when no step length increases `logp`
function backtrack(logp, x, fx, step, slope)
    t = 1.0
    while t > 1.0e-12
        candidate = x + t * step
        fcandidate = logp(candidate)
        if isfinite(fcandidate) && fcandidate >= fx + 1.0e-4 * t * slope
            return candidate, fcandidate, t * step
        end
        t /= 2
    end
    return x, fx, nothing
end

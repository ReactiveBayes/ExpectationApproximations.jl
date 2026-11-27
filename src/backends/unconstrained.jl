"""
    UnconstrainedLegendre(n = 8)

Composite Gauss–Legendre quadrature in the unconstrained coordinate of a distribution's domain:
`u = log x` for a positive variable and `u = logit x` for one on the unit interval (see
[`Domain`](@ref)). It is the rule for `Gamma`, `Exponential`, `Chisq`, `Erlang`, `InverseGamma`,
`Weibull` and `Beta` distributions in [`EADefault`](@ref), because the integrands of inference
are rarely smooth in `x` at the boundary: `log x`, `log(1 - x)` and the log-density of a Poisson
or a Bernoulli observation are singular at `0` or `1`, but smooth in `u`.

The density of `u` is unimodal for these families. The rule finds its mode, then splits the
`u` axis into panels, with boundaries where the log-density has fallen by 1, 3, 7, 13, 21, 31 and
43 from the mode, and at distances 1, 2, 4, …, 32 from the mode, so that integrands that grow like
`xᵏ = exp(k u)` are resolved too. Each panel has `n` Gauss–Legendre points, and the weights are the
density at the points, normalised to one. The mass beyond the outermost boundaries is below
`exp(-43)`. With the default `n = 8`, at most about 200 points, `E[log x]` is accurate to about
`1e-12` for a Gamma shape or Beta parameter of 0.05 and above, where Gauss–Laguerre and
Gauss–Jacobi with 16 points are off by up to 6.

A point whose `x` would round to the boundary of the domain, such as `x < floatmin()` for a Gamma
with a shape below about 0.04 or `1 - x < eps()` for a Beta, is placed at the nearest
representable value and keeps its weight. Below those parameters some of the mass is not
representable as a `Float64` at all, and an integrand singular at that boundary loses accuracy:
`E[log x]` under `Gamma(0.01, 1)` is off by about 0.08. Near one, floating-point numbers are
coarser than near zero: `1 - x` at a point close to one carries a relative rounding error of
about `eps() / (1 - x)`, so `E[log(1 - x)]` under a Beta with a second parameter of 0.5 is accurate
only to about `1e-8`, while `E[log x]` under one with a first parameter of 0.5 is accurate to
rounding.

[`GaussLaguerre`](@ref) and [`GaussJacobi`](@ref) remain available, and they are exact for
polynomials in `x`.

`prepare_expectation(UnconstrainedLegendre(), q)` also works for any other continuous univariate
distribution whose log-density is unimodal in the unconstrained coordinate of its
[`domain`](@ref), the real line when it has none. Support is declared only for the families
above. The shapes of a `Gamma` or a `Beta` must be plain numbers, while dual numbers pass through
the scale of a `Gamma` and through the integrand.

```jldoctest
julia> isapprox(BayesBase.expectation(EA.UnconstrainedLegendre(), log, Exponential(1.0)), -Base.MathConstants.eulergamma; rtol = 1e-12)
true
```
"""
struct UnconstrainedLegendre
    nodes::Vector{Float64}   # Gauss–Legendre on [-1, 1]
    weights::Vector{Float64}

    function UnconstrainedLegendre(n::Integer = 8)
        checkpoints(n)
        nodes, weights = gausslegendre(n)
        return new(nodes, weights)
    end
end

Base.length(rule::UnconstrainedLegendre) = length(rule.nodes)
Base.show(io::IO, rule::UnconstrainedLegendre) = print(io, "UnconstrainedLegendre(", length(rule), ")")
Base.:(==)(a::UnconstrainedLegendre, b::UnconstrainedLegendre) = a.nodes == b.nodes
Base.hash(rule::UnconstrainedLegendre, h::UInt) = hash(rule.nodes, hash(:UnconstrainedLegendre, h))

const UnconstrainedFamily = Union{Gamma, Exponential, Chisq, Erlang, InverseGamma, Weibull, Beta}

BayesBase.supports_expectation(::UnconstrainedLegendre, ::Type, ::Type{<:UnconstrainedFamily}) = true

# A Gamma through the rule of its standard form, `v = log(x / θ)`, so that a dual scale passes
# through the points
function BayesBase.prepare_expectation(rule::UnconstrainedLegendre, q::Gamma)
    a, θ = params(q)
    return gamma_points(rule, shapevalue(a), θ)
end
BayesBase.prepare_expectation(rule::UnconstrainedLegendre, q::Exponential) = gamma_points(rule, 1.0, scale(q))
BayesBase.prepare_expectation(rule::UnconstrainedLegendre, q::Chisq) = gamma_points(rule, shapevalue(q.ν / 2), 2.0)
BayesBase.prepare_expectation(rule::UnconstrainedLegendre, q::Erlang) = gamma_points(rule, Float64(shape(q)), scale(q))

function gamma_points(rule::UnconstrainedLegendre, a::Float64, θ)
    us, weights = panel_rule(rule, v -> a * v - exp(v), log(a))
    lower, upper = log(floatmin(Float64)), log(floatmax(Float64))
    logθ = log(ForwardDiff.value(θ))
    points = [θ * exp(clamp(v, lower - logθ, upper - logθ)) for v in us]
    return ExpectationPointSet(points, weights)
end

function BayesBase.prepare_expectation(rule::UnconstrainedLegendre, q::Beta)
    a, b = map(shapevalue, params(q))
    us, weights = panel_rule(rule, u -> -a * log1pexp(-u) - b * log1pexp(u), log(a / b))
    lower, upper = log(floatmin(Float64)), logit(prevfloat(1.0))
    return ExpectationPointSet([logistic(clamp(u, lower, upper)) for u in us], weights)
end

function BayesBase.prepare_expectation(rule::UnconstrainedLegendre, q::ContinuousUnivariateDistribution)
    D = something(domain(q), RealLine())
    ℓ = u -> logpdf(q, forward(D, u)) + logjacobian(D, u)
    us, weights = panel_rule(rule, ℓ, unconstrained_mode(ℓ, inverse(D, start_point(q))))
    # points that round onto the boundary, where the density is not defined, are left out
    keep = [insupport(q, forward(D, u)) && isfinite(forward(D, u)) for u in us]
    kept = weights[keep]
    return ExpectationPointSet([forward(D, u) for u in us[keep]], kept ./ sum(kept))
end

start_point(q) = quantile(q, 0.5)

const PANEL_DROPS = (1.0, 3.0, 7.0, 13.0, 21.0, 31.0, 43.0)
const PANEL_DISTANCES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

# The points in `u` and their normalised weights, for the log-density `ℓ` with mode `m`
function panel_rule(rule::UnconstrainedLegendre, ℓ::L, m::Real) where {L}
    ℓm = ℓ(m)
    isfinite(ℓm) || throw(ArgumentError("the log-density is not finite at its mode $m"))
    left = falls(ℓ, m, ℓm, -1)
    right = falls(ℓ, m, ℓm, 1)
    lower, upper = last(left), last(right)
    boundaries = Float64[left..., m, right...]
    for d in PANEL_DISTANCES
        lower < m - d && push!(boundaries, m - d)
        m + d < upper && push!(boundaries, m + d)
    end
    sort!(boundaries)
    points = Float64[]
    logweights = Float64[]
    for k in 1:(length(boundaries) - 1)
        a, b = boundaries[k], boundaries[k + 1]
        b > a || continue
        half, middle = (b - a) / 2, (b + a) / 2
        for (t, w) in zip(rule.nodes, rule.weights)
            u = middle + half * t
            push!(points, u)
            push!(logweights, log(w * half) + ℓ(u))
        end
    end
    top = maximum(v -> isnan(v) ? -Inf : v, logweights)
    weights = [isfinite(v) ? exp(v - top) : 0.0 for v in logweights]
    return points, weights ./ sum(weights)
end

# The points on one side of the mode where `ℓ` has fallen by each of `PANEL_DROPS`, outwards
function falls(ℓ, m, ℓm, direction)
    found = Float64[]
    inside, step = Float64(m), 1.0e-3 * (1 + abs(m))
    for drop in PANEL_DROPS
        level = ℓm - drop
        above(u) = (value = ℓ(u); !isnan(value) && value > level)
        outside = inside + direction * step
        while above(outside) && abs(outside - m) < 1.0e8
            inside, step = outside, 2step
            outside = inside + direction * step
        end
        for _ in 1:60
            abs(outside - inside) <= 1.0e-6 * abs(outside - m) && break
            middle = (inside + outside) / 2
            above(middle) ? (inside = middle) : (outside = middle)
        end
        push!(found, outside)
        inside, step = outside, abs(outside - m)
    end
    return found
end

# The mode of a unimodal `ℓ`, by bracketing from `u0` and golden-section search
function unconstrained_mode(ℓ, u0)
    u0 = Float64(u0)
    f0 = ℓ(u0)
    isfinite(f0) || throw(ArgumentError("the log-density is not finite at the starting point $u0"))
    step = 1.0e-2 * (1 + abs(u0))
    direction = ℓ(u0 + step) > f0 ? 1 : (ℓ(u0 - step) > f0 ? -1 : 0)
    a, b = u0 - step, u0 + step
    if direction != 0
        inner, value = u0, f0
        while true
            candidate = inner + direction * step
            fcandidate = ℓ(candidate)
            (isnan(fcandidate) || fcandidate <= value || abs(candidate - u0) > 1.0e8) && (b = candidate; break)
            a, inner, value, step = inner, candidate, fcandidate, 2step
        end
        a, b = minmax(a, b)
    end
    ratio = (sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = ℓ(c), ℓ(d)
    for _ in 1:200
        abs(b - a) <= 1.0e-10 * (1 + abs(a) + abs(b)) && break
        if fc > fd || isnan(fd)
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = ℓ(c)
        else
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = ℓ(d)
        end
    end
    return (a + b) / 2
end

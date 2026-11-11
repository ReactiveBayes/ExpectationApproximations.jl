```@meta
CurrentModule = ExpectationApproximations
```

# Gaussian approximations of a density

Much of approximate inference replaces a density it cannot handle by a Gaussian it can. There is
more than one way to choose that Gaussian, and the choice matters. This package implements the
three that inference schemes use, each as a function of a log-density given as plain Julia code:

| Function | The Gaussian | Used by |
|:---------|:-------------|:--------|
| [`laplace`](@ref) | centred at the mode, with the curvature there | the Laplace approximation, Laplace propagation |
| [`moment_matching`](@ref) | with the density's mean and covariance | expectation propagation |
| [`gaussian_variational`](@ref) | closest in ``\mathrm{KL}(q \,\|\, p)`` | variational inference, natural-gradient message passing |

Each takes `logp`, the log of an unnormalised density, and a starting point `x0`, a number or a
vector. Each returns a named tuple with the Gaussian `distribution`, an approximation `logZ` of
the log-normaliser ``\log \int e^{\log p(x)} dx``, whether it `converged`, and the number of
`iterations`. Gradients and Hessians come from ForwardDiff, so `logp` must accept its dual
numbers, which ordinary Julia code does.

```@example projections
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations
nothing # hide
```

## Three Gaussians for one density

Take the skewed density

```math
p(x) \propto \exp\left(x - \tfrac{1}{2} e^{x}\right),
```

a Gumbel distribution for the minimum. Its normaliser, mean and variance are known: ``Z = 2``,
``\mathbb{E}[x] = \log 2 - \gamma`` with ``\gamma`` Euler's constant, and
``\mathrm{Var}[x] = \pi^2 / 6``. Its mode is ``\log 2``.

```@example projections
logp(x) = x - exp(x) / 2

laplace = EA.laplace(logp, 0.0)
matching = EA.moment_matching(logp, 0.0; backend = EA.GaussHermite(40))
variational = EA.gaussian_variational(logp, 0.0; backend = EA.GaussHermite(40))

γ = Base.MathConstants.eulergamma
rows = [
    ("exact", log(2) - γ, π^2 / 6, log(2)),
    ("laplace", mean(laplace.distribution), var(laplace.distribution), laplace.logZ),
    ("moment_matching", mean(matching.distribution), var(matching.distribution), matching.logZ),
    ("gaussian_variational", mean(variational.distribution), var(variational.distribution), variational.logZ),
]
for (name, m, v, logZ) in rows
    println(rpad(name, 22), "mean ", round(m; digits = 4), "   variance ", round(v; digits = 4), "   logZ ", round(logZ; digits = 4))
end
```

```@example projections
using Plots
xs = range(-4, 3; length = 400)
plot(xs, exp.(logp.(xs)) ./ 2; label = "p(x)", linewidth = 3, color = :black, xlabel = "x", ylabel = "density")
plot!(xs, pdf.(laplace.distribution, xs); label = "laplace", linestyle = :dash)
plot!(xs, pdf.(matching.distribution, xs); label = "moment_matching")
plot!(xs, pdf.(variational.distribution, xs); label = "gaussian_variational", linestyle = :dot, linewidth = 2)
```

The three answer different questions:

- **Laplace** looks only at the mode and the curvature there. It is cheap, needs no quadrature,
  and is exact for a Gaussian. Here it sits at the mode ``\log 2``, well to the right of the mean,
  and its variance, ``1``, misses the heavy left tail. Its `logZ` is an estimate of
  ``\log Z``, neither a lower nor an upper bound.
- **Moment matching** reproduces the mean and the variance, which is what expectation propagation
  needs, and its `logZ` is ``\log Z`` itself, up to the quadrature's error. To cover the left
  tail it spreads out: it is lower than ``p`` at the peak and puts mass on the right, where ``p``
  has little.
- **The variational Gaussian** maximises the evidence lower bound
  ``\mathbb{E}_q[\log p] + H[q]``, so its `logZ` is a lower bound on ``\log Z``. Penalised for
  mass where ``p`` has little, it keeps the narrow width of the peak but moves toward the mean.
  On this density its bound happens to equal Laplace's estimate of ``\log Z``; the bound of
  Laplace's Gaussian itself is lower:

```@example projections
elbo(q) = BayesBase.expectation(EA.GaussHermite(40), logp, q) + entropy(q)
(laplace_gaussian = elbo(laplace.distribution), variational_gaussian = elbo(variational.distribution))
```

## Why moment matching places its points on the density

Expectation propagation computes the moments of a *tilted* density: a factor ``f`` times a
Gaussian cavity ``c``. A natural way is quadrature on the cavity, ``\sum_i w_i f(x_i)`` with points
from ``c``, which is what [`tilted_mean_cov`](@ref) does. It fails when the factor is much sharper
than the cavity: all the points fall where the factor is negligible.

```@example projections
cavity, y, r = Normal(0.0, 10.0), 3.0, 1.0e-4           # a wide cavity, a sharp observation
logf(x) = logpdf(Normal(x, sqrt(r)), y)
v_exact = inv(1 / var(cavity) + 1 / r)                  # conjugate, so the answer is known

_, _, v_cavity = EA.tilted_mean_cov(EA.GaussHermite(20), logf, cavity)
result = EA.moment_matching(x -> logf(x) + logpdf(cavity, x), 0.0)
(exact = v_exact, points_on_cavity = v_cavity, moment_matching = var(result.distribution))
```

[`moment_matching`](@ref) first finds the density's Laplace approximation, places the points on
it, and weighs each point by the density against that Gaussian. The points then sit where the
mass is, whatever the cavity, and a Gaussian density is matched exactly.

## A posterior in two dimensions

The same functions take vectors. The bioassay logistic regression from Gelman et al., *Bayesian
Data Analysis*, has a two-dimensional posterior with a strong skew in the slope:

```@example projections
using StatsFuns: logistic
x = [-0.86, -0.3, -0.05, 0.73]
n = [5, 5, 5, 5]
y = [0, 1, 3, 5]
logposterior(θ) = logpdf(MvNormal(zeros(2), [16.0 0.0; 0.0 400.0]), θ) +
    sum(y[i] * log(logistic(θ[1] + θ[2] * x[i])) + (n[i] - y[i]) * log(logistic(-(θ[1] + θ[2] * x[i]))) for i in eachindex(x))

for (name, result) in (
        "laplace" => EA.laplace(logposterior, zeros(2)),
        "moment_matching" => EA.moment_matching(logposterior, zeros(2)),
        "gaussian_variational" => EA.gaussian_variational(logposterior, zeros(2)),
    )
    println(rpad(name, 22), "mean ", round.(mean(result.distribution); digits = 2), "   sd ", round.(sqrt.(var(result.distribution)); digits = 2))
end
```

The posterior of the slope is skewed to the right, so the moment-matched and variational means lie
above Laplace's mode. Moment matching in two dimensions uses ``20^2 = 400`` points by default.
For higher dimensions, pass a cheaper `backend`, such as
[`SphericalRadialCubature`](@ref)`()` with ``2d`` points.

## How each one is computed

- [`laplace`](@ref) runs Newton's method with a backtracking line search from `x0`. A step that
  leaves the region where `logp` is finite is shortened. Where the Hessian is not negative
  definite, at a start where `logp` is not concave, its eigenvalues are reflected and floored, so
  the iteration still climbs.
- [`moment_matching`](@ref) runs `laplace`, then one weighted quadrature, as above. Where Laplace's
  method fails, or the weights are degenerate, it returns the Gaussian the points were placed on
  with `converged = false`.
- [`gaussian_variational`](@ref) writes the bound as a quadrature over a standard Gaussian at
  fixed points, ``\sum_i w_i \log p(m + L z_i) + \log\det L + \text{const}``. That is a smooth
  function of the mean ``m`` and the Cholesky factor ``L``, which Newton's method maximises,
  starting from Laplace's approximation. At the maximum it satisfies the fixed-point conditions of
  conjugate-computation variational inference, which natural-gradient message passing reaches one
  damped step at a time.

All three also work in a constrained domain: see [Constrained variables](@ref).

```@docs
laplace
moment_matching
gaussian_variational
```

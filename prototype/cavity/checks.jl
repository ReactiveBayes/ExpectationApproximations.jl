# Checks of the engine itself: exact where BP is exact, closed edges untouched, and the loop
# among raw sites at one variable (design/cavity_propagation.md, sections 2 and 5).
#
#     julia --project=~/Projects/Julia/universal-dev checks.jl

include(joinpath(@__DIR__, "common.jl"))
import ExpectationApproximations as EA
using SpecialFunctions: loggamma

# 1. A linear-Gaussian chain whose observations are raw log-densities
T, φ, σ², r = 30, 0.9, 0.2, 0.5
rng = StableRNG(3)
x = zeros(T); x[1] = randn(rng)
for t in 2:T
    x[t] = φ * x[t - 1] + sqrt(σ²) * randn(rng)
end
y = x .+ sqrt(r) .* randn(rng, T)

# the exact posterior and evidence, from the dense joint
P = zeros(T, T)
P[1, 1] = 1.0
for t in 2:T
    P[t, t] += 1 / σ²; P[t - 1, t - 1] += φ^2 / σ²; P[t, t - 1] -= φ / σ²; P[t - 1, t] -= φ / σ²
end
Σprior = inv(P)
Σpost = inv(P + I / r)
mpost = Σpost * (y ./ r)
logevidence = logpdf(MvNormal(zeros(T), Symmetric(Σprior + r * I)), y)

println("linear-Gaussian chain, raw observations")
for policy in POLICIES
    graph = scalar_chain(T, 0.0, 1.0, φ, σ², t -> Observation(v -> logpdf(Normal(v[1], sqrt(r)), y[t])))
    result = run!(graph, policy; schedule = :sweep)
    qs = [marginal(graph, t) for t in 1:T]
    δm = maximum(abs, only.(mean.(qs)) - mpost)
    δv = maximum(abs, only.(cov.(qs)) - diag(Σpost))
    check("$(policyname(policy)): the exact posterior", δm < 1.0e-8 && δv < 1.0e-8, @sprintf("max |Δmean| %.1e, max |Δvar| %.1e, %d sweeps", δm, δv, result.iterations))
    F = free_energy(graph)
    check("$(policyname(policy)): free energy = -log p(y)", isapprox(F, -logevidence; rtol = 1.0e-8), @sprintf("%.10f against %.10f", F, -logevidence))
    Flocal = local_free_energy(graph, policy)
    check("$(policyname(policy)): the local free energy is the same", isapprox(Flocal, F; rtol = 1.0e-10), @sprintf("%.10f", Flocal))
    obs = [e for e in graph.edges if e.factor > T]
    δs = maximum(e -> max(abs(e.site.h[1] - y[e.variable] / r), abs(e.site.J[1] - 1 / r)), obs)
    check("$(policyname(policy)): each site is its message", δs < 1.0e-6, @sprintf("max difference %.1e", δs))
end

# 2. The same chain with closed observations: no projection at all
graph = scalar_chain(T, 0.0, 1.0, φ, σ², t -> Prior(MvNormal([y[t]], fill(r, 1, 1))))
run!(graph, ExpectationPropagation(); schedule = :sweep)
δm = maximum(abs, [only(mean(marginal(graph, t))) for t in 1:T] - mpost)
check("closed observations: exact, and no projection", graph.counters.projections == 0 && δm < 1.0e-10, @sprintf("%d projections, max |Δmean| %.1e", graph.counters.projections, δm))
check("closed observations: free energy = -log p(y)", isapprox(free_energy(graph), -logevidence; rtol = 1.0e-10), @sprintf("%.10f", free_energy(graph)))

# 3. One variable with two raw edges: the loop among sites
println("one variable, two raw edges")
logpoisson(k) = v -> k * v[1] - exp(v[1]) - loggamma(k + 1)
logpost(v) = logpdf(Normal(0, 1), v[1]) + logpoisson(3)(v) + logpoisson(1)(v)
function two_raw(policy; damping = 0.0)
    graph = Graph()
    variable!(graph, :x, 1)
    factor!(graph, Prior(MvNormal([0.0], fill(1.0, 1, 1))), :x)
    factor!(graph, Observation(logpoisson(3)), :x)
    factor!(graph, Observation(logpoisson(1)), :x)
    result = run!(graph, policy; damping, tol = 1.0e-12)
    return graph, result
end
graph, result = two_raw(LaplacePropagation())
global_laplace = EA.laplace(logpost, [0.0]).distribution
q = marginal(graph, :x)
δ = max(abs(only(mean(q)) - only(mean(global_laplace))), abs(only(cov(q)) - only(cov(global_laplace))))
check("LP's fixed point is the Laplace approximation of the posterior", δ < 1.0e-8, @sprintf("max difference %.1e, %d iterations", δ, result.iterations))

graph, result = two_raw(ExpectationPropagation(40))
q = marginal(graph, :x)
residual = 0.0
for k in (2, 3)
    edge = graph.edges[k]
    cavity = Cavity.distribution(Cavity.belief(graph, 1) - edge.site)
    _, m, V = EA.tilted_mean_cov(EA.GaussHermite(40), edge.inbound.logf, cavity)
    global residual = max(residual, abs(only(m) - only(mean(q))), abs(only(V) - only(cov(q))))
end
check("EP's fixed point matches the moments of both tilted densities", residual < 1.0e-8, @sprintf("max residual %.1e, %d iterations", residual, result.iterations))
grid = range(-6, 6; length = 20001)
w = exp.(logpost.(grid) .- maximum(logpost.(grid))); w ./= sum(w)
mexact = sum(w .* grid); vexact = sum(w .* (grid .- mexact) .^ 2)
@printf("    exact: mean %.4f var %.4f; EP: %.4f %.4f; Laplace: %.4f %.4f\n", mexact, vexact, only(mean(q)), only(cov(q)), only(mean(global_laplace)), only(cov(global_laplace)))
_, damped = two_raw(ExpectationPropagation(40); damping = 0.5)
@printf("    iterations to 1e-12: undamped %d, damped (β = 0.5) %d\n", result.iterations, damped.iterations)

finish("checks")

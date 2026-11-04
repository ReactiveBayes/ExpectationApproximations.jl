# The Poisson count series of the "Laplace Poisson Count Series" notebook, against its exact grid
# forward-backward: consistent beliefs, the means, and the free energy against -log p(y).
#
#     julia --project=~/Projects/Julia/universal-dev models/counts.jl

include(joinpath(@__DIR__, "..", "common.jl"))
import ExpectationApproximations as EA
using SpecialFunctions: loggamma

φ, σ², T = 0.9, 0.1, 150
rng = StableRNG(42)
x_true = zeros(T)
x_true[1] = rand(rng, Normal(0.5, 1.0))
for t in 2:T
    x_true[t] = φ * x_true[t - 1] + rand(rng, Normal(0.0, sqrt(σ²)))
end
y = [rand(rng, Poisson(exp(x))) for x in x_true]
logpoisson(k, x) = k * x - exp(x) - loggamma(k + 1)

function exact_smoother(y; grid = range(-5, 5, length = 801))
    h = step(grid)
    K = [pdf(Normal(φ * u, sqrt(σ²)), v) * h for u in grid, v in grid]
    likelihood(t) = [exp(logpoisson(y[t], u)) for u in grid]
    α, c = zeros(length(grid), T), zeros(T)
    a = [pdf(Normal(0.5, 1.0), u) * h for u in grid] .* likelihood(1)
    c[1] = sum(a); α[:, 1] = a ./ c[1]
    for t in 2:T
        a = (K' * α[:, t - 1]) .* likelihood(t)
        c[t] = sum(a); α[:, t] = a ./ c[t]
    end
    β = ones(length(grid), T)
    for t in (T - 1):-1:1
        β[:, t] = K * (likelihood(t + 1) .* β[:, t + 1]) ./ c[t + 1]
    end
    P = α .* β
    P ./= sum(P; dims = 1)
    m = [sum(P[:, t] .* grid) for t in 1:T]
    s = [sqrt(sum(P[:, t] .* (grid .- m[t]) .^ 2)) for t in 1:T]
    return (; mean = m, sd = s, logevidence = sum(log, c))
end
exact = exact_smoother(y)

# Two kinds of agreement at the fixed point:
# - each raw edge's projection against its final cavity is the marginal itself;
# - the belief of each transition, the factor times its two cavities, has the variables'
#   marginals as its own marginals (what failed under today's per-product projections).
function disagreement(graph, policy)
    worst_site, worst_joint = 0.0, 0.0
    for edge in graph.edges
        q = Cavity.belief(graph, edge.variable)
        if edge.status === :projected
            p = Cavity.project(policy, edge.inbound, Cavity.distribution(edge.cavity)).q
            worst_site = max(worst_site, abs(only(mean(p)) - only(mean(Cavity.distribution(q)))), abs(only(cov(p)) - only(cov(Cavity.distribution(q)))))
        end
    end
    for node in graph.factors
        node.factor isa LinearGaussian || continue
        e_out, e_in = graph.edges[node.edges[1]], graph.edges[node.edges[2]]
        h, J, _ = Cavity.joint(node.factor)
        joint = Cavity.distribution(NP(h, J) + NP([e_out.cavity.h; e_in.cavity.h], cat(e_out.cavity.J, e_in.cavity.J; dims = (1, 2))))
        for (k, e) in ((1, e_out), (2, e_in))
            q = Cavity.distribution(Cavity.belief(graph, e.variable))
            worst_joint = max(worst_joint, abs(mean(joint)[k] - only(mean(q))), abs(cov(joint)[k, k] - only(cov(q))))
        end
    end
    return worst_site, worst_joint
end

println("Poisson count series, T = $T")
rows = []
for policy in POLICIES
    graph = scalar_chain(T, 0.5, 1.0, φ, σ², t -> Observation(v -> logpoisson(y[t], v[1])))
    result = run!(graph, policy; schedule = :sweep, tol = 1.0e-6)
    qs = [marginal(graph, t) for t in 1:T]
    m, s = first.(scalar.(qs)), last.(scalar.(qs))
    site, joint = disagreement(graph, policy)
    check("$(policyname(policy)): converges", result.converged && graph.counters.failed == 0, @sprintf("%d sweeps", result.iterations))
    check("$(policyname(policy)): every belief agrees with the marginals", site < 1.0e-6 && joint < 1.0e-6, @sprintf("sites %.1e, transitions' joints %.1e", site, joint))
    push!(rows, (policyname(policy), rms(m - exact.mean), maximum(abs, m - exact.mean), maximum(abs, s - exact.sd), free_energy(graph)))
end
@printf("    %-24s %10s %10s %10s %12s\n", "", "RMS Δmean", "max Δmean", "max Δsd", "free energy")
for r in rows
    @printf("    %-24s %10.4f %10.4f %10.4f %12.4f\n", r...)
end
@printf("    %-24s %10s %10s %10s %12.4f\n", "exact, -log p(y)", "", "", "", -exact.logevidence)
println("    (today: Laplace 0.112 and 207.35, CVI 0.028 and 207.74)")
ep = rows[2]
check("EP's means closer to the grid than today's CVI", ep[2] < 0.028, @sprintf("%.4f < 0.028", ep[2]))
check("EP's free energy closer to -log p(y) than today's", abs(ep[5] + exact.logevidence) < abs(207.35 + exact.logevidence), @sprintf("|%.4f - %.4f| = %.4f", ep[5], -exact.logevidence, abs(ep[5] + exact.logevidence)))

finish("count series")

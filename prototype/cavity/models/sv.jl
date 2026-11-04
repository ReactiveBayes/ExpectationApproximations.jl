# Stochastic volatility, the "Laplace Stochastic Volatility" notebook's model and data: Laplace
# propagation's fixed point must be the global Laplace approximation (design, section 3).
#
#     julia --project=~/Projects/Julia/universal-dev models/sv.jl

include(joinpath(@__DIR__, "..", "common.jl"))
import ExpectationApproximations as EA
using StatsFuns: log2π

rng = StableRNG(1234)
T, μ, φ, σ = 200, -1.0, 0.95, 0.3
x_true = zeros(T)
x_true[1] = rand(rng, Normal(0, σ / sqrt(1 - φ^2)))
for t in 2:T
    x_true[t] = φ * x_true[t - 1] + σ * randn(rng)
end
y = [exp((μ + x_true[t]) / 2) * randn(rng) for t in 1:T]
logobs(t) = v -> (s = μ + v[1]; -(log2π + s + y[t]^2 * exp(-s)) / 2)

# the global Laplace approximation and the elliptical slice sampler of the notebook
function logprior(x)
    s = logpdf(Normal(0, σ / sqrt(1 - φ^2)), x[1])
    for t in 2:T
        s += logpdf(Normal(φ * x[t - 1], σ), x[t])
    end
    return s
end
loglikelihood(x) = sum(logobs(t)([x[t]]) for t in 1:T)
q_global = EA.laplace(x -> logprior(x) + loglikelihood(x), zeros(T)).distribution
m_global, s_global = mean(q_global), sqrt.(diag(cov(q_global)))

function prior_draw(rng)
    x = zeros(T)
    x[1] = σ / sqrt(1 - φ^2) * randn(rng)
    for t in 2:T
        x[t] = φ * x[t - 1] + σ * randn(rng)
    end
    return x
end
function elliptical_slice_sampling(rng, x0; nsamples, burnin)
    x, ℓ = copy(x0), loglikelihood(x0)
    s1, s2 = zeros(T), zeros(T)
    for iteration in 1:(nsamples + burnin)
        ν = prior_draw(rng)
        threshold = ℓ + log(rand(rng))
        θ = 2π * rand(rng)
        lo, hi = θ - 2π, θ
        while true
            proposal = x .* cos(θ) .+ ν .* sin(θ)
            ℓproposal = loglikelihood(proposal)
            if ℓproposal > threshold
                x, ℓ = proposal, ℓproposal
                break
            end
            θ < 0 ? (lo = θ) : (hi = θ)
            θ = lo + (hi - lo) * rand(rng)
        end
        if iteration > burnin
            s1 .+= x
            s2 .+= x .^ 2
        end
    end
    m = s1 ./ nsamples
    return m, sqrt.(s2 ./ nsamples .- m .^ 2)
end
m_ess, s_ess = elliptical_slice_sampling(StableRNG(7), m_global; nsamples = 50_000, burnin = 5_000)

build() = scalar_chain(T, 0.0, σ^2 / (1 - φ^2), φ, σ^2, t -> Observation(logobs(t)))

println("stochastic volatility, T = $T")
rows = []
for policy in POLICIES
    graph = build()
    result = time = nothing
    time = @elapsed result = run!(graph, policy; schedule = :sweep, tol = 1.0e-6)
    qs = [marginal(graph, t) for t in 1:T]
    m, s = first.(scalar.(qs)), last.(scalar.(qs))
    parallel = build()
    presult = run!(parallel, policy; schedule = :parallel, tol = 1.0e-6, iterations = 1000)
    if policy isa LaplacePropagation
        δ = max(maximum(abs, m - m_global), maximum(abs, s - s_global))
        check("LP's fixed point is the global Laplace approximation", result.converged && δ < 1.0e-6, @sprintf("max difference %.1e", δ))
    end
    check("$(policyname(policy)): converges, no failed or waiting site at the end", result.converged && parallel.counters.failed == 0, @sprintf("%d sweeps, %d parallel iterations, %.1f s", result.iterations, presult.iterations, time))
    push!(rows, (policyname(policy), rms(m - m_ess), rms(s - s_ess), free_energy(graph), result.iterations, presult.iterations))
end
@printf("    %-24s %10s %10s %12s %7s %9s\n", "", "RMSE mean", "RMSE sd", "free energy", "sweeps", "parallel")
for r in rows
    @printf("    %-24s %10.4f %10.4f %12.4f %7d %9d\n", r...)
end
@printf("    %-24s %10.4f %10.4f\n", "global Laplace", rms(m_global - m_ess), rms(s_global - s_ess))
println("    (against the slice sampler; today's Laplace message passing: 0.177 and 0.019)")
ep = rows[2]
check("EP's means closer to the sampler than the global Laplace", ep[2] < rms(m_global - m_ess), @sprintf("%.4f < %.4f", ep[2], rms(m_global - m_ess)))

finish("stochastic volatility")

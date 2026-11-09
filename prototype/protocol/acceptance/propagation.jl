# Acceptance test of U7a (PLAN.md): the three models of design/cavity_propagation.md, section 10,
# inferred with RxInfer's `fallback = ExpectationPropagation()` and `LaplacePropagation()`. The
# posteriors and the free energy must be the prototype engine's (`prototype/cavity`), which
# computes the same sites by sweeps, where its sweeps converge (the count series and stochastic
# volatility), and they must meet the references: the count series' exact grid, the global
# Laplace approximation and a slice sampler for stochastic volatility, and a particle smoother
# and filter for the pendulum. Run in the universal-dev environment:
#
#     julia --project=~/Projects/Julia/universal-dev propagation.jl

include(joinpath(@__DIR__, "..", "..", "cavity", "Cavity.jl"))
import .Cavity
using RxInfer, BayesBase, Distributions, LinearAlgebra, StableRNGs, Printf
import ExpectationApproximations as EA
using SpecialFunctions: loggamma
using StatsFuns: log2π

passed = true
function check(name, ok, detail)
    global passed &= ok
    @printf("  %-66s %s  %s\n", name, ok ? "ok" : "FAILED", detail)
    flush(stdout)
end
rms(a) = sqrt(sum(abs2, a) / length(a))

const SCHEMES = (
    ("LP", RxInfer.LaplacePropagation(), Cavity.LaplacePropagation()),
    ("EP", RxInfer.ExpectationPropagation(), Cavity.ExpectationPropagation(20)),
)

# The prototype's scalar chain x_1 ~ N(m1, v1), x_t ~ N(φ x_{t-1}, σ²), one raw observation per t
function scalar_chain(T, m1, v1, φ, σ², logobs)
    graph = Cavity.Graph()
    foreach(t -> Cavity.variable!(graph, t, 1), 1:T)
    Cavity.factor!(graph, Cavity.Prior(MvNormal([m1], fill(v1, 1, 1))), 1)
    for t in 2:T
        Cavity.factor!(graph, Cavity.LinearGaussian(fill(φ, 1, 1), [0.0], fill(σ², 1, 1)), t, t - 1)
    end
    foreach(t -> Cavity.factor!(graph, Cavity.Observation(logobs(t)), t), 1:T)
    return graph
end
function prototype(graph, policy, T; kwargs...)
    result = Cavity.run!(graph, policy; schedule = :sweep, tol = 1.0e-9, iterations = 200, kwargs...)
    return (; result, F = Cavity.local_free_energy(graph, policy))
end

# What is compared: the posteriors' means and standard deviations, and the last free energy
function agreement(name, rx, m, s, F)
    δm, δs, δF = maximum(abs, rx.m - m), maximum(abs, rx.s - s), abs(rx.F - F)
    check("$name: the prototype's posteriors", δm < 1.0e-6 && δs < 1.0e-6, @sprintf("max |Δmean| %.1e, max |Δsd| %.1e", δm, δs))
    check("$name: the prototype's free energy", δF < 1.0e-6, @sprintf("%.8f against %.8f", rx.F, F))
    return nothing
end

## 1. The Poisson count series, against its exact grid

struct LogPoisson{X} <: DiscreteUnivariateDistribution
    x::X
end
BayesBase.logpdf(d::LogPoisson, k) = k * d.x - exp(d.x) - loggamma(k + 1)
BayesBase.insupport(::LogPoisson, k) = true
@define_factor_node(node = LogPoisson, type = Stochastic, interfaces = [:out, :x])

@model function counts(y, φ, σ²)
    x[1] ~ NormalMeanVariance(0.5, 1.0)
    y[1] ~ LogPoisson(x[1])
    for t in 2:length(y)
        x[t] ~ NormalMeanVariance(φ * x[t - 1], σ²)
        y[t] ~ LogPoisson(x[t])
    end
end

let φ = 0.9, σ² = 0.1, T = 150
    rng = StableRNG(42)
    x_true = zeros(T); x_true[1] = rand(rng, Normal(0.5, 1.0))
    for t in 2:T
        x_true[t] = φ * x_true[t - 1] + rand(rng, Normal(0.0, sqrt(σ²)))
    end
    y = [rand(rng, Poisson(exp(v))) for v in x_true]
    logpoisson(k, v) = k * v - exp(v) - loggamma(k + 1)

    # forward-backward on a grid
    grid = range(-5, 5, length = 801)
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
    exact = [sum(P[:, t] .* grid) for t in 1:T]
    evidence = sum(log, c)

    println("Poisson count series, T = $T")
    for (name, scheme, policy) in SCHEMES
        graph = scalar_chain(T, 0.5, 1.0, φ, σ², t -> v -> logpoisson(y[t], v[1]))
        p = prototype(graph, policy, T)
        qs = [Cavity.marginal(graph, t) for t in 1:T]
        time = @elapsed result = infer(model = counts(φ = φ, σ² = σ²), data = (y = y,), iterations = 20, fallback = scheme, free_energy = true, returnvars = (x = KeepLast(),))
        rx = (; m = mean.(result.posteriors[:x]), s = std.(result.posteriors[:x]), F = last(result.free_energy))
        agreement(name, rx, only.(mean.(qs)), sqrt.(only.(cov.(qs))), p.F)
        @printf("    %s: RMS Δmean from the grid %.1e, free energy %.4f against -log p(y) %.4f, %.1f s\n", name, rms(rx.m - exact), rx.F, -evidence, time)
        if name == "EP"
            check("EP: the grid's means", rms(rx.m - exact) < 1.0e-3, @sprintf("RMS difference %.1e", rms(rx.m - exact)))
            check("EP: the free energy close to -log p(y)", abs(rx.F + evidence) < 0.1, @sprintf("|%.4f - %.4f|", rx.F, -evidence))
        end
    end
end

## 2. Stochastic volatility: Laplace propagation is the global Laplace approximation, EP is
## closer to the sampler

struct SVObservation{X, M} <: ContinuousUnivariateDistribution
    x::X
    μ::M
end
BayesBase.logpdf(d::SVObservation, y) = (v = d.μ + d.x; -(log2π + v + y^2 * exp(-v)) / 2)
BayesBase.insupport(::SVObservation, y) = true
@define_factor_node(node = SVObservation, type = Stochastic, interfaces = [:out, :x, :μ])

@model function sv(y, μ, φ, σ)
    x[1] ~ NormalMeanVariance(0.0, σ^2 / (1 - φ^2))
    y[1] ~ SVObservation(x[1], μ)
    for t in 2:length(y)
        x[t] ~ NormalMeanVariance(φ * x[t - 1], σ^2)
        y[t] ~ SVObservation(x[t], μ)
    end
end

let T = 200, μ = -1.0, φ = 0.95, σ = 0.3
    rng = StableRNG(1234)
    x_true = zeros(T); x_true[1] = rand(rng, Normal(0, σ / sqrt(1 - φ^2)))
    for t in 2:T
        x_true[t] = φ * x_true[t - 1] + σ * randn(rng)
    end
    y = [exp((μ + x_true[t]) / 2) * randn(rng) for t in 1:T]
    logobs(t) = v -> logpdf(SVObservation(v[1], μ), y[t])
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

    # the elliptical slice sampler of the prototype and the notebook
    function prior_draw(rng)
        x = zeros(T)
        x[1] = σ / sqrt(1 - φ^2) * randn(rng)
        for t in 2:T
            x[t] = φ * x[t - 1] + σ * randn(rng)
        end
        return x
    end
    function slice_sampler(rng, x0; nsamples, burnin)
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
    m_ess, _ = slice_sampler(StableRNG(7), m_global; nsamples = 50_000, burnin = 5_000)

    println("stochastic volatility, T = $T")
    for (name, scheme, policy) in SCHEMES
        graph = scalar_chain(T, 0.0, σ^2 / (1 - φ^2), φ, σ^2, logobs)
        p = prototype(graph, policy, T)
        qs = [Cavity.marginal(graph, t) for t in 1:T]
        time = @elapsed result = infer(model = sv(μ = μ, φ = φ, σ = σ), data = (y = y,), iterations = 20, fallback = scheme, free_energy = true, returnvars = (x = KeepLast(),))
        rx = (; m = mean.(result.posteriors[:x]), s = std.(result.posteriors[:x]), F = last(result.free_energy))
        agreement(name, rx, only.(mean.(qs)), sqrt.(only.(cov.(qs))), p.F)
        @printf("    %s: RMSE of the means against the sampler %.4f (global Laplace %.4f), free energy %.4f, %.1f s\n", name, rms(rx.m - m_ess), rms(m_global - m_ess), rx.F, time)
        if name == "LP"
            δ = max(maximum(abs, rx.m - m_global), maximum(abs, rx.s - s_global))
            check("LP: the global Laplace approximation", δ < 1.0e-6, @sprintf("max difference %.1e", δ))
        else
            check("EP: closer to the sampler than the global Laplace", rms(rx.m - m_ess) < rms(m_global - m_ess), @sprintf("%.4f < %.4f", rms(rx.m - m_ess), rms(m_global - m_ess)))
        end
    end
end

## 3. The pendulum, smoothed through its two functions, against a particle smoother

const Δt, g = 0.05, 9.81
transition(x) = [x[1] + x[2] * Δt, x[2] - g * sin(x[1]) * Δt]
sine(x) = sin(x[1])

@model function pendulum(y, Q, r)
    x[1] ~ MvNormalMeanCovariance([1.5, 0.0], [0.1 0.0; 0.0 0.1])
    s[1] := sine(x[1])
    y[1] ~ NormalMeanVariance(s[1], r)
    for t in 2:length(y)
        x[t] ~ MvNormalMeanCovariance(transition(x[t - 1]), Q)
        s[t] := sine(x[t])
        y[t] ~ NormalMeanVariance(s[t], r)
    end
end

let qc = 0.5, r = 0.1, T = 100
    Q = qc * [Δt^3 / 3 Δt^2 / 2; Δt^2 / 2 Δt]
    prior = MvNormal([1.5, 0.0], [0.1 0.0; 0.0 0.1])
    rng = StableRNG(7)
    x_true = Vector{Vector{Float64}}(undef, T)
    x_true[1] = [1.5, 0.0]
    for t in 2:T
        x_true[t] = rand(rng, MvNormal(transition(x_true[t - 1]), Q))
    end
    y = [rand(rng, Normal(sine(x), sqrt(r))) for x in x_true]
    θ_true = first.(x_true)

    # The graph as RxInfer builds it: x̂ₜ = transition(xₜ₋₁), xₜ ~ N(x̂ₜ, Q), sₜ = sine(xₜ),
    # yₜ ~ N(sₜ, r), with sₜ visited before xₜ in the sweeps
    function build()
        graph, order = Cavity.Graph(), Any[]
        for t in 1:T
            if t > 1
                Cavity.variable!(graph, (:x̂, t), 2); push!(order, (:x̂, t))
            end
            Cavity.variable!(graph, (:s, t), 1); push!(order, (:s, t))
            Cavity.variable!(graph, (:x, t), 2); push!(order, (:x, t))
        end
        Cavity.factor!(graph, Cavity.Prior(prior), (:x, 1))
        for t in 1:T
            if t > 1
                Cavity.factor!(graph, Cavity.Delta(transition), (:x̂, t), (:x, t - 1))
                Cavity.factor!(graph, Cavity.LinearGaussian(Matrix(1.0I, 2, 2), zeros(2), Q), (:x, t), (:x̂, t))
            end
            Cavity.factor!(graph, Cavity.Delta(sine), (:s, t), (:x, t))
            Cavity.factor!(graph, Cavity.Prior(MvNormal([y[t]], fill(r, 1, 1))), (:s, t))
        end
        return graph, order
    end

    # forward filtering, backward simulation of whole trajectories
    function particle_smoother(; n = 5000, m = 500, rng = StableRNG(1))
        L, W = cholesky(Q).L, inv(Q)
        particles, logweights = zeros(2, n, T), zeros(n, T)
        current = rand(rng, prior, n)
        for t in 1:T
            if t > 1
                ancestors = rand(rng, Categorical(exp.(logweights[:, t - 1])), n)
                for j in 1:n
                    current[:, j] = transition(particles[:, ancestors[j], t - 1]) + L * randn(rng, 2)
                end
            end
            particles[:, :, t] = current
            lw = [logpdf(Normal(sin(current[1, j]), sqrt(r)), y[t]) for j in 1:n]
            lw .-= maximum(lw)
            lw .-= log(sum(exp, lw))
            logweights[:, t] = lw
        end
        predicted = zeros(2, n, T)
        for t in 1:T, j in 1:n
            predicted[:, j, t] = transition(particles[:, j, t])
        end
        s1 = zeros(T)
        lw, w = zeros(n), zeros(n)
        for _ in 1:m
            j = rand(rng, Categorical(exp.(logweights[:, T])))
            x1, x2 = particles[1, j, T], particles[2, j, T]
            s1[T] += x1
            for t in (T - 1):-1:1
                for i in 1:n
                    d1, d2 = x1 - predicted[1, i, t], x2 - predicted[2, i, t]
                    lw[i] = logweights[i, t] - (W[1, 1] * d1^2 + 2 * W[1, 2] * d1 * d2 + W[2, 2] * d2^2) / 2
                end
                w .= exp.(lw .- maximum(lw))
                u, acc, j = rand(rng) * sum(w), 0.0, n
                for i in 1:n
                    acc += w[i]
                    if acc >= u
                        j = i
                        break
                    end
                end
                x1, x2 = particles[1, j, t], particles[2, j, t]
                s1[t] += x1
            end
        end
        return s1 ./ m
    end
    # -log p(y) by a bootstrap particle filter, averaged over seeds
    function pf_free_energy(seed; n = 50_000)
        rng = StableRNG(seed)
        L = cholesky(Q).L
        particles = rand(rng, prior, n)
        logZ = 0.0
        for t in 1:T
            if t > 1
                for j in 1:n
                    particles[:, j] = transition(particles[:, j]) + L * randn(rng, 2)
                end
            end
            lw = [logpdf(Normal(sin(particles[1, j]), sqrt(r)), y[t]) for j in 1:n]
            top = maximum(lw)
            w = exp.(lw .- top)
            logZ += top + log(sum(w) / n)
            particles = particles[:, rand(rng, Categorical(w ./ sum(w)), n)]
        end
        return -logZ
    end
    m_ps = particle_smoother()
    estimates = [pf_free_energy(seed) for seed in 1:4]
    F_pf = sum(estimates) / 4

    println("pendulum, T = $T")
    # The prototype's sweeps do not settle here: under LP they oscillate far from the particle
    # smoother, and under EP, where it waits on improper cavities, they stall at a change of
    # about 1e-7 per sweep. So the pendulum is compared with the references, not to 1e-6 with the
    # prototype: RxInfer must settle, and both schemes must meet the particle smoother and filter.
    for (name, scheme, policy) in SCHEMES
        graph, order = build()
        p = prototype(graph, policy, T; order)
        pm = [mean(Cavity.distribution(Cavity.belief(graph, graph.names[(:x, t)])))[1] for t in 1:T]
        time = @elapsed result = infer(model = pendulum(Q = Q, r = r), data = (y = y,), iterations = 30, fallback = scheme, free_energy = true, returnvars = (x = KeepLast(),))
        rx = (; m = [mean(q)[1] for q in result.posteriors[:x]], F = result.free_energy)
        @printf("    %s: the prototype %s after %d sweeps (last change %.1e), RMS difference from the particle smoother %.3f, free energy %.4f\n", name, p.result.converged ? "converged" : "did not converge", p.result.iterations, last(p.result.history), rms(pm - m_ps), p.F)
        @printf("    %s: RMSE %.3f, RMS difference from the particle smoother %.3f (its RMSE %.3f), free energy %.4f against %.4f ± %.4f, %.1f s\n", name, rms(rx.m - θ_true), rms(rx.m - m_ps), rms(m_ps - θ_true), last(rx.F), F_pf, std(estimates), time)
        δF = abs(rx.F[end] - rx.F[end - 10])
        check("$name: the free energy settles", δF < 1.0e-6, @sprintf("|F₃₀ - F₂₀| %.1e", δF))
        check("$name: close to the particle smoother", rms(rx.m - m_ps) < 0.05, @sprintf("RMS difference %.3f", rms(rx.m - m_ps)))
        check("$name: the free energy close to the particle filter's -log p(y)", abs(last(rx.F) - F_pf) < 0.2, @sprintf("|%.4f - %.4f|", last(rx.F), F_pf))
    end
end

println(passed ? "propagation: PASSED" : "propagation: FAILED")
passed || exit(1)

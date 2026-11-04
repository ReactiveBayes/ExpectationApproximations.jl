# The pendulum of the "Laplace Pendulum" notebook, smoothed: today's fallback can only filter it,
# because the messages backwards through `transition` and `sine` have no Gaussian of their own.
# Against a particle smoother (forward filtering, backward simulation), with the open questions
# O1 (a site for the forward direction of a Delta node), O2 (skip or clip an improper marginal)
# and O3 (improper outbound messages).
#
#     julia --project=~/Projects/Julia/universal-dev models/pendulum.jl

include(joinpath(@__DIR__, "..", "common.jl"))

const Δt, g = 0.05, 9.81
transition(x) = [x[1] + x[2] * Δt, x[2] - g * sin(x[1]) * Δt]
sine(x) = sin(x[1])
const qc, r, T = 0.5, 0.1, 100
const Q = qc * [Δt^3/3 Δt^2/2; Δt^2/2 Δt]
rng = StableRNG(7)
x_true = Vector{Vector{Float64}}(undef, T)
x_true[1] = [1.5, 0.0]
for t in 2:T
    x_true[t] = rand(rng, MvNormal(transition(x_true[t - 1]), Q))
end
const y = [rand(rng, Normal(sine(x), sqrt(r))) for x in x_true]
θ_true = first.(x_true)
const prior = MvNormal([1.5, 0.0], [0.1 0.0; 0.0 0.1])

# The graph as RxInfer builds it: x̂ₜ = transition(xₜ₋₁), xₜ ~ N(x̂ₜ, Q), sₜ = sine(xₜ), yₜ ~ N(sₜ, r)
function build(; forward_site = false, n = 20)
    graph = Graph()
    order = Any[]
    for t in 1:T
        if t > 1
            variable!(graph, (:x̂, t), 2); push!(order, (:x̂, t))
        end
        # sₜ before xₜ: its outbound message to `sine` is the observation's, which xₜ's
        # pulled-back message needs
        variable!(graph, (:s, t), 1); push!(order, (:s, t))
        variable!(graph, (:x, t), 2); push!(order, (:x, t))
    end
    factor!(graph, Prior(prior), (:x, 1))
    for t in 1:T
        if t > 1
            factor!(graph, Delta(transition; forward_site, n), (:x̂, t), (:x, t - 1))
            factor!(graph, LinearGaussian(Matrix(1.0I, 2, 2), zeros(2), Q), (:x, t), (:x̂, t))
        end
        factor!(graph, Delta(sine; forward_site, n), (:s, t), (:x, t))
        factor!(graph, Prior(MvNormal([y[t]], fill(r, 1, 1))), (:s, t))
    end
    return graph, order
end

# The particle smoother: a bootstrap filter, then backward simulation of whole trajectories
function particle_smoother(; n = 5000, m = 500, rng = StableRNG(1))
    L, W = cholesky(Q).L, inv(Q)
    particles = zeros(2, n, T)
    logweights = zeros(n, T)
    current = rand(rng, prior, n)
    filtered = zeros(T)
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
        filtered[t] = sum(exp.(lw) .* current[1, :])
    end
    predicted = zeros(2, n, T)
    for t in 1:T, j in 1:n
        predicted[:, j, t] = transition(particles[:, j, t])
    end
    s1, s2 = zeros(T), zeros(T)
    lw, w = zeros(n), zeros(n)
    for _ in 1:m
        j = rand(rng, Categorical(exp.(logweights[:, T])))
        x1, x2 = particles[1, j, T], particles[2, j, T]
        s1[T] += x1; s2[T] += x1^2
        for t in (T - 1):-1:1
            for i in 1:n
                d1, d2 = x1 - predicted[1, i, t], x2 - predicted[2, i, t]
                lw[i] = logweights[i, t] - (W[1, 1] * d1^2 + 2 * W[1, 2] * d1 * d2 + W[2, 2] * d2^2) / 2
            end
            top = maximum(lw)
            w .= exp.(lw .- top)
            # inverse-CDF draw from the unnormalised weights
            u, acc, j = rand(rng) * sum(w), 0.0, n
            for i in 1:n
                acc += w[i]
                if acc >= u
                    j = i
                    break
                end
            end
            x1, x2 = particles[1, j, t], particles[2, j, t]
            s1[t] += x1; s2[t] += x1^2
        end
    end
    mean_θ = s1 ./ m
    return (; mean = mean_θ, sd = sqrt.(max.(s2 ./ m .- mean_θ .^ 2, 0.0)), filtered)
end
reference = @time particle_smoother()

coverage(m, s) = sum(abs.(m - θ_true) .<= 2s) / T
println("pendulum, T = $T: the particle smoother's RMSE against the truth ", @sprintf("%.3f", rms(reference.mean - θ_true)))

configurations = [
    ("LaplacePropagation", LaplacePropagation(), (;), :skip),
    ("ExpectationPropagation", ExpectationPropagation(20), (;), :skip),
    ("CVIPropagation", CVIPropagation(), (;), :skip),
    ("EP, points on the cavity", ExpectationPropagation(20; center = :cavity), (;), :skip),
    ("EP, forward sites (O1)", ExpectationPropagation(20), (; forward_site = true), :skip),
    ("EP, clip (O2)", ExpectationPropagation(20), (;), :clip),
]
rows = []
results = Dict{String, Any}()
for (name, policy, options, improper) in configurations
    graph, order = build(; options...)
    time = @elapsed result = run!(graph, policy; schedule = :sweep, order, tol = 1.0e-6, iterations = 60, improper)
    beliefs = [Cavity.belief(graph, graph.names[(:x, t)]) for t in 1:T]
    proper = isproper.(beliefs)
    all(proper) || println("    $name: the marginals of x at t = ", findall(!, proper), " are improper")
    m = [p ? mean(Cavity.distribution(b))[1] : NaN for (b, p) in zip(beliefs, proper)]
    s = [p ? sqrt(cov(Cavity.distribution(b))[1, 1]) : NaN for (b, p) in zip(beliefs, proper)]
    results[name] = (; m, s, graph, result)
    c = graph.counters
    result.converged || println("    $name: the last sweeps' largest change ", @sprintf("%.1e", last(result.history)))
    isempty(c.improper_receivers) || println("    $name: indefinite outbound messages went to ", c.improper_receivers)
    push!(rows, (name, rms(m - θ_true), rms(m - reference.mean), coverage(m, s), result.converged ? result.iterations : -result.iterations, c.failed, c.improper_marginals, c.improper_cavities, time))
end
@printf("    %-24s %8s %9s %9s %7s %7s %10s %10s %7s\n", "", "RMSE", "RMS Δ PS", "coverage", "sweeps", "failed", "improper q", "improper c", "time")
for row in rows
    @printf("    %-24s %8.3f %9.3f %9.2f %7d %7d %10d %10d %6.1fs\n", row...)
end
@printf("    %-24s %8.3f %9.3f %9.2f\n", "particle smoother", rms(reference.mean - θ_true), 0.0, coverage(reference.mean, reference.sd))
println("    (sweeps < 0: not converged after that many; improper q/c: marginals/outbound messages found improper, counted per visit)")

graph, _ = build()
parallel = run!(graph, ExpectationPropagation(20); schedule = :parallel, tol = 1.0e-6, iterations = 2000)
println("    EP under the parallel schedule: ", parallel.converged ? "$(parallel.iterations) iterations" : "not converged after $(parallel.iterations)")

ep = results["ExpectationPropagation"]
# The free energy: the global form needs every cavity's normaliser, and some are improper here;
# the local form (`local_free_energy`) needs none. Against the particle filter's -log p(y).
function pf_logevidence(seed; n = 50_000)
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
    return logZ
end
estimates = [pf_logevidence(seed) for seed in 1:4]
F_pf, F_ep = -sum(estimates) / 4, local_free_energy(results["ExpectationPropagation"].graph, ExpectationPropagation(20))
check("EP's free energy is close to the particle filter's -log p(y)", abs(F_ep - F_pf) < 0.2, @sprintf("%.4f against %.4f ± %.4f", F_ep, F_pf, std(estimates)))

check("EP smooths, every site settled", ep.result.converged && ep.graph.counters.failed == 0, @sprintf("%d sweeps", ep.result.iterations))
check("EP's smoothed angle is close to the particle smoother", rms(ep.m - reference.mean) < 0.15, @sprintf("RMS difference %.3f", rms(ep.m - reference.mean)))

# Today's way, for comparison: RxInfer on the whole model, unscented by hand and the fallback
using RxInfer
@model function pendulum_whole(y, Q, r)
    x[1] ~ MvNormalMeanCovariance([1.5, 0.0], [0.1 0.0; 0.0 0.1])
    s[1] := sine(x[1])
    y[1] ~ NormalMeanVariance(s[1], r)
    for t in 2:length(y)
        x[t] ~ MvNormalMeanCovariance(transition(x[t - 1]), Q)
        s[t] := sine(x[t])
        y[t] ~ NormalMeanVariance(s[t], r)
    end
end
for (name, kwargs) in (
        ("RxInfer, unscented by hand", (; algorithm = @algorithm(begin
            transition() -> Unscented()
            sine() -> Unscented()
        end), initialization = @initialization(begin
            μ(x) = MvNormalMeanCovariance([0.0, 0.0], [10.0 0.0; 0.0 10.0])
        end), iterations = 10)),
        ("RxInfer, fallback = CVI()", (; fallback = CVI())),
    )
    try
        result = infer(; model = pendulum_whole(Q = Q, r = r), data = (y = y,), returnvars = (x = KeepLast(),), kwargs...)
        m = [mean(q)[1] for q in result.posteriors[:x]]
        s = [sqrt(cov(q)[1, 1]) for q in result.posteriors[:x]]
        @printf("    %-24s %8.3f %9.3f %9.2f\n", name, rms(m - θ_true), rms(m - reference.mean), coverage(m, s))
    catch error
        println("    $name: ", first(sprint(showerror, error), 160))
    end
end

finish("pendulum")

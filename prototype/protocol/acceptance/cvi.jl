# Acceptance test of U7b (PLAN.md): `fallback = CVI()`, the variational Gaussian as the projection,
# and ExponentialFamilyProjection's deterministic `ExpectationStrategy` for a family beyond the
# Gaussian. Run in the universal-dev environment:
#
#     julia --project=~/Projects/Julia/universal-dev cvi.jl

using RxInfer, BayesBase, Distributions, LinearAlgebra, StatsFuns, SpecialFunctions, Printf
using ExponentialFamilyProjection
import ExpectationApproximations as EA

passed = true
function check(name, ok, detail)
    global passed &= ok
    @printf("  %-62s %s  %s\n", name, ok ? "ok" : "FAILED", detail)
end

# 1. The bioassay: CVI's posterior is the variational Gaussian of the hand-written joint, and its
#    free energy, the negative bound, is below Laplace's and above -log p(y)
logbinomial(n, k) = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
struct BinomialLogistic{T, X, N} <: DiscreteUnivariateDistribution
    θ::T
    x::X
    n::N
end
BayesBase.logpdf(d::BinomialLogistic, y) = (η = d.θ[1] + d.θ[2] * d.x; -y * log1pexp(-η) - (d.n - y) * log1pexp(η) + logbinomial(d.n, y))
BayesBase.insupport(::BinomialLogistic, y) = true
@define_factor_node(node = BinomialLogistic, type = Stochastic, interfaces = [:out, :θ, :x, :n])

x, n, y = [-0.86, -0.30, -0.05, 0.73], [5, 5, 5, 5], [0, 1, 3, 5]
prior_θ = MvNormalMeanCovariance(zeros(2), diagm([16.0, 400.0]))
@model function bioassay(y, x, n)
    θ ~ prior_θ
    for i in eachindex(y)
        y[i] ~ BinomialLogistic(θ, x[i], n[i])
    end
end
logposterior(θ) = logpdf(prior_θ, θ) + sum(logpdf(BinomialLogistic(θ, x[i], n[i]), y[i]) for i in eachindex(y))

println("bioassay")
cvi = infer(model = bioassay(x = x, n = n), data = (y = y,), fallback = CVI(), free_energy = true)
laplace = infer(model = bioassay(x = x, n = n), data = (y = y,), fallback = LaplaceApproximation(), free_energy = true)
reference = EA.gaussian_variational(logposterior, zeros(2))
q = cvi.posteriors[:θ]
δ = max(maximum(abs, mean(q) - mean(reference.distribution)) / norm(mean(reference.distribution)), maximum(abs, cov(q) - cov(reference.distribution)) / norm(cov(reference.distribution)))
check("CVI equals gaussian_variational of the joint", δ < 1.0e-8, @sprintf("max relative difference %.1e", δ))
αs, βs = range(-4, 6; length = 400), range(-10, 40; length = 400)
grid = [logposterior([α, β]) for α in αs, β in βs]
logevidence = maximum(grid) + log(sum(exp.(grid .- maximum(grid))) * step(αs) * step(βs))
fcvi, flap = only(cvi.free_energy), only(laplace.free_energy)
check("-log p(y) < free energy of CVI < Laplace's", -logevidence < fcvi < flap, @sprintf("%.3f < %.3f < %.3f", -logevidence, fcvi, flap))
P = exp.(grid .- maximum(grid)); P ./= sum(P)
Eβ = sum(P .* [β for α in αs, β in βs])
check("E[β] of CVI closer to the grid than Laplace's", abs(mean(q)[2] - Eβ) < abs(mean(laplace.posteriors[:θ])[2] - Eβ), @sprintf("grid %.2f, CVI %.2f, Laplace %.2f", Eβ, mean(q)[2], mean(laplace.posteriors[:θ])[2]))

# 2. The zero-inflated Poisson under mean field: CVI's free energy below Laplace's
struct ZeroInflatedPoisson{P, L} <: DiscreteUnivariateDistribution
    p::P
    λ::L
end
BayesBase.logpdf(d::ZeroInflatedPoisson, k) = k == 0 ? logaddexp(log(d.p), log1p(-d.p) - d.λ) : log1p(-d.p) + k * log(d.λ) - d.λ - loggamma(k + 1)
BayesBase.insupport(::ZeroInflatedPoisson, k) = true
@define_factor_node(node = ZeroInflatedPoisson, type = Stochastic, interfaces = [:out, :p, :λ])
@model function monastery(y)
    p ~ Beta(1.0, 1.0)
    λ ~ Gamma(2.0, 1.0)
    for i in eachindex(y)
        y[i] ~ ZeroInflatedPoisson(p, λ)
    end
end
counts = [0, 2, 0, 1, 0, 0, 3, 1, 0, 2, 0, 1, 0, 0, 1, 2, 0, 0, 1, 0]
mean_field = @constraints begin
    q(p, λ) = q(p)q(λ)
end
println("zero-inflated Poisson, mean field")
run(fallback) = infer(model = monastery(), data = (y = counts,), constraints = mean_field, iterations = 30, fallback = fallback, free_energy = true)
fz_cvi, fz_lap = last(run(CVI()).free_energy), last(run(LaplaceApproximation()).free_energy)
check("free energy of CVI below Laplace's", fz_cvi < fz_lap, @sprintf("%.4f < %.4f", fz_cvi, fz_lap))

# 3. A Beta projection by quadrature: a conjugate Beta-Bernoulli posterior through a rule-free
#    likelihood is projected onto itself
struct Flip{P} <: DiscreteUnivariateDistribution
    p::P
end
BayesBase.logpdf(d::Flip, y) = y * log(d.p) + (1 - y) * log1p(-d.p)
BayesBase.insupport(::Flip, y) = true
@define_factor_node(node = Flip, type = Stochastic, interfaces = [:out, :p])
@model function coin(y)
    p ~ Beta(2.0, 2.0)
    y .~ Flip(p)
end
flips = Float64[1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
k = count(isone, flips)
projection = ProjectedTo(Beta; parameters = ProjectionParameters(strategy = ExpectationStrategy(EA.EADefault()), niterations = 2000, tolerance = 1.0e-10))
println("Beta projection")
qp = infer(model = coin(), data = (y = flips,), fallback = LaplaceApproximation(), options = (default_form_constraint = (q = projection,),)).posteriors[:p]
target = (2.0 + k, 2.0 + length(flips) - k)
check("equals the conjugate Beta", all(isapprox.(params(qp), target; rtol = 1.0e-3)), @sprintf("Beta(%.4f, %.4f) against Beta(%g, %g)", params(qp)..., target...))

println(passed ? "CVI: PASSED" : "CVI: FAILED")
passed || exit(1)

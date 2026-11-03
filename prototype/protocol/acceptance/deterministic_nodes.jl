# Acceptance test of D48 and D49 (PLAN.md, U6 "One fallback for every deterministic node"): the
# bioassay of the logistic regression notebook written as mathematics, its Binomials expanded into
# Bernoulli trials, through `fallback = LaplaceApproximation()`. Its deterministic nodes, `dot` and
# a Delta node, go through the one rule fallback wherever their own rules do not apply, and the
# posterior must be the fused likelihood node's and Laplace's of the joint. Without the free
# energy: Bernoulli's average energy needs q(p) on (0, 1), the range of a Delta node's output,
# which waits until after the merge. Run in the universal-dev environment:
#
#     julia --project=~/Projects/Julia/universal-dev deterministic_nodes.jl

using RxInfer, BayesBase, Distributions, LinearAlgebra, StatsFuns, SpecialFunctions, Printf
import ReactiveMP, MessagePassingRulesBase
import ExpectationApproximations as EA

x = [-0.86, -0.30, -0.05, 0.73]
n = [5, 5, 5, 5]
y = [0, 1, 3, 5]
prior_θ = MvNormalMeanCovariance(zeros(2), diagm([4.0^2, 20.0^2]))

# The trials, y[i] successes out of n[i] at dose x[i]
doses = reduce(vcat, [fill(x[i], n[i]) for i in eachindex(x)])
outcomes = reduce(vcat, [[ones(y[i]); zeros(n[i] - y[i])] for i in eachindex(x)])

@model function bioassay_dot(y, doses, prior)
    θ ~ prior
    for j in eachindex(y)
        p[j] := logistic(dot([1.0, doses[j]], θ))
        y[j] ~ Bernoulli(p[j])
    end
end

dose_response(θ, dose) = logistic(θ[1] + θ[2] * dose)

@model function bioassay_delta(y, doses, prior)
    θ ~ prior
    for j in eachindex(y)
        p[j] := dose_response(θ, doses[j])
        y[j] ~ Bernoulli(p[j])
    end
end

# The notebook's fused node: the Binomial likelihood of a group of trials
logbinomial(n, k) = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
struct BinomialLogistic{T, X, N} <: DiscreteUnivariateDistribution
    θ::T
    x::X
    n::N
end
function BayesBase.logpdf(d::BinomialLogistic, y)
    η = d.θ[1] + d.θ[2] * d.x
    return -y * log1pexp(-η) - (d.n - y) * log1pexp(η) + logbinomial(d.n, y)
end
BayesBase.insupport(::BinomialLogistic, y) = true
@define_factor_node(node = BinomialLogistic, type = Stochastic, interfaces = [:out, :θ, :x, :n])

@model function bioassay_fused(y, x, n, prior)
    θ ~ prior
    for i in eachindex(y)
        y[i] ~ BinomialLogistic(θ, x[i], n[i])
    end
end

logposterior(θ) = logpdf(prior_θ, θ) + sum(logpdf(BinomialLogistic(θ, x[i], n[i]), y[i]) for i in eachindex(y))
reference = EA.laplace(logposterior, zeros(2)).distribution

# Which rule gave each message: a node's own, or the fallback
calls = Dict{String, Int}()
function record(event)
    node = ReactiveMP.functionalform(event.mapping.factornode)
    by = event.rule isa MessagePassingRulesBase.RuleSpec ? "its rule" : string(nameof(typeof(event.rule)))
    key = string(node, " → ", event.mapping.target, ": ", by)
    calls[key] = get(calls, key, 0) + 1
    return nothing
end

difference(q, r) = max(maximum(abs, mean(q) - mean(r)) / maximum(abs, mean(r)), maximum(abs, cov(q) - cov(r)) / maximum(abs, cov(r)))

results = [
    ("p := logistic(dot([1, x], θ))", bioassay_dot(doses = doses, prior = prior_θ), (y = outcomes,)),
    ("p := dose_response(θ, x)", bioassay_delta(doses = doses, prior = prior_θ), (y = outcomes,)),
    ("the fused node", bioassay_fused(x = x, n = n, prior = prior_θ), (y = y,)),
]

passed = true
@printf("Laplace of the joint: E[θ] = %s\n", round.(mean(reference); digits = 4))
for (name, model, data) in results
    empty!(calls)
    result = infer(model = model, data = data, fallback = LaplaceApproximation(), callbacks = (after_message_rule_call = record,))
    q = result.posteriors[:θ]
    δ = difference(q, reference)
    ok = δ <= 1.0e-8
    global passed &= ok
    @printf("%-32s E[θ] = %-22s max relative difference to Laplace %.1e  %s\n", name, string(round.(mean(q); digits = 4)), δ, ok ? "ok" : "FAILED")
    for (key, count) in sort(collect(calls))
        @printf("    %3d× %s\n", count, key)
    end
end

println(passed ? "deterministic nodes: PASSED" : "deterministic nodes: FAILED")
passed || exit(1)

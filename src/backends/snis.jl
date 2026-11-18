"""
    SNIS(proposal, n, rng = nothing)

Self-normalised importance sampling: `n` samples from `proposal`, weighted by the ratio of the
target density to the proposal density and normalised. It is for a target that can be evaluated
but not sampled, such as the unnormalised product of a prior and messages. Labelled
`ExpectationStochastic`. Unless `rng` is given here, the random number generator comes from
`BayesBase.with_expectation_rng`. The weights are computed in log space, and
[`effective_sample_size`](@ref) estimates how many samples they are worth.
"""
struct SNIS{P, R}
    proposal::P
    n::Int
    rng::R

    function SNIS(proposal::P, n::Integer, rng::R) where {P, R}
        n > 0 || throw(ArgumentError("importance sampling needs at least one sample, got $n"))
        return new{P, R}(proposal, n, rng)
    end
end

SNIS(proposal, n::Integer) = SNIS(proposal, n, nothing)

BayesBase.has_expectation_label(::SNIS, ::ExpectationStochastic) = true
# Targets with a log-density; a product of independent distributions is not a density to evaluate
const Evaluable = Union{Distribution, ProductOf, LinearizedProductOf}

BayesBase.supports_expectation(::SNIS, ::Type, ::Type{<:Evaluable}) = true
BayesBase.get_expectation_rng(sampler::SNIS) = sampler.rng
BayesBase.with_expectation_rng(sampler::SNIS{P, Nothing}, rng) where {P} = SNIS(sampler.proposal, sampler.n, rng)
BayesBase.with_expectation_rng(sampler::SNIS, rng) = sampler

function BayesBase.prepare_expectation(sampler::SNIS, target::Evaluable)
    isnothing(sampler.rng) && throw(ArgumentError("SNIS has no random number generator; provide one with `BayesBase.with_expectation_rng`"))
    samples = [rand(sampler.rng, sampler.proposal) for _ in 1:(sampler.n)]
    logweights = [logpdf(target, x) - logpdf(sampler.proposal, x) for x in samples]
    weights = exp.(logweights .- maximum(logweights))
    return ExpectationPointSet(samples, weights ./ sum(weights))
end

"""
    effective_sample_size(prepared::ExpectationPointSet)

`1 / Σᵢ wᵢ²` for normalised weights: the number of equally weighted samples that the weighted
ones are worth. It is `n` when the proposal is the target, and much smaller when the proposal
fits the target poorly.
"""
effective_sample_size(prepared::ExpectationPointSet) = sum(prepared.weights)^2 / sum(abs2, prepared.weights)

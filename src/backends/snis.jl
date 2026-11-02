"""
    SNIS(proposal, n, rng = nothing)

Self-normalised importance sampling: `n` samples from `proposal`, weighted by the ratio of the
target's density to the proposal's and normalised, for a target that can be evaluated but not
sampled, such as an unnormalised product of a prior and messages. Labelled `Stochastic`; the
random number generator comes from `BayesBase.withrng` unless set here. The weights are computed
in log space, and [`effective_sample_size`](@ref) tells how many samples they are worth.
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

BayesBase.haslabel(::SNIS, ::Stochastic) = true
# Targets with a log-density; a product of independent distributions is not a density to evaluate
const Evaluable = Union{Distribution, ProductOf, LinearizedProductOf}

BayesBase.supports(::SNIS, ::Type, ::Type{<:Evaluable}) = true
BayesBase.getrng(sampler::SNIS) = sampler.rng
BayesBase.withrng(sampler::SNIS{P, Nothing}, rng) where {P} = SNIS(sampler.proposal, sampler.n, rng)
BayesBase.withrng(sampler::SNIS, rng) = sampler

function BayesBase.prepare_expectation(sampler::SNIS, target::Evaluable)
    isnothing(sampler.rng) && throw(ArgumentError("SNIS has no random number generator; provide one with `BayesBase.withrng`"))
    samples = [rand(sampler.rng, sampler.proposal) for _ in 1:(sampler.n)]
    logweights = [logpdf(target, x) - logpdf(sampler.proposal, x) for x in samples]
    weights = exp.(logweights .- maximum(logweights))
    return PointSet(samples, weights ./ sum(weights))
end

"""
    effective_sample_size(prepared::PointSet)

`1 / Σᵢ wᵢ²` for normalised weights: the number of equally weighted samples that the weighted
ones are worth, `n` when the proposal is the target and far fewer when it fits poorly.
"""
effective_sample_size(prepared::PointSet) = sum(prepared.weights)^2 / sum(abs2, prepared.weights)

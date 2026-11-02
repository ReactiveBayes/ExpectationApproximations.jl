"""
    MonteCarlo(n, rng = nothing)

The average over `n` samples, drawn with `rng` when the expectation is prepared, for any
distribution that can be sampled. Labelled `ExpectationStochastic`. Without a random number
generator it refuses to run; `BayesBase.with_expectation_rng` provides one from above and never
replaces one set here.
"""
struct MonteCarlo{R}
    n::Int
    rng::R

    function MonteCarlo(n::Integer, rng::R) where {R}
        n > 0 || throw(ArgumentError("Monte Carlo needs at least one sample, got $n"))
        return new{R}(n, rng)
    end
end

MonteCarlo(n::Integer) = MonteCarlo(n, nothing)

BayesBase.has_expectation_label(::MonteCarlo, ::ExpectationStochastic) = true
BayesBase.supports_expectation(::MonteCarlo, ::Type, ::Type{<:Distribution}) = true
BayesBase.get_expectation_rng(sampler::MonteCarlo) = sampler.rng
BayesBase.with_expectation_rng(sampler::MonteCarlo{Nothing}, rng) = MonteCarlo(sampler.n, rng)
BayesBase.with_expectation_rng(sampler::MonteCarlo, rng) = sampler

function BayesBase.prepare_expectation(sampler::MonteCarlo, q::Distribution)
    isnothing(sampler.rng) && throw(ArgumentError("MonteCarlo has no random number generator; provide one with `BayesBase.with_expectation_rng`"))
    samples = [rand(sampler.rng, q) for _ in 1:(sampler.n)]
    return ExpectationPointSet(samples, fill(1 / sampler.n, sampler.n))
end

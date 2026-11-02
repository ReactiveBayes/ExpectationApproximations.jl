"""
    MonteCarlo(n, rng = nothing)

The average over `n` samples, drawn with `rng` when the expectation is prepared, for any
distribution that can be sampled. Labelled `Stochastic`. Without a random number generator it
refuses to run; `BayesBase.withrng` provides one from above and never replaces one set here.
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

BayesBase.haslabel(::MonteCarlo, ::Stochastic) = true
BayesBase.supports(::MonteCarlo, ::Type, ::Type{<:Distribution}) = true
BayesBase.getrng(sampler::MonteCarlo) = sampler.rng
BayesBase.withrng(sampler::MonteCarlo{Nothing}, rng) = MonteCarlo(sampler.n, rng)
BayesBase.withrng(sampler::MonteCarlo, rng) = sampler

function BayesBase.prepare_expectation(sampler::MonteCarlo, q::Distribution)
    isnothing(sampler.rng) && throw(ArgumentError("MonteCarlo has no random number generator; provide one with `BayesBase.withrng`"))
    samples = [rand(sampler.rng, q) for _ in 1:(sampler.n)]
    return PointSet(samples, fill(1 / sampler.n, sampler.n))
end

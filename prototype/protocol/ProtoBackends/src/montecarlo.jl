"""
    MonteCarlo(n, rng = nothing)

The average over `n` samples, drawn with `rng` when the expectation is prepared. Stochastic.
Without an RNG it refuses to run; [`withrng`](@ref ProtoBase.withrng) provides one from above.
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

ProtoBase.haslabel(::MonteCarlo, ::Stochastic) = true
ProtoBase.supports(::MonteCarlo, ::Type, ::Type{<:Distribution}) = true
ProtoBase.getrng(sampler::MonteCarlo) = sampler.rng
ProtoBase.withrng(sampler::MonteCarlo{Nothing}, rng) = MonteCarlo(sampler.n, rng)
ProtoBase.withrng(sampler::MonteCarlo, rng) = sampler

function ProtoBase.prepare_expectation(sampler::MonteCarlo, q::Distribution)
    isnothing(sampler.rng) && throw(ArgumentError("MonteCarlo has no RNG; provide one with `ProtoBase.withrng`"))
    samples = [rand(sampler.rng, q) for _ in 1:(sampler.n)]
    return PointSet(samples, fill(1 / sampler.n, sampler.n))
end

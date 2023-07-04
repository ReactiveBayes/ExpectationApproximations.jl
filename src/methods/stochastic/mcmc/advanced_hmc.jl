using AdvancedHMC, ForwardDiff, LogDensityProblems, LinearAlgebra, Random
import ExponentialFamily: KnownExponentialFamilyDistribution, sufficientstatistics,distributiontype,reconstructargument!,logpartition, getnaturalparameters


struct AHMC <: AbstractApproximationMethod
    dim 
    metric
    n_samples
    n_adapts 
end

struct LogTargetDensity
    dim::Integer
    q
    μ
end

function Random.rand(ef::KnownExponentialFamilyDistribution{T}) where {T} 
    return rand(convert(Distribution,ef)) 
end
function logpartition(ef::KnownExponentialFamilyDistribution{T}, ηvec::Vector{F}) where {T, F <: Real}
    ηef = getnaturalparameters(ef)
    reconstructargument!(ηef, ηef, ηvec)
    return logpartition(KnownExponentialFamilyDistribution(T, ηef))
end

LogDensityProblems.logdensity(p::LogTargetDensity, θ) = logpdf(p.q,θ) + log(p.μ(θ))
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim

function approximate_expectation(method::AHMC, q::Union{<:KnownExponentialFamilyDistribution,<:Distribution}, f::F) where {F}
    _,_,result = prod(method,q,f)
    return sum(result)/length(result)
end

# function approximate_distribution(method::AHMC, q::Union{<:KnownExponentialFamilyDistribution,<:Distribution}, f::F) where {F}
#     samples,weights,_ = prod(method,q,f)

#     expectation_suffstats = mapreduce(d ->sufficientstatistics(q,d[1])*d[2],+, Iterators.zip(samples,weights))
#     l(θ) = logpartition(q,θ) - logpartition(q)
    
# end


function Base.prod(method::AHMC, q::Union{<:KnownExponentialFamilyDistribution,<:Distribution}, f::F) where {F}
    ltd = LogTargetDensity(method.dim,q,f)
    initial_θ = rand(ltd.q)
    hamiltonian = Hamiltonian(method.metric, ltd, ForwardDiff)
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(method.metric), StepSizeAdaptor(0.8, integrator))
    samples, _ = AdvancedHMC.sample(hamiltonian, proposal, initial_θ, method.n_samples, adaptor, method.n_adapts; progress=false)
    weights = map(d -> exp(LogDensityProblems.logdensity(ltd,d)), samples) 

    return samples, weights ./ sum(weights), weights
end
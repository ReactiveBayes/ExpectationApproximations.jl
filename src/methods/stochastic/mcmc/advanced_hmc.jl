using AdvancedHMC, ForwardDiff, LogDensityProblems, LinearAlgebra, Random
import ExponentialFamily: KnownExponentialFamilyDistribution, sufficientstatistics,distributiontype,reconstructargument!,logpartition, getnaturalparameters
import AdvancedHMC: sample, AbstractMetric, AbstractTrajectorySampler, AbstractTerminationCriterion, AbstractIntegrator, PhasePoint
import AdvancedHMC: AbstractAdaptor, MassMatrixAdaptor, StepSizeAdaptor, StanHMCAdaptor
import LogDensityProblems: logdensity,capabilities,dimension

const N_SAMPLES = 1000
const N_ADAPTS  = 1000
const ϵLEAP = 0.01
const STEP_SIZE_ADAPTOR = 0.8

struct AHMC{T, M, G, I, TS, TC} <: AbstractApproximationMethod
    dim :: T
    metric :: M
    n_samples :: T
    n_adapts :: T
    autograd :: G
    automatic_ϵ :: Bool
    integrator :: I 
    trajectory_sampler :: TS
    termination_criterion :: TC 
    function AHMC(dim::T, metric::M, n_samples::T = N_SAMPLES, n_adapts::T = N_ADAPTS, autograd::G = ForwardDiff, automatic_ϵ = true,
        integrator::I = Leapfrog(ϵLEAP),trajectory_sampler :: TS = MultinomialTS, termination_criterion::TC = GeneralisedNoUTurn()) where 
        {T <: Integer, M <: AbstractMetric, G, I <: AbstractIntegrator, TS, TC <: AbstractTerminationCriterion}
        
        return new{T, M, G,I, TS, TC}(dim,metric,n_samples,n_adapts,autograd,automatic_ϵ, integrator, trajectory_sampler, termination_criterion)
    end
end

struct LogTargetDensity{I, T, F}
    dim :: I
    q   :: T 
    μ   :: F 

    function LogTargetDensity(dim::I, q::T, μ::F) where {I <: Integer, T <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}, F}
        return new{I,T,F}(dim,q,μ)
    end
end

Random.rand(ef::KnownExponentialFamilyDistribution{T}) where {T <: Distribution} = Random.rand(convert(T,ef))

function logpartition(ef::KnownExponentialFamilyDistribution{T}, ηvec::Vector{F}) where {T, F <: Real}
    ηef = getnaturalparameters(ef)
    reconstructargument!(ηef, ηef, ηvec)
    return logpartition(KnownExponentialFamilyDistribution(T, ηef))
end

logdensity(p::LogTargetDensity, θ) = logpdf(p.q,θ) + log(p.μ(θ))
capabilities(::LogTargetDensity) = LogDensityProblems.LogDensityOrder{1}()
dimension(p::LogTargetDensity) = p.dim

function approximate_expectation(method::AHMC, q::Union{<:KnownExponentialFamilyDistribution,<:Distribution}, f::F) where {F}
    sample_list = prod(method,q,f)
    return mean(get_unnormalised_weights(get_meta(sample_list)))
end

function Base.prod(method::AHMC, q::Union{<:KnownExponentialFamilyDistribution,<:Distribution}, f::F) where {F}
    ltd = LogTargetDensity(method.dim, q, f)
    initial_θ = Random.rand(ltd.q)
    hamiltonian = Hamiltonian(method.metric, ltd, method.autograd)

    if method.automatic_ϵ == true
        initial_ϵ  = find_good_stepsize(hamiltonian, initial_θ)
        integrator = Leapfrog(initial_ϵ)
    else
        integrator = method.integrator
    end

    proposal = HMCKernel(Trajectory{method.trajectory_sampler}(integrator, method.termination_criterion))
    
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(method.metric), StepSizeAdaptor(STEP_SIZE_ADAPTOR, integrator))
    
    samples, _ = sample(hamiltonian, proposal, initial_θ, method.n_samples, adaptor, method.n_adapts; progress=false)
    
    raw_weights = map(d -> exp(logdensity(ltd,d)), samples) 
    weights_sum = sum(raw_weights)
    norm_weights = raw_weights ./ weights_sum

    H_x = mapreduce(d -> d[2] * logdensity(ltd,d[1]), +, Iterators.zip(samples,norm_weights))

    H_y = log(weights_sum) - log(method.n_samples)
    H_x = -H_x

    entropy = H_x + H_y

    meta = SampleListMeta(raw_weights, entropy, nothing, nothing)
  
    return SampleList(samples, norm_weights, meta)
end

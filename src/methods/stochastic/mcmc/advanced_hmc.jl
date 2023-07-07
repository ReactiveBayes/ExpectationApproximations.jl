using AdvancedHMC, ForwardDiff, LogDensityProblems, LinearAlgebra, Random
import ExponentialFamily: KnownExponentialFamilyDistribution, sufficientstatistics,distributiontype,reconstructargument!,logpartition, getnaturalparameters,rand, variate_form
import AdvancedHMC: sample, AbstractMetric, AbstractTrajectorySampler, AbstractTerminationCriterion, AbstractIntegrator, PhasePoint
import AdvancedHMC: AbstractAdaptor, MassMatrixAdaptor, StepSizeAdaptor, StanHMCAdaptor,DiagEuclideanMetric
import LogDensityProblems: logdensity,capabilities,dimension
import Distributions: VariateForm, Univariate, Multivariate, Matrixvariate

const N_SAMPLES = 1000
const N_ADAPTS  = 1000
const ϵLEAP = 0.01
const STEP_SIZE_ADAPTOR = 0.8

struct AHMC{T, M, G, I, TS, TC} <: AbstractApproximationMethod
    dim :: T
    n_samples :: T
    n_adapts :: T
    metric :: M
    autograd :: G
    automatic_ϵ :: Bool
    integrator :: I 
    trajectory_sampler :: TS
    termination_criterion :: TC 
    
    function AHMC(dim::T, n_samples::T = N_SAMPLES, n_adapts::T = N_ADAPTS, metric::M = DiagEuclideanMetric(dim), autograd::G = ForwardDiff, automatic_ϵ = true,
        integrator::I = Leapfrog(ϵLEAP),trajectory_sampler :: TS = MultinomialTS, termination_criterion::TC = GeneralisedNoUTurn()) where 
        {T <: Integer, M <: AbstractMetric, G, I <: AbstractIntegrator, TS, TC <: AbstractTerminationCriterion}
        return new{T, M, G, I, TS, TC}(dim,n_samples,n_adapts,metric,autograd,automatic_ϵ, integrator, trajectory_sampler, termination_criterion)
    end
end

struct LogTargetDensity{V, I, T, F}
    dim :: I
    q   :: T
    μ   :: F 

    function LogTargetDensity(dim::I, q::T, μ::F) where {I <: Integer, T <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}, F}
            V = variate_form(q)
        return new{V,I,T,F}(dim,q,μ)
    end

    function LogTargetDensity(dim::I, q::T, μ::F) where {I <: Integer, T , F <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}}
            V = variate_form(μ)
        return new{V,I,T,F}(dim,q,μ)
    end

    function LogTargetDensity(dim::I, q::T, μ::F) where 
        {I <: Integer, T <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}, F <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}}
            @assert variate_form(q) == variate_form(μ) " $(q) and $(μ) must have the same variate type"
            V = variate_form(q)
        return new{V,I,T,F}(dim,q,μ)
    end

    function LogTargetDensity(dim::I, q::T, μ::F) where {I <: Integer, T, F}
        return new{V,I,T,F}(dim,q,μ) where {V <: VariateForm}
    end
end

logdensity(p::LogTargetDensity{Univariate,I,T,F}, θ) where  {I <: Integer, T, F} = log(p.q(θ[1])) + log(p.μ(θ[1]))
logdensity(p::LogTargetDensity{Multivariate,I,T,F}, θ) where {I <: Integer, T , F} = log(p.q(θ)) + log(p.μ(θ))
function logdensity(p::LogTargetDensity{Matrixvariate,I,T,F}, θ) where {I <: Integer, T, F} 
        n = Int(sqrt(length(θ)))
        θnew = reshape(θ,n,n)
    return log(p.q(θnew)) + log(p.μ(θnew))
end

logdensity(p::LogTargetDensity{Univariate,I,T,F}, θ) where 
    {I <: Integer, T <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}, F} = logpdf(p.q,θ[1]) + log(p.μ(θ[1]))
logdensity(p::LogTargetDensity{Multivariate,I,T,F}, θ) where 
    {I <: Integer, T <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}, F} = logpdf(p.q,θ) + log(p.μ(θ))
function logdensity(p::LogTargetDensity{Matrixvariate,I,T,F}, θ) where 
    {I <: Integer, T <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}, F}       
        n = Int(sqrt(length(θ)))
        θnew = reshape(θ,n,n)
    return logpdf(p.q,θnew) + log(p.μ(θnew))
end

logdensity(p::LogTargetDensity{Univariate,I,T,F}, θ) where 
    {I <: Integer, T , F <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}} = log(p.q(θ[1])) + logpdf(p.μ, θ[1])
logdensity(p::LogTargetDensity{Multivariate,I,T,F}, θ) where 
    {I <: Integer, T , F <: Union{<:KnownExponentialFamilyDistribution,<:Distribution} } = log(p.q(θ)) + logpdf(p.μ, θ)
function logdensity(p::LogTargetDensity{Matrixvariate,I,T,F}, θ) where 
    {I <: Integer, T , F <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}}        
        n = Int(sqrt(length(θ)))
        θnew = reshape(θ,n,n)
    return log(p.q(θnew)) + logpdf(p.μ, θnew)
end

logdensity(p::LogTargetDensity{Univariate,I,T,F}, θ) where 
    {I <: Integer, T <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}, F <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}} =
        logpdf(p.q,θ[1]) + logpdf(p.μ, θ[1])
logdensity(p::LogTargetDensity{Multivariate,I,T,F}, θ) where 
    {I <: Integer, T <: Union{<:KnownExponentialFamilyDistribution,<:Distribution} , F <: Union{<:KnownExponentialFamilyDistribution,<:Distribution} } = 
        logpdf(p.q, θ) + logpdf(p.μ, θ)
function logdensity(p::LogTargetDensity{Matrixvariate,I,T,F}, θ) where 
    {I <: Integer, T <: Union{<:KnownExponentialFamilyDistribution,<:Distribution} , F <: Union{<:KnownExponentialFamilyDistribution,<:Distribution}}      
        n = Int(sqrt(length(θ)))
        θnew = reshape(θ,n,n)
    return logpdf(p.q, θnew) + logpdf(p.μ, θnew)
end

capabilities(::LogTargetDensity) = LogDensityProblems.LogDensityOrder{1}()
dimension(p::LogTargetDensity) = p.dim

function approximate_expectation(method::AHMC, q::T, f::F) where {T, F}
    sample_list = prod(method,q, x -> 1)
    samples = get_samples(sample_list)
    return mapreduce(d -> f(d), +, samples) / length(samples)
end

function approximate_distribution(method::AHMC, q::T, f::F) where {T, F}
   return prod(method,q,f)
end

function Base.prod(method::AHMC, q::Union{<:KnownExponentialFamilyDistribution,<:Distribution}, f::F) where {F}
    ltd = LogTargetDensity(method.dim, q, f) 
    hamiltonian = Hamiltonian(method.metric, ltd, method.autograd)
    var_form = variate_form(q)
    if var_form == Univariate || var_form == Matrixvariate
        initial_θ = as_vec(Random.rand(ltd.q))
    else
        initial_θ = Random.rand(ltd.q)
    end
    if method.automatic_ϵ == true 
        initial_ϵ  = find_good_stepsize(hamiltonian, initial_θ)
        integrator = Leapfrog(initial_ϵ)
    else
        integrator = method.integrator
    end

    proposal = HMCKernel(Trajectory{method.trajectory_sampler}(integrator, method.termination_criterion))
    
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(method.metric), StepSizeAdaptor(STEP_SIZE_ADAPTOR, integrator))
    
    samples, _ = sample(hamiltonian, proposal, initial_θ, method.n_samples, adaptor, method.n_adapts; verbose=false,progress=false)
    
    raw_weights = map(d -> exp(logdensity(ltd,d)), samples) 
    weights_sum = sum(raw_weights)
    norm_weights = raw_weights ./ weights_sum

    H_x = mapreduce(d -> d[2] * logdensity(ltd,d[1]), +, Iterators.zip(samples,norm_weights))

    H_y = log(weights_sum) - log(method.n_samples)
    H_x = -H_x

    entropy = H_x + H_y

    meta = SampleListMeta(raw_weights, entropy, nothing, nothing)
    
    if var_form == Univariate
        samples = map(d -> d[1], samples)
    elseif var_form == Matrixvariate
        n = Int(sqrt(method.dim))
        samples = map(d -> reshape(d,n,n),samples)
    end 

    return SampleList(samples, norm_weights, meta)
end

Base.prod(method::AHMC, q::F, f::Union{<:KnownExponentialFamilyDistribution,<:Distribution}) where {F} = prod(method,f,q)

## In case two known distributions are multiplied, it must be chosen which distribution a sample should be drawn to determine step size.
## For example if Gamma and Normal is multiplied it is better to choose Gamma to sample from ow negative valued samples might break step size determination.
## By default SampleLeft is used. If a usere wants to change the behavior than sample_direction function should be defined.
struct SampleRight end #Sample from left distribution in the prod
struct SampleLeft end #Sample from right distribution

function sample_direction(q,f) 
    return SampleLeft()
end

function Base.prod(method::AHMC,q::Union{<:KnownExponentialFamilyDistribution,<:Distribution},f::Union{<:KnownExponentialFamilyDistribution,<:Distribution} )
    return prod(method,sample_direction(q,f),q,f)
end
function Base.prod(method::AHMC,::SampleRight, q::Union{<:KnownExponentialFamilyDistribution,<:Distribution},f::Union{<:KnownExponentialFamilyDistribution,<:Distribution}) 
    ltd = LogTargetDensity(method.dim, q, f) 
    hamiltonian = Hamiltonian(method.metric, ltd, method.autograd) 
    var_form = variate_form(q)
    if var_form == Univariate || var_form == Matrixvariate
        initial_θ = as_vec(Random.rand(f))
    else
        initial_θ = Random.rand(f)
    end
    if method.automatic_ϵ == true 
        initial_ϵ  = find_good_stepsize(hamiltonian, initial_θ)
        integrator = Leapfrog(initial_ϵ)
    else
        integrator = method.integrator
    end

    proposal = HMCKernel(Trajectory{method.trajectory_sampler}(integrator, method.termination_criterion))
    
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(method.metric), StepSizeAdaptor(STEP_SIZE_ADAPTOR, integrator))
    
    samples, _ = sample(hamiltonian, proposal, initial_θ, method.n_samples, adaptor, method.n_adapts; verbose=false,progress=false)
    
    raw_weights = map(d -> exp(logdensity(ltd,d)), samples) 
    weights_sum = sum(raw_weights)
    norm_weights = raw_weights ./ weights_sum

    H_x = mapreduce(d -> d[2] * logdensity(ltd,d[1]), +, Iterators.zip(samples,norm_weights))

    H_y = log(weights_sum) - log(method.n_samples)
    H_x = -H_x

    entropy = H_x + H_y

    meta = SampleListMeta(raw_weights, entropy, nothing, nothing)
    
    if var_form == Univariate
        samples = map(d -> d[1], samples)
    elseif var_form == Matrixvariate
        n = Int(sqrt(method.dim))
        samples = map(d -> reshape(d,n,n),samples)
    end 

    return SampleList(samples, norm_weights, meta)
end

function Base.prod(method::AHMC,::SampleLeft, q::Union{<:KnownExponentialFamilyDistribution,<:Distribution},f::Union{<:KnownExponentialFamilyDistribution,<:Distribution}) 
    ltd = LogTargetDensity(method.dim, q, f) 
    hamiltonian = Hamiltonian(method.metric, ltd, method.autograd) 
    var_form = variate_form(q)
    if var_form == Univariate || var_form == Matrixvariate
        initial_θ = as_vec(Random.rand(q))
    else
        initial_θ = Random.rand(q)
    end
    if method.automatic_ϵ == true 
        initial_ϵ  = find_good_stepsize(hamiltonian, initial_θ)
        integrator = Leapfrog(initial_ϵ)
    else
        integrator = method.integrator
    end

    proposal = HMCKernel(Trajectory{method.trajectory_sampler}(integrator, method.termination_criterion))
    
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(method.metric), StepSizeAdaptor(STEP_SIZE_ADAPTOR, integrator))
    
    samples, _ = sample(hamiltonian, proposal, initial_θ, method.n_samples, adaptor, method.n_adapts; verbose=false,progress=false)
    
    raw_weights = map(d -> exp(logdensity(ltd,d)), samples) 
    weights_sum = sum(raw_weights)
    norm_weights = raw_weights ./ weights_sum

    H_x = mapreduce(d -> d[2] * logdensity(ltd,d[1]), +, Iterators.zip(samples,norm_weights))

    H_y = log(weights_sum) - log(method.n_samples)
    H_x = -H_x

    entropy = H_x + H_y

    meta = SampleListMeta(raw_weights, entropy, nothing, nothing)
    
    if var_form == Univariate
        samples = map(d -> d[1], samples)
    elseif var_form == Matrixvariate
        n = Int(sqrt(method.dim))
        samples = map(d -> reshape(d,n,n),samples)
    end 

    return SampleList(samples, norm_weights, meta)
end

import Base.prod
export ForwardDiffGrad
import ExponentialFamily: isproper, logpdf_sample_friendly, rand, convert
using LinearAlgebra
using Random

struct CVMP{R, O, G, B} <: AbstractApproximationMethod
    rng::R
    n_samples::Int ## to be used in ReactiveMP messsage computations. ExpectationApproximations do not use this field
    n_iterations::Int
    opt::O
    grad::G
    n_gradpoints::Int
    enforce_proper_messages::Val{B}
    warn::Bool

    function CVMP(
        rng::R, n_samples::Int, n_iterations::Int, opt::O, grad::G = ForwardDiffGrad(), n_gradpoints::Int = 1, enforce_proper_messages::Val{B} = Val(true), warn::Bool = true
    ) where {R, O, G, B}
        return new{R, O, G, B}(rng, n_samples, n_iterations, opt, grad, n_gradpoints, enforce_proper_messages, warn)
    end
end

function CVMP(n_samples::Int, n_iterations::Int, opt, grad = ForwardDiffGrad(), n_gradpoints::Int = 1, enforce_proper_messages::Val = Val(true), warn::Bool = true)
    return CVMP(Random.GLOBAL_RNG, n_samples, n_iterations, opt, grad, n_gradpoints, enforce_proper_messages, warn)
end

struct CVMPFactor
    exponentialfamily
    positivemeasure
    in_marginal
end

get_positive_measure(factor::CVMPFactor) = factor.positivemeasure
get_exponential_family(factor::CVMPFactor) = factor.exponentialfamily

function approximate_expectation(method::CVMP, q::CVIFactor, f::F) where {F}
    positivemeasure = get_positive_measure(q)
    exponentialfamily = get_exponential_family(q)
    in_marginal = get_marginal(q)
    product = prod(method, positivemeasure, exponentialfamily, in_marginal, f)
    return mean(product)
end

function cvmp_init(_::CVMP, _, exponentialfamily, _, _)
    return exponentialfamily
end

struct ForwardDiffGrad end

compute_derivative(::ForwardDiffGrad, f::F, value::Real) where {F}       = ForwardDiff.derivative(f, value)
compute_gradient(::ForwardDiffGrad, f::F, vec::AbstractVector) where {F} = ForwardDiff.gradient(f, vec)

function total_derivative(approximation, f, s::Real)
    return s * compute_derivative(approximation.grad, f, s)
end

function total_derivative(approximation, f, s::Tuple)
    f_for_grad = (x) -> f(x...)
    gradient_at_s = ReactiveMP.compute_gradient(approximation.grad, f_for_grad, collect(s))
    return dot(collect(s), gradient_at_s)
end

function Base.prod(approximation::CVMP, inbound, exponentialfamily::ExponentialFamily.KnownExponentialFamilyDistribution, in_marginal, nonlinearity)
    rng = something(approximation.rng, Random.default_rng())

    # Initial parameters of projected distribution
    exponentialfamily_current = ccmp_init(approximation, inbound, exponentialfamily, in_marginal, nonlinearity)

    # Some distributions implement "sampling" efficient versions
    # returns the same distribution by default
    _, in_marginal_friendly = logpdf_sample_friendly(in_marginal)

    hasupdated = false

    samples = cvilinearize(rand(rng, in_marginal_friendly, n_gradpoints))
    # compute gradient of log-likelihood
    # the multiplication between two logpdfs is correct
    # we take the derivative with respect to `η`
    # `logpdf(outbound, sample)` does not depend on `η` and is just a simple scalar constant
    weights = map((sample) -> total_derivative(approximation, nonlinearity, sample) * pdf(inbound, sample), samples)
    logq = let samples = samples, weights = weights
        (η) -> mean(
            map(
                (sample, weight) ->
                    weight * logpdf(ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, η, nothing), nonlinearity(sample...)),
                samples,
                weights
            )
        )
    end

    η = ExponentialFamily.getnaturalparameters(exponentialfamily_current)
    η_outbound_params = ExponentialFamily.getnaturalparameters(exponentialfamily_outbound)

    for _ in 1:n_iterations
        ∇logq = ReactiveMP.compute_gradient(approximation.grad, logq, ExponentialFamily.getnaturalparameters(exponentialfamily_current))

        # compute Fisher matrix 
        Fisher = ExponentialFamily.fisherinformation(exponentialfamily_current)

        ∇f = Fisher \ ∇logq

        # compute gradient on natural parameters
        ∇ = ExponentialFamily.getnaturalparameters(exponentialfamily_current) - ExponentialFamily.getnaturalparameters(exponentialfamily_outbound) - ∇f
        
        # perform gradient descent step
        λ_new = ReactiveMP.cvi_update!(opt, ExponentialFamily.getnaturalparameters(exponentialfamily_current), ∇)
        p_new = ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, λ_new)
        message_new = ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, λ_new - η_outbound_params)
        
        # check whether updated natural parameters are proper
        if ExponentialFamily.isproper(p_new) && ExponentialFamily.isproper(message_new)
            exponentialfamily_current = ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, λ_new, nothing)
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVMP approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return exponentialfamily_current
end
import Base.prod
export CVMP, CVMPFactor, ForwardDiffGrad, SimpleInitStrategy
import ExponentialFamily: isproper, logpdf_sample_friendly, rand, convert
using LinearAlgebra
using Random

struct ForwardDiffGrad end

compute_derivative(::ForwardDiffGrad, f::F, value::Real) where {F}       = ForwardDiff.derivative(f, value)
compute_gradient(::ForwardDiffGrad, f::F, vec::AbstractVector) where {F} = ForwardDiff.gradient(f, vec)
compute_gradient(grad::ForwardDiffGrad, f::F, value::Real) where {F}     = compute_derivative(grad, f, value)

function total_derivative(grad::ForwardDiffGrad, f, s::Real)
    return s * compute_derivative(grad, f, s)
end

function total_derivative(approximation, f, s::Tuple)
    f_for_grad = (x) -> f(x...)
    gradient_at_s = ReactiveMP.compute_gradient(approximation.grad, f_for_grad, collect(s))
    return dot(collect(s), gradient_at_s)
end



struct CVMP{R, O, G, B, S} <: AbstractApproximationMethod
    rng::R
    n_iterations::Int
    opt::O
    grad::G
    n_gradpoints::Int
    enforce_proper_messages::Val{B}
    warn::Bool
    initstrategy::S
end

struct SimpleInitStrategy end

function cvmp_init(_::CVMP{R, O, G, B, SimpleInitStrategy}, _, exponentialfamily, _, _) where {R, O, G, B}
    return exponentialfamily
end

struct CVMPFactor
    exponentialfamily
    positivemeasure
    in_marginal
end

get_positive_measure(factor::CVMPFactor) = factor.positivemeasure
get_exponential_family(factor::CVMPFactor) = factor.exponentialfamily
get_inmarginal(factor::CVMPFactor) = factor.in_marginal

function approximate_expectation(method::CVMP, q::CVMPFactor, f::F) where {F}
    positivemeasure = get_positive_measure(q)
    exponentialfamily = get_exponential_family(q)
    in_marginal = get_marginal(q)
    product = prod(method, positivemeasure, exponentialfamily, in_marginal, f)
    return mean(product)
end

function compute_grad_logq(approximation::CVMP, ef::ExponentialFamily.KnownExponentialFamilyDistribution{T}, samples, weights, nonlinearity) where {T}
    logq =
        (η) -> mean(
            map(
                (sample, weight) -> weight * logpdf(ExponentialFamily.KnownExponentialFamilyDistribution(T, η, ExponentialFamily.getconditioner(ef)), nonlinearity(sample...)),
                samples,
                weights
            )
        )
    return compute_gradient(approximation.grad, logq, ExponentialFamily.getnaturalparameters(ef))
end

function Base.prod(approximation::CVMP, positivemeasure, exponentialfamily::ExponentialFamily.KnownExponentialFamilyDistribution, in_marginal, nonlinearity)
    rng = something(approximation.rng, Random.default_rng())

    # Initial parameters of projected distribution
    exponentialfamily_current = cvmp_init(approximation, inbound, exponentialfamily, in_marginal, nonlinearity)

    # Some distributions implement "sampling" efficient versions
    # returns the same distribution by default
    _, in_marginal_friendly = logpdf_sample_friendly(in_marginal)

    hasupdated = false

    samples = cvilinearize(rand(rng, in_marginal_friendly, n_gradpoints))
    weights = map((sample) -> total_derivative(approximation, nonlinearity, sample) * pdf(positivemeasure, sample), samples)

    η_outbound_params = ExponentialFamily.getnaturalparameters(exponentialfamily_outbound)

    for _ in 1:n_iterations
        ∇logq = compute_grad_logq(approximation, exponentialfamily_current, samples, weights, nonlinearity)

        # compute Fisher matrix 
        Fisher = ExponentialFamily.fisherinformation(exponentialfamily_current)

        ∇f = Fisher \ ∇logq

        # compute gradient on natural parameters
        ∇ = ExponentialFamily.getnaturalparameters(exponentialfamily_current) - ExponentialFamily.getnaturalparameters(exponentialfamily_outbound) - ∇f

        # perform gradient descent step
        λ_new = cvi_update!(opt, ExponentialFamily.getnaturalparameters(exponentialfamily_current), ∇)
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

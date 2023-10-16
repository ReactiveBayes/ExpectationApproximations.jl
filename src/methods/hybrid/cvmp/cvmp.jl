export CVMP, CVMPFactor, SimpleInitStrategy
import ExponentialFamily: isproper, convert
import Base.rand
using LinearAlgebra
using Random
using Distributions
using Optimisers

function cvmp_setup(opt::Optimisers.AbstractRule, x)
    # We rely on the `deepcopy` here, because some optimizers from the `Optimizers.jl` may change their state
    return Optimisers.setup(deepcopy(opt), init_model(x))
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
    marginal
end

get_positive_measure(factor::CVMPFactor) = factor.positivemeasure
get_exponential_family(factor::CVMPFactor) = factor.exponentialfamily
get_marginal(factor::CVMPFactor) = factor.marginal

function approximate_distribution(method::CVMP, q::CVMPFactor, f::F) where {F}
    positivemeasure = get_positive_measure(q)
    exponentialfamily = get_exponential_family(q)
    marginal = get_marginal(q)
    return prod(method, positivemeasure, exponentialfamily, marginal, f)
end

function compute_grad_logq(approximation::CVMP, ef::ExponentialFamily.ExponentialFamilyDistribution{T}, samples, weights, nonlinearity) where {T}
    logq =
        (η) -> mean(
            map(
                (sample, weight) -> weight * logpdf(ExponentialFamily.ExponentialFamilyDistribution(T, η, ExponentialFamily.getconditioner(ef)), nonlinearity(sample...)),
                samples,
                weights
            )
        )
    return compute_gradient(approximation.grad, logq, ExponentialFamily.getnaturalparameters(ef))
end

function Base.prod(approximation::CVMP, positivemeasure, ef::ExponentialFamily.ExponentialFamilyDistribution{T}, marginal, nonlinearity) where {T}
    rng = something(approximation.rng, Random.default_rng())

    # Initial parameters of projected distribution
    exponentialfamily_current = cvmp_init(approximation, positivemeasure, ef, marginal, nonlinearity)

    # Some distributions implement "sampling" efficient versions
    # returns the same distribution by default
    marginal_friendly = convert(Distribution, marginal)

    hasupdated = false

    samples = hybridlinearize(Random.rand(rng, marginal_friendly, approximation.n_gradpoints))
    weights = map((sample) -> total_derivative(approximation.grad, nonlinearity, sample) * pdf(positivemeasure, sample), samples)

    η_outbound_params = ExponentialFamily.getnaturalparameters(ef)

    state_tree = cvmp_setup(approximation.opt, ExponentialFamily.getnaturalparameters(exponentialfamily_current))

    for _ in 1:(approximation.n_iterations)
        ∇logq = compute_grad_logq(approximation, exponentialfamily_current, samples, weights, nonlinearity)

        # compute Fisher matrix 
        Fisher = ExponentialFamily.fisherinformation(exponentialfamily_current)

        ∇f = Fisher \ ∇logq

        # compute gradient on natural parameters
        ∇ = ExponentialFamily.getnaturalparameters(exponentialfamily_current) - η_outbound_params - ∇f

        # perform gradient descent step
        state_tree, λ_new = Optimisers.update(state_tree, init_model(ExponentialFamily.getnaturalparameters(exponentialfamily_current)), init_model(∇))
        p_new = ExponentialFamily.ExponentialFamilyDistribution(T, λ_new.params[1], ExponentialFamily.getconditioner(ef), nothing)
        message_new = ExponentialFamily.ExponentialFamilyDistribution(T, λ_new.params[1] - η_outbound_params, ExponentialFamily.getconditioner(ef), nothing)

        if ExponentialFamily.isproper(p_new) && enforce_proper_message(approximation.enforce_proper_messages, p_new, message_new)
            exponentialfamily_current = ExponentialFamily.ExponentialFamilyDistribution(T, λ_new.params[1], ExponentialFamily.getconditioner(ef))
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVMP approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return exponentialfamily_current
end

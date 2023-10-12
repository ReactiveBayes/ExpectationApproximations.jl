export CVI, CVIFactor
using ExponentialFamily
using Random
using LinearAlgebra

"""
    CVI

The `CVI` structure defines the approximation method hyperparameters of the `prod(approximation::CVI, logp::F, dist)`.
This method performs an approximation of the product of the `dist` and `logp` with Stochastic Variational message passing (SVMP-CVI) (See [`Probabilistic programming with stochastic variational message passing`](https://biaslab.github.io/publication/probabilistic_programming_with_stochastic_variational_message_passing/)).

Arguments
 - `rng`: random number generator
 - `n_iterations`: number of iteration for the natural parameters gradient optimization
 - `opt`: optimizer, which will be used to perform the natural parameters gradient optimization step
 - `grad`: optional, defaults to `ForwardDiffGrad()`, structure to select how the gradient and the hessian will be computed
 - `n_gradpoints`: optional, defaults to 1, number of points to estimate gradient of the likelihood (dist*logp)
 - `enforce_proper_messages`: optional, defaults to true, ensures that a message, computed towards the exponentialfamily edges, is a proper distribution, must be of type `Val(true)/Val(false)`
 - `warn`: optional, defaults to true, enables or disables warnings related to the optimization steps

!!! note 
    `n_gradpoints` option is ignored in the Gaussian case
"""
struct CVI{R, O, G, B} <: AbstractApproximationMethod
    rng::R
    n_iterations::Int
    opt::O
    grad::G
    n_gradpoints::Int
    enforce_proper_messages::Val{B}
    warn::Bool
end

struct CVIFactor
    exponentialfamily
    positivemeasure
end

get_positive_measure(factor::CVIFactor) = factor.positivemeasure
get_exponential_family(factor::CVIFactor) = factor.exponentialfamily

function approximate_distribution(method::CVI, q::CVIFactor)
    positivemeasure = get_positive_measure(q)
    exponentialfamily = get_exponential_family(q)
    return prod(method, positivemeasure, exponentialfamily)
end

function cvi_init(_, _, exponentialfamily)
    return exponentialfamily
end

function compute_grad_logq(approximation::CVI, ef::ExponentialFamily.ExponentialFamilyDistribution{T}, positivemeasure, samples) where {T}
    logq = (η) -> mean((sample) -> logpdf(positivemeasure, sample) * logpdf(ExponentialFamily.ExponentialFamilyDistribution(T, η, ExponentialFamily.getconditioner(ef)), sample), samples)
    return compute_gradient(approximation.grad, logq, ExponentialFamily.getnaturalparameters(ef))
end

## If logpdf method is not defined this function can be utilized by passing the positive measure
## and working with the log of the positive measure 
function Base.prod(approximation::CVI, positivemeasure, ef::ExponentialFamily.ExponentialFamilyDistribution{T}) where T
    ##requires naturalparams function to be defined on the exponentialfamily
    rng = something(approximation.rng, Random.default_rng())

    # Initial parameters of projected distribution
    ef_current = cvi_init(approximation, positivemeasure, ef)

    ef_divisor = deepcopy(ef_current)

    hasupdated = false

    state_tree = cvmp_setup(approximation.opt, ExponentialFamily.getnaturalparameters(ef_current))

    for _ in 1:(approximation.n_iterations)

        # Some distributions implement "sampling" efficient versions
        # returns the same distribution by default
        sampling_frindly_current_ef = convert(Distribution, ef_current)

        samples = hybridlinearize(rand(rng, sampling_frindly_current_ef, approximation.n_gradpoints))

        ∇logq = compute_grad_logq(approximation, ef_current, positivemeasure, samples)

        # compute Fisher matrix 
        Fisher = ExponentialFamily.fisherinformation(ef_current)

        # compute natural gradient
        ∇f = Fisher \ ∇logq

        # compute gradient on natural parameters
        ∇ = ExponentialFamily.getnaturalparameters(ef_current) - ExponentialFamily.getnaturalparameters(ef_divisor) - ∇f

        # perform gradient descent step
        _, λ_new = Optimisers.update(state_tree, init_model(ExponentialFamily.getnaturalparameters(ef_current)), init_model(∇))

        p_new = ExponentialFamily.ExponentialFamilyDistribution(T, λ_new.params[1], ExponentialFamily.getconditioner(ef))
        if ExponentialFamily.isproper(p_new) && enforce_proper_message(approximation.enforce_proper_messages, p_new, ef_divisor)
            ef_current = ExponentialFamily.ExponentialFamilyDistribution(T, λ_new.params[1], ExponentialFamily.getconditioner(ef))
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVI approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return ef_current
end


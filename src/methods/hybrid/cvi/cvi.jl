import Base.prod
export CVI, ForwardDiffGrad, ZygoteGrad, prod
import ReactiveMP: isproper, logpdf_sample_friendly, rand, convert
import ReactiveMP: naturalparams, as_naturalparams, lognormalizer, NaturalParameters
using Random
using LinearAlgebra

"""
    cvi_update!(opt, λ, ∇)
"""
function cvi_update! end

function cvi_update!(callback::F, λ, ∇) where {F <: Function}
    return callback(λ, ∇)
end

cvilinearize(vector::AbstractVector) = vector
cvilinearize(matrix::AbstractMatrix) = eachcol(matrix)

"""
    CVI

The `CVI` structure defines the approximation method hyperparameters of the `prod(approximation::CVI, logp::F, dist)`.
This method performs an approximation of the product of the `dist` and `logp` with Stochastic Variational message passing (SVMP-CVI) (See [`Probabilistic programming with stochastic variational message passing`](https://biaslab.github.io/publication/probabilistic_programming_with_stochastic_variational_message_passing/)).

Arguments
 - `rng`: random number generator
 - `n_samples`: number of samples to use for statistics approximation
 - `n_iterations`: number of iteration for the natural parameters gradient optimization
 - `opt`: optimizer, which will be used to perform the natural parameters gradient optimization step
 - `grad`: optional, defaults to `ForwardDiffGrad()`, structure to select how the gradient and the hessian will be computed
 - `n_gradpoints`: optional, defaults to 1, number of points to estimate gradient of the likelihood (dist*logp)
 - `enforce_proper_messages`: optional, defaults to true, ensures that a message, computed towards the exponentialfamily edges, is a proper distribution, must be of type `Val(true)/Val(false)`
 - `warn`: optional, defaults to true, enables or disables warnings related to the optimization steps

!!! note 
    `n_gradpoints` option is ignored in the Gaussian case

!!! note 
    Run `using Flux` in your Julia session to enable the `Flux` optimizers support for the CVI approximation method.
    Run `using Zygote` in your Julia session to enable the `ZygoteGrad()` option support for the CVI `grad` parameter.
    Run `using DiffResults` in your Julia session to enable faster gradient computations in case if all inputs are of the `Gaussian` type.
"""
struct CVI{R, O, G, B} <: AbstractApproximationMethod
    rng::R
    n_samples::Int ## to be used in ReactiveMP messsage computations. ExpectationApproximations do not use this field
    n_iterations::Int
    opt::O
    grad::G
    n_gradpoints::Int
    enforce_proper_messages::Val{B}
    warn::Bool

    function CVI(
        rng::R, n_samples::Int, n_iterations::Int, opt::O, grad::G = ForwardDiffGrad(), n_gradpoints::Int = 1, enforce_proper_messages::Val{B} = Val(true), warn::Bool = true
    ) where {R, O, G, B}
        return new{R, O, G, B}(rng, n_samples, n_iterations, opt, grad, n_gradpoints, enforce_proper_messages, warn)
    end
end

function CVI(n_samples::Int, n_iterations::Int, opt, grad = ForwardDiffGrad(), n_gradpoints::Int = 1, enforce_proper_messages::Val = Val(true), warn::Bool = true)
    return CVI(Random.GLOBAL_RNG, n_samples, n_iterations, opt, grad, n_gradpoints, enforce_proper_messages, warn)
end

"""Alias for the `CVI` method. See help for [`CVI`](@ref)"""

struct CVIFactor
    exponentialfamily
    positivemeasure
end

get_positive_measure(factor::CVIFactor) = factor.positivemeasure
get_exponential_family(factor::CVIFactor) = factor.exponentialfamily

function approximate_expectation(method::CVI, q::CVIFactor, f::F) where {F}
    positivemeasure = get_positive_measure(q)
    exponentialfamily = get_exponential_family(q)

    exponentialfamily = prod(method, positivemeasure, exponentialfamily)
    importancesampling = ImportanceSampling(method.rng, n_samples, exponentialfamily)
    return approximate_expectation(importancesampling, newdist, f)
end

#---------------------------
# CVI implementations
#---------------------------

struct ForwardDiffGrad end

compute_derivative(::ForwardDiffGrad, f::F, value::Real) where {F}       = ForwardDiff.derivative(f, value)
compute_gradient(::ForwardDiffGrad, f::F, vec::AbstractVector) where {F} = ForwardDiff.gradient(f, vec)
compute_hessian(::ForwardDiffGrad, f::F, vec::AbstractVector) where {F}  = ForwardDiff.hessian(f, vec)

function compute_second_derivative(grad::G, logp::F, z_s::Real) where {G, F}
    first_derivative = (x) -> compute_derivative(grad, logp, x)
    return compute_derivative(grad, first_derivative, z_s)
end

# We perform the check in case if the `enforce_proper_messages` setting is set to `Val{true}`
enforce_proper_message(::Val{true}, λ::NaturalParameters, η::NaturalParameters) = isproper(λ - η)

# We skip the check in case if the `enforce_proper_messages` setting is set to `Val{false}`
enforce_proper_message(::Val{false}, λ::NaturalParameters, η::NaturalParameters) = true

function compute_fisher_matrix(approximation::CVI, ::Type{T}, vec::AbstractVector) where {T <: NaturalParameters}
    neg_lognormalizer = (x) -> -lognormalizer(as_naturalparams(T, x))

    return -compute_hessian(approximation.grad, neg_lognormalizer, vec)
end

## If logpdf method is not defined this function can be utilized by passing the positive measure
## and working with the log of the positive measure 
function prod(approximation::CVI, positivemeasure, exponentialfamily)
    ##requires naturalparams function to be defined on the exponentialfamily
    rng = something(approximation.rng, Random.default_rng())

    # Natural parameters of incoming distribution message
    η_exponentialfamily = naturalparams(exponentialfamily)

    # Natural parameter type of incoming distribution
    T = typeof(η_exponentialfamily)

    # Initial parameters of projected distribution
    λ_current = naturalparams(exponentialfamily)

    hasupdated = false

    for _ in 1:(approximation.n_iterations)

        # Some distributions implement "sampling" efficient versions
        # returns the same distribution by default
        _, q_friendly = logpdf_sample_friendly(convert(Distribution, λ_current))

        samples = cvilinearize(rand(rng, q_friendly, approximation.n_gradpoints))

        logq = let samples = samples, positivemeasure = positivemeasure, T = T
            (η) -> mean((sample) -> logpdf(positivemeasure, sample) * logpdf(as_naturalparams(T, η), sample), samples)
        end

        ∇logq = compute_gradient(approximation.grad, logq, vec(λ_current))

        # compute Fisher matrix 
        Fisher = compute_fisher_matrix(approximation, T, vec(λ_current))

        # compute natural gradient
        ∇f = Fisher \ ∇logq

        # compute gradient on natural parameters
        ∇ = λ_current - η_exponentialfamily - as_naturalparams(T, ∇f)

        # perform gradient descent step
        λ_new = as_naturalparams(T, cvi_update!(approximation.opt, λ_current, ∇))

        # check whether updated natural parameters are proper
        if isproper(λ_new) && enforce_proper_message(approximation.enforce_proper_messages, λ_new, η_exponentialfamily)
            λ_current = λ_new
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVI approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return convert(Distribution, λ_current)
end

using Requires

function __init__()
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        function cvi_update!(opt::Flux.Optimise.AbstractOptimiser, λ::NaturalParameters, ∇::NaturalParameters)
            return Flux.Optimise.update!(opt, vec(λ), vec(∇))
        end
    end

    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        export ZygoteGrad

        struct ZygoteGrad end

        compute_gradient(::ZygoteGrad, f::F, vec::AbstractVector) where {F} = Zygote.gradient(f, vec)[1]
        compute_hessian(::ZygoteGrad, f::F, vec::AbstractVector) where {F}  = Zygote.hessian(f, vec)
        compute_derivative(::ZygoteGrad, f::F, value::Real) where {F}       = Zygote.gradient(f, value)[1]
    end

    @require DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5" begin
        function compute_df_mv(::CVI{R, O, ForwardDiffGrad}, logp::F, vec::AbstractVector) where {R, O, F}
            result = DiffResults.HessianResult(vec)
            result = ForwardDiff.hessian!(result, logp, vec)
            return DiffResults.gradient(result), DiffResults.hessian(result) ./ 2
        end
    end
end

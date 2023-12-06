export CVI, CVIFactor
using ExponentialFamily
using Random
using LinearAlgebra

"""
    CVI

The `CVI` structure defines the approximation method hyperparameters of the `prod(approximation::CVI, logp::F, dist)`.
This method performs an approximation of the product of the `dist` and `logp` with Stochastic Variational message passing (SVMP-CVI) (See [`Probabilistic programming with stochastic variational message passing`](https://biaslab.github.io/publ  ication/probabilistic_programming_with_stochastic_variational_message_passing/)).

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

## If logpdf method is not defined this function can be utilized by passing the positive measure
## and working with the log of the positive measure 
function Base.prod(approximation::CVI, positivemeasure, ef::ExponentialFamily.ExponentialFamilyDistribution{T}) where {T}
    rng = something(approximation.rng, Random.default_rng())

    nat_params = ExponentialFamily.getnaturalparameters(ef)

    natural_manifold = ExponentialFamily.getmanifold(ExponentialFamily.ExponentialFamilyDistribution(T))

    function cvi_objective(natural_manifold, p, positive_measure; rng = StableRNG(42))
        nat_dist = point2dist(natural_manifold, p)
        samples = rand(rng, nat_dist, approximation.n_gradpoints)
        nat_logpdf = logpdf(nat_dist, samples)
        pos_logpdf = logpdf(positive_measure, sample)
        return dot(nat_logpdf, pos_logpdf) / n_gradpoints
    end

    function score(M, p)
        return cvi_objective(M, p, positivemeasure)
    end

    function grad_from_backend(backend, score)
        function grad(M, p)
            ManifoldDiff.gradient(M, score, p, backend)
        end
        return grad - nat_params
    end

    result = ExponentialFamilyManifolds.optimize_on_manifold(
        approximation.method,
        natural_manifold,
        score,
        grad_from_backend(approximation.backend),
        rand(rng, natural_manifold),
        approximation.stopping_criterion,
    )

    point_to_dist = point2dist(natural_manifold, result)
    
    return point_to_dist
end

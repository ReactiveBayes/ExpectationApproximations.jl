"""
    ProtoBackends

Backends for the expectation protocol of BayesBase, standing in for ExpectationApproximations
and ClosedFormExpectations (milestones U0.2 and U1 of PLAN.md). Every method is defined on a type this
package owns. Nothing is exported.
"""
module ProtoBackends

using Compat: @compat
using Distributions: Distribution, ContinuousUnivariateDistribution, Normal, MvNormal, Gamma, LogNormal, LogitNormal,
    Categorical, mean, var, std, cov, shape, scale, params, probs, ncategories, quantile
using ExponentialFamily: UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily
using FastGaussQuadrature: gausshermite, gausslegendre
using LinearAlgebra: cholesky, Symmetric
using BayesBase: BayesBase, PointMass, PointSet, FirstOf, Exact, Stochastic, Logpdf
using SpecialFunctions: digamma
using StatsFuns: logistic, log2π

@compat public ClosedForms, GaussHermite, QuantileLegendre, MonteCarlo, default_router

const Gaussian = Union{Normal, UnivariateNormalDistributionsFamily}
const MvGaussian = Union{MvNormal, MultivariateNormalDistributionsFamily}

include("closedforms.jl")
include("gausshermite.jl")
include("quantilelegendre.jl")
include("montecarlo.jl")

"""
    default_router(n)

Closed forms first, then Gauss–Hermite with `n` points for Gaussians (and transformed
Gaussians, point masses and categoricals), then Gauss–Legendre in probability space with `n`
points for any other continuous univariate distribution.
"""
default_router(n::Integer) = FirstOf(ClosedForms(), GaussHermite(n), QuantileLegendre(n))

function __init__()
    BayesBase.register_backend!(ClosedForms())
    BayesBase.register_backend!(GaussHermite(16); priority = 1)
    BayesBase.register_backend!(QuantileLegendre(16))
    return nothing
end

end

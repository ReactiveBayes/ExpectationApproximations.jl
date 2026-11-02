"""
    ExpectationApproximations

Numerical backends for the expectation protocol of BayesBase: quadrature, cubature, sigma points
and sampling compute `E_q[f]` through `BayesBase.expectation` and
`BayesBase.prepare_expectation`, alongside the local Gaussian projections that approximate
inference builds on (Laplace, moment matching, linearisation) and transforms to unconstrained
coordinates.

Nothing is exported: the names are public and used qualified, such as
`ExpectationApproximations.GaussHermite`.
"""
module ExpectationApproximations

using BayesBase: BayesBase, PointSet, ProductPointSet, PointMass, FirstOf, Exact, Stochastic, NoBackendError,
    expectation, prepare_expectation, which_backend
using Compat: @compat
using Distributions: Distribution, ContinuousUnivariateDistribution, Normal, MvNormal, AbstractMvNormal, LogNormal,
    LogitNormal, Gamma, Exponential, Beta, DiscreteNonParametric, Bernoulli, Binomial, DiscreteUniform, Dirac,
    mean, std, cov, params, scale, support, probs, pdf, quantile, failprob, succprob
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal, SymTridiagonal, Symmetric, cholesky, dot, eigen, eigvals, isposdef, logdet, norm
using SpecialFunctions: gamma, loggamma
using StaticArrays: StaticVector, SVector
using StatsFuns: log2π, logistic

@compat public gausshermite, gausslaguerre, gaussjacobi, gausslegendre
@compat public GaussHermite, GaussLaguerre, GaussJacobi, QuantileLegendre, ExactSum, MonteCarlo, EADefault
@compat public mean_cov, mean_cov_cross, tilted_mean_cov, laplace
@compat public Testing

include("rules/golubwelsch.jl")
include("backends/gausshermite.jl")
include("backends/orthogonal.jl")
include("backends/exactsum.jl")
include("backends/montecarlo.jl")
include("backends/default.jl")

include("projections/moments.jl")
include("projections/laplace.jl")

include("Testing.jl")

end

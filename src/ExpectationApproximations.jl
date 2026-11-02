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

using BayesBase: BayesBase, PointSet, PointMass, FirstOf, Exact, Stochastic
using Compat: @compat
using Distributions: Distribution, ContinuousUnivariateDistribution, Normal, AbstractMvNormal, LogNormal,
    LogitNormal, Gamma, Exponential, Beta, DiscreteNonParametric, Bernoulli, Binomial, DiscreteUniform, Dirac,
    mean, std, cov, params, scale, support, probs, pdf, quantile, failprob, succprob
using ForwardDiff: ForwardDiff
using LinearAlgebra: SymTridiagonal, Symmetric, cholesky, eigvals
using SpecialFunctions: gamma, loggamma
using StaticArrays: StaticVector, SVector
using StatsFuns: logistic

@compat public gausshermite, gausslaguerre, gaussjacobi, gausslegendre
@compat public GaussHermite, GaussLaguerre, GaussJacobi, QuantileLegendre, ExactSum, MonteCarlo, EADefault

include("rules/golubwelsch.jl")
include("backends/gausshermite.jl")
include("backends/orthogonal.jl")
include("backends/exactsum.jl")
include("backends/montecarlo.jl")
include("backends/default.jl")

end

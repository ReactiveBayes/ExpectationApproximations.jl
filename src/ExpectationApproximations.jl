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

using BayesBase: BayesBase
using Compat: @compat
using LinearAlgebra: SymTridiagonal, eigvals
using SpecialFunctions: gamma, loggamma

@compat public gausshermite, gausslaguerre, gaussjacobi, gausslegendre

include("rules/golubwelsch.jl")

end

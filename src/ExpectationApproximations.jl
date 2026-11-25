"""
    ExpectationApproximations

Numerical backends for the expectation protocol of BayesBase. Quadrature, cubature, sigma
points and sampling compute `E_q[f]` through `BayesBase.expectation` and
`BayesBase.prepare_expectation`. The package also implements Gaussian approximations of a density
(Laplace, moment matching and the variational Gaussian), linearisation, and transforms to
unconstrained coordinates for variables with a constrained domain.

Nothing is exported. The names are public and used qualified, such as
`ExpectationApproximations.GaussHermite`.
"""
module ExpectationApproximations

using BayesBase: BayesBase, ExpectationPointSet, ExpectationProductPointSet, FactorizedJoint, PointMass, ExpectationFirstOf, ExpectationExact, ExpectationStochastic, UnsupportedExpectationError,
    ProductOf, LinearizedProductOf, getleft, getright, expectation, prepare_expectation, which_expectation_backend,
    ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf
using Compat: @compat
using Distributions: Distributions, Distribution, UnivariateDistribution, VariateForm, Univariate, Multivariate,
    Continuous, Dirichlet, insupport, logpdf, ContinuousUnivariateDistribution, Normal, MvNormal, AbstractMvNormal, LogNormal,
    LogitNormal, Gamma, Exponential, Beta, Chisq, Erlang, InverseGamma, Weibull, Uniform, Laplace, Logistic, Gumbel, Poisson, Geometric,
    NegativeBinomial, VonMises, DiscreteNonParametric, Bernoulli, Binomial, DiscreteUniform, Dirac,
    mean, std, cov, params, scale, shape, support, probs, pdf, quantile, failprob, succprob, skewness, kurtosis, entropy
using DomainSets: DomainSets
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal, I, SymTridiagonal, Symmetric, cholesky, dot, eigen, eigvals, isposdef, issuccess, logdet, norm
using SpecialFunctions: gamma, loggamma
using StaticArrays: StaticVector, SVector
using Random: AbstractRNG
using StatsFuns: log1pexp, log2π, logistic, logit

@compat public gausshermite, gausslaguerre, gaussjacobi, gausslegendre
@compat public GaussHermite, GaussLaguerre, GaussJacobi, UnconstrainedLegendre, QuantileLegendre, ExactSum, MonteCarlo, EADefault, register_backends!
@compat public Unscented, SigmaPointSet, GenUnscented, SphericalRadialCubature, SNIS, effective_sample_size
@compat public mean_cov, mean_cov_cross, tilted_mean_cov, laplace, gaussian_variational, moment_matching, linearize
@compat public Domain, RealLine, Positive, UnitInterval, LowerBounded, UpperBounded, Bounded, Simplex, Componentwise
@compat public domain, forward, inverse, logjacobian, pushforward, TransformedGaussian
@compat public Testing, MPRACompat

include("rules/golubwelsch.jl")
include("backends/gausshermite.jl")
include("backends/orthogonal.jl")
include("backends/unconstrained.jl")
include("backends/exactsum.jl")
include("backends/montecarlo.jl")
include("backends/snis.jl")
include("backends/default.jl")
include("backends/registry.jl")

include("transforms/domains.jl")
include("transforms/transformed.jl")
include("projections/moments.jl")
include("projections/laplace.jl")
include("projections/variational.jl")
include("projections/matching.jl")
include("projections/linearize.jl")
include("backends/sigmapoints.jl")

include("Testing.jl")
include("compat/MPRACompat.jl")

end

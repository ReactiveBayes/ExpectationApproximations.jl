# The distributions of ExponentialFamily that are not Distributions' own types: its univariate
# Gaussians, through Gauss–Hermite and the sigma points, and `GammaShapeRate`, through the
# composite rule in `log x` and, chosen explicitly, Gauss–Laguerre. Its multivariate Gaussians are `AbstractMvNormal`s, which the core covers.
module ExpectationApproximationsExponentialFamilyExt

using ExpectationApproximations: ExpectationApproximations as EA, GaussHermite, GaussLaguerre, UnconstrainedLegendre, Unscented, GenUnscented,
    SphericalRadialCubature
using ExponentialFamily: NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision, GammaShapeRate
using BayesBase: BayesBase, ExpectationPointSet, mean, std, shape, rate

const UnivariateGaussian = Union{NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision}

BayesBase.supports_expectation(::GaussHermite, ::Type, ::Type{<:UnivariateGaussian}) = true
BayesBase.prepare_expectation(rule::GaussHermite, q::UnivariateGaussian) = EA.scalar_points(rule, mean(q), std(q))

BayesBase.supports_expectation(::Unscented, ::Type, ::Type{<:UnivariateGaussian}) = true
BayesBase.prepare_expectation(rule::Unscented, q::UnivariateGaussian) = EA.unscented_points(rule, mean(q), std(q))

BayesBase.supports_expectation(::GenUnscented, ::Type, ::Type{<:UnivariateGaussian}) = true
BayesBase.prepare_expectation(::GenUnscented, q::UnivariateGaussian) = EA.generalised_points(mean(q), std(q), 0.0, 3.0)

BayesBase.supports_expectation(::SphericalRadialCubature, ::Type, ::Type{<:UnivariateGaussian}) = true
BayesBase.prepare_expectation(::SphericalRadialCubature, q::UnivariateGaussian) = EA.cubature_points(mean(q), std(q))

BayesBase.supports_expectation(::UnconstrainedLegendre, ::Type, ::Type{<:GammaShapeRate}) = true
BayesBase.prepare_expectation(rule::UnconstrainedLegendre, q::GammaShapeRate) = EA.gamma_points(rule, EA.shapevalue(shape(q)), inv(rate(q)))

BayesBase.supports_expectation(::GaussLaguerre, ::Type, ::Type{<:GammaShapeRate}) = true

function BayesBase.prepare_expectation(rule::GaussLaguerre, q::GammaShapeRate)
    nodes, weights = EA.probability_laguerre(rule.n, shape(q) - 1)
    return ExpectationPointSet(nodes ./ rate(q), weights)
end

end

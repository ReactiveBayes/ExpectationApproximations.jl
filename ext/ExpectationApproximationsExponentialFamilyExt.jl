# The distributions of ExponentialFamily that are not Distributions' own types: its univariate
# Gaussians, through Gauss–Hermite, and `GammaShapeRate`, through Gauss–Laguerre. Its multivariate
# Gaussians are `AbstractMvNormal`s, which the core already covers.
module ExpectationApproximationsExponentialFamilyExt

using ExpectationApproximations: ExpectationApproximations as EA, GaussHermite, GaussLaguerre
using ExponentialFamily: NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision, GammaShapeRate
using BayesBase: BayesBase, PointSet, mean, std, shape, rate

const UnivariateGaussian = Union{NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision}

BayesBase.supports(::GaussHermite, ::Type, ::Type{<:UnivariateGaussian}) = true
BayesBase.prepare_expectation(rule::GaussHermite, q::UnivariateGaussian) = EA.scalar_points(rule, mean(q), std(q))

BayesBase.supports(::GaussLaguerre, ::Type, ::Type{<:GammaShapeRate}) = true

function BayesBase.prepare_expectation(rule::GaussLaguerre, q::GammaShapeRate)
    nodes, weights = EA.probability_laguerre(rule.n, shape(q) - 1)
    return PointSet(nodes ./ rate(q), weights)
end

end

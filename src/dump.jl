using ExpectationApproximations, BenchmarkTools
using Distributions, ExponentialFamily
import Distributions: skewness, kurtosis, cov, var
import ExpectationApproximations: GenUnscented, unscented_statistics, approximate_expectation,_cov
using LinearAlgebra
skewness(dist::MvNormalMeanCovariance)   = zeros(length(dist.μ))
kurtosis(dist::MvNormalMeanCovariance, correction::Bool=false)   = 3*var(dist).^2
skewness(dist::MvNormalMeanPrecision)   = zeros(length(dist.μ))
kurtosis(dist::MvNormalMeanPrecision, correction::Bool=false)   = 3*var(dist).^2
skewness(dist::MvNormalWeightedMeanPrecision)   = zeros(length(dist.xi))
kurtosis(dist::MvNormalWeightedMeanPrecision, correction::Bool=false)   = 3*var(dist).^2
skewness(dist::UnivariateGaussianDistributionsFamily) = 0.0
kurtosis(dist::UnivariateGaussianDistributionsFamily, correction::Bool=false) = 3*var(dist)^2
dist =Gamma(10rand(),10rand())
       

mdist = mean(dist)
vdist = _cov(dist)
sdist = skewness(dist)
kdist = kurtosis(dist, false)
brtol = 1e-2
method = GenUnscented{Multivariate,Univariate}(1,4)

g(x) = [x^2, x^3, x^20, x^8]
ExpectationApproximations.approximate_expectation(method,dist,g)
ExpectationApproximations.getcache(method.cache, (:weights, 3))


@benchmark ExpectationApproximations.approximate_expectation(method,dist,g)
@code_warntype ExpectationApproximations.approximate_expectation(method,dist,g)
samples = [rand(dist) for i=1:100000];
mapped_samples = map(d -> g(d),samples);
mean_estimate_mc = mean(mapped_samples);
var_estimate_mc = var(mapped_samples);
# skewness_estimate_mc = mean(map(d -> (d - mean_estimate_mc)^3, mapped_samples))
# kurtosis_estimate_mc = mean(map(d -> (d - mean_estimate_mc)^4, mapped_samples))
mean_estimate_mc

dist = MvNormalMeanCovariance(randn(20),10diagm(rand(20)))
       
mdist = mean(dist)
vdist = _cov(dist)
sdist = skewness(dist)
kdist = kurtosis(dist, false)
brtol = 1e-2
method = GenUnscented{Multivariate, Multivariate}(20,20)
g(x) = abs.(x)
m = ExpectationApproximations.approximate_expectation(method,dist,g)
method.cache
ExpectationApproximations.getcache(method.cache, (:weights, 41))
res = ExpectationApproximations.Unscented_approximate(method,dist,g)
samples = rand(dist, 1000000)
trans_samples = g.(samples)
msamples = mean(trans_samples,dims=2)
cov(trans_samples,dims=2)
mean((trans_samples .- msamples).^4,dims=2)
approximate_expectation(method, dist, ExpectationApproximations.adjust_integrand(DiagonalMomentGenerator(Val(3), mdist), g))


@benchmark ExpectationApproximations.Unscented_approximate(method,(dist, ),g)
@code_warntype ExpectationApproximations.Unscented_approximate(method,(dist, ),g)
@code_warntype ExpectationApproximations.adjust_integrand(MomentGenerator(Multivariate, Val(2), mdist), g)
@benchmark ExpectationApproximations.Unscented_approximate(method,(dist, ),g)
@code_warntype ExpectationApproximations.adjust_integrand(DiagonalMomentGenerator(Val(2), zeros(5)), g)

@code_warntype ExpectationApproximations.unscented_statistics(method,Val(true),(dist, ),g)
@code_warntype approximate_expectation(method, dist, g)

dist = (MvNormalMeanCovariance(randn(3),rand(InverseWishart(5,diagm(ones(3))))), MvNormalMeanCovariance(randn(5),diagm(rand(5))))
method = GenUnscented{Multivariate, Multivariate}(8,4)
g(x,y) = [x + y[1:3]; y[5] ]
@code_warntype ExpectationApproximations.approximate(method,g, dist)




# ExpectationApproximations.approximate_expectation(method, dist, ExpectationApproximations.adjust_integrand(ExpectationApproximations.MomentGenerator(Multivariate, Val(2), mdist), g))
x = zeros(3)
using TensorOperations
A = zeros(3,3,3)
@tensor begin
    A[i,j,k] = x[i]*x[j]*x[k]
end

ds = size.(mean.(dist))

dist = (MvNormalMeanCovariance(randn(3),rand(InverseWishart(5,diagm(ones(3))))), MvNormalMeanCovariance(randn(5),diagm(rand(5))))
method = GenUnscented{Multivariate, Multivariate}(8,4)
g(x,y) = [x + y[1:3]; y[5] ]
estimate_genut = ExpectationApproximations.Unscented_approximate(method,dist,g)
# samples = [rand(dist) for i=1:10000];
# mapped_samples = map(d -> g(d),samples);
# @benchmark map(d -> g(d),samples)
# mean_estimate_mc = mean(mapped_samples);
# var_estimate_mc = cov(mapped_samples);

@benchmark ExpectationApproximations.approximate_expectation(method,dist,g, Val(false))
@code_warntype ExpectationApproximations.Unscented_approximate(method,dist,g)






using ExpectationApproximations,LinearAlgebra,ExponentialFamily, AdvancedHMC,ForwardDiff,LogDensityProblems

D = 2
q = MvNormalMeanCovariance(zeros(D), [1.0 0.0; 0.0 1.0] )
ef = convert(ExponentialFamily.KnownExponentialFamilyDistribution, q)
f(x) = exp(-tr(x*x'))
metric = DiagEuclideanMetric(D)
n_samples, n_adapts = 2_00, 1_000
method = ExpectationApproximations.AHMC(D, metric,n_samples,n_adapts)

samples, weights,_ = prod(method,q,f)
expectation = ExpectationApproximations.approximate(method,q,f)
# ExpectationApproximations.approximate_distribution(method,ef,f)

f(zeros(D))
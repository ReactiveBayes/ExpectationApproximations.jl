# Benchmarks of ExpectationApproximations, run with `make benchmark`, or compared with a branch
# with `make benchmark-compare branch=main`.
using ExpectationApproximations
using BayesBase
using BenchmarkTools
using Distributions

const EA = ExpectationApproximations
const SUITE = BenchmarkGroup()

# Rules, computed once per backend or per preparation
SUITE["rules"]["gausshermite(32)"] = @benchmarkable EA.gausshermite(32)
SUITE["rules"]["gausslaguerre(32, 1.5)"] = @benchmarkable EA.gausslaguerre(32, 1.5)

# Preparing an expectation
const normal = Normal(0.5, 1.5)
const mvnormal = MvNormal([1.0, -1.0], [2.0 0.5; 0.5 1.0])
const mixture = BayesBase.FactorizedJoint((Categorical([0.4, 0.6]), MvNormal([-2.0, 2.0], [0.1 0.0; 0.0 0.1]), MvNormal([0.5, -0.5], [0.05 0.0; 0.0 0.05])))
SUITE["prepare"]["GaussHermite(16), Normal"] = @benchmarkable BayesBase.prepare_expectation(EA.GaussHermite(16), $normal)
SUITE["prepare"]["GaussHermite(5), MvNormal(2)"] = @benchmarkable BayesBase.prepare_expectation(EA.GaussHermite(5), $mvnormal)
SUITE["prepare"]["EADefault(5), mixture product"] = @benchmarkable BayesBase.prepare_expectation(EA.EADefault(5), $mixture)

# Evaluating a prepared expectation, the hot path of the universal rules: allocation-free
const scalar = BayesBase.prepare_expectation(EA.GaussHermite(16), normal)
const routed = BayesBase.prepare_expectation(EA.EADefault(16), normal)
const product = BayesBase.prepare_expectation(EA.EADefault(5), mixture)
loglik(y, (z, m, s)) = (s[z] - log(2π) - exp(s[z]) * (y - m[z])^2) / 2
SUITE["expectation"]["GaussHermite(16), sin"] = @benchmarkable BayesBase.expectation($scalar, sin)
SUITE["expectation"]["EADefault(16), sin"] = @benchmarkable BayesBase.expectation($routed, sin)
SUITE["expectation"]["mixture message"] = @benchmarkable BayesBase.expectation($product, inputs -> loglik(1.0, inputs))

# Projections
SUITE["projections"]["laplace, scalar"] = @benchmarkable EA.laplace(x -> -(x - 2)^2 / 18 - x^4 / 100, 0.0)
SUITE["projections"]["laplace, vector"] = @benchmarkable EA.laplace(x -> logpdf($mvnormal, x) - sum(abs2, x)^2 / 100, zeros(2))
SUITE["projections"]["mean_cov, Unscented"] = @benchmarkable EA.mean_cov(EA.Unscented(), x -> [x[1] * x[2], sin(x[1])], $mvnormal)
SUITE["projections"]["tilted_mean_cov, GaussHermite(48)"] = @benchmarkable EA.tilted_mean_cov(EA.GaussHermite(48), x -> -(x - 1)^2, $normal)

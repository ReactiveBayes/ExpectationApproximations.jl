@testitem "conformance: GaussHermite" tags = [:backends, :conformance] begin
    using Distributions
    const EA = ExpectationApproximations
    μ, σ = 0.5, 1.5
    EA.Testing.test_backend(
        EA.GaussHermite(16), [
            EA.Testing.Case(Normal(μ, σ), x -> x^4, μ^4 + 6μ^2 * σ^2 + 3σ^4),
            EA.Testing.Case(Normal(0.0, 1.0), cos, exp(-1 / 2)),
            EA.Testing.Case(LogNormal(0.3, 0.5), identity, exp(0.3 + 0.5^2 / 2)),
            EA.Testing.Case(MvNormal([1.0, -1.0], [2.0 0.5; 0.5 1.0]), x -> x[1] * x[2], 0.5 - 1.0),
        ]
    )
end

@testitem "conformance: GaussLaguerre and GaussJacobi" tags = [:backends, :conformance] begin
    using Distributions
    const EA = ExpectationApproximations
    EA.Testing.test_backend(
        EA.GaussLaguerre(8), [
            EA.Testing.Case(Gamma(0.3, 2.0), abs2, 0.3 * 1.3 * 4.0),
            EA.Testing.Case(Exponential(2.0), x -> x^3, 6 * 8.0),
        ]
    )
    EA.Testing.test_backend(
        EA.GaussJacobi(8), [
            EA.Testing.Case(Beta(2.0, 5.0), x -> x^3, (2 * 3 * 4) / (7 * 8 * 9)),
            EA.Testing.Case(Beta(0.5, 0.5), identity, 0.5),
        ]
    )
end

@testitem "conformance: QuantileLegendre" tags = [:backends, :conformance] begin
    using Distributions
    const EA = ExpectationApproximations
    EA.Testing.test_backend(
        EA.QuantileLegendre(256), [
            EA.Testing.Case(Weibull(2.0, 1.0), identity, mean(Weibull(2.0, 1.0)); rtol = 1.0e-4),
            EA.Testing.Case(Beta(2.0, 5.0), identity, 2 / 7; rtol = 1.0e-6),
        ]
    )
end

@testitem "conformance: ExactSum" tags = [:backends, :conformance] begin
    using Distributions
    const EA = ExpectationApproximations
    EA.Testing.test_backend(
        EA.ExactSum(), [
            EA.Testing.Case(Categorical([0.2, 0.3, 0.5]), x -> x^2, 0.2 + 1.2 + 4.5),
            EA.Testing.Case(Binomial(10, 0.3), identity, 3.0),
            EA.Testing.Case(Bernoulli(0.3), identity, 0.3),
            EA.Testing.Case(Dirac(1.5), x -> 2x, 3.0),
            EA.Testing.Case(EA.BayesBase.PointMass(2.0), abs2, 4.0),
        ]
    )
end

@testitem "conformance: MonteCarlo" tags = [:backends, :conformance] begin
    using Distributions
    const EA = ExpectationApproximations
    EA.Testing.test_backend(
        EA.MonteCarlo(20_000), [
            EA.Testing.Case(Gamma(2.0, 2.0), identity, 4.0; rtol = 0.05),
            EA.Testing.Case(Normal(1.0, 1.0), abs2, 2.0; rtol = 0.05),
        ]
    )
end

@testitem "conformance: EADefault" tags = [:backends, :conformance] begin
    using Distributions
    const EA = ExpectationApproximations
    EA.Testing.test_backend(
        EA.EADefault(), [
            EA.Testing.Case(Normal(0.0, 1.0), cos, exp(-1 / 2)),
            # E[log x] = digamma(2.5): log is singular at zero, so quadrature converges slowly, about
            # 1e-3 relative at 16 points; closed forms are what this expectation needs
            EA.Testing.Case(Gamma(2.5, 1.0), log, 0.7031566406452432; rtol = 2.0e-3),
            EA.Testing.Case(Beta(2.0, 3.0), identity, 0.4),
            EA.Testing.Case(Categorical([0.25, 0.75]), identity, 1.75),
            EA.Testing.Case(Weibull(2.0, 1.0), identity, mean(Weibull(2.0, 1.0)); rtol = 1.0e-3),
        ]
    )
end

@testitem "conformance: ExponentialFamily's types" tags = [:backends, :conformance, :extensions] begin
    using ExponentialFamily, StaticArrays
    const EA = ExpectationApproximations
    EA.Testing.test_backend(
        EA.EADefault(), [
            EA.Testing.Case(NormalMeanVariance(1.0, 4.0), abs2, 5.0),
            EA.Testing.Case(MvNormalMeanCovariance(SVector(1.0, -1.0), SMatrix{2, 2}(2.0, 0.5, 0.5, 1.0)), x -> x' * x, 3.0 + 2.0),
            EA.Testing.Case(GammaShapeRate(2.5, 4.0), abs2, 2.5 * 3.5 / 16),
        ]
    )
end

@testitem "SNIS: an unnormalised product of densities" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, StableRNGs
    # N(0, 1) × N(1, 1) is proportional to N(1/2, 1/2)
    target = P.ProductOf(Normal(0.0, 1.0), Normal(1.0, 1.0))
    backend = EA.SNIS(Normal(0.0, 2.0), 50_000, StableRNG(1))
    @test P.supports_expectation(backend, Any, typeof(target)) === true
    @test P.expectation(backend, identity, target) ≈ 0.5 rtol = 2.0e-2
    @test P.expectation(backend, abs2, target) ≈ 0.5 + 0.25 rtol = 2.0e-2
    @test P.expectation(EA.SNIS(Exponential(3.0), 50_000, StableRNG(2)), identity, Gamma(3.0, 1.0)) ≈ 3.0 rtol = 2.0e-2
end

@testitem "SNIS: needs an RNG, labelled stochastic, reproducible" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, StableRNGs
    @test_throws ArgumentError EA.SNIS(Normal(0.0, 1.0), 0)
    @test_throws ArgumentError P.expectation(EA.SNIS(Normal(0.0, 2.0), 100), identity, Normal(0.0, 1.0))
    rng = StableRNG(1)
    @test P.get_expectation_rng(P.with_expectation_rng(EA.SNIS(Normal(0.0, 2.0), 100), rng)) === rng
    @test P.is_expectation_stochastic(EA.SNIS(Normal(0.0, 2.0), 100), identity, Normal(0.0, 1.0)) === true
    estimate(seed) = P.expectation(EA.SNIS(Normal(0.0, 2.0), 1000, StableRNG(seed)), identity, Normal(0.3, 1.0))
    @test estimate(7) == estimate(7)
end

@testitem "SNIS: the effective sample size" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, StableRNGs
    # sampling the target itself gives equal weights, so every sample counts
    same = P.prepare_expectation(EA.SNIS(Normal(0.0, 1.0), 1000, StableRNG(1)), Normal(0.0, 1.0))
    @test EA.effective_sample_size(same) ≈ 1000
    # a poor proposal gives far fewer
    poor = P.prepare_expectation(EA.SNIS(Normal(5.0, 0.5), 1000, StableRNG(1)), Normal(0.0, 1.0))
    @test EA.effective_sample_size(poor) < 50
end

@testitem "conformance: SNIS" tags = [:backends, :conformance] begin
    using Distributions
    const EA = ExpectationApproximations
    EA.Testing.test_backend(
        EA.SNIS(Normal(0.0, 3.0), 100_000), [
            EA.Testing.Case(Normal(0.5, 1.0), identity, 0.5; rtol = 3.0e-2),
            EA.Testing.Case(Gamma(3.0, 1.0), identity, 3.0; rtol = 3.0e-2),
        ]
    )
end

@testitem "SNIS: support only for targets the proposal can weigh" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    univariate = EA.SNIS(Normal(0.0, 2.0), 10)
    @test P.supports_expectation(univariate, Any, Gamma{Float64}) === true
    @test P.supports_expectation(univariate, Any, Poisson{Float64}) === false
    @test P.supports_expectation(univariate, Any, typeof(MvNormal(zeros(2), [1.0 0.0; 0.0 1.0]))) === false
    multivariate = EA.SNIS(MvNormal(zeros(2), [1.0 0.0; 0.0 1.0]), 10)
    @test P.supports_expectation(multivariate, Any, typeof(MvNormal(zeros(2), [1.0 0.0; 0.0 1.0]))) === true
    @test P.supports_expectation(multivariate, Any, Normal{Float64}) === false
end

@testitem "SNIS: samples outside the target's support weigh nothing" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, StableRNGs
    # half the samples of the proposal are negative, where InverseGamma's logpdf throws
    @test P.expectation(EA.SNIS(Normal(0.0, 3.0), 100_000, StableRNG(3)), identity, InverseGamma(3.0, 2.0)) ≈ 1.0 rtol = 5.0e-2
end

@testitem "SNIS: a target that defines no insupport" tags = [:backends] setup = [EAHelpers] begin
    using BayesBase, Distributions, StableRNGs, LinearAlgebra
    # a user's own target, with a log-density and nothing else, as a posterior written by hand
    struct Shifted <: ContinuousMultivariateDistribution
        μ::Vector{Float64}
    end
    Base.length(d::Shifted) = length(d.μ)
    Distributions.logpdf(d::Shifted, x::AbstractVector) = logpdf(MvNormal(d.μ, Matrix(1.0I, 2, 2)), x)
    sampler = BayesBase.with_expectation_rng(EA.SNIS(MvNormal(zeros(2), Matrix(4.0I, 2, 2)), 20_000), StableRNG(1))
    @test BayesBase.expectation(sampler, identity, Shifted([0.7, -0.3])) ≈ [0.7, -0.3] atol = 0.05
end

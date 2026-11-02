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

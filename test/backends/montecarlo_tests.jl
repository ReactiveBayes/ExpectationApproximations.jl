@testitem "MonteCarlo: needs an RNG, which with_expectation_rng provides" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, StableRNGs
    @test_throws ArgumentError EA.MonteCarlo(0)
    @test P.get_expectation_rng(EA.MonteCarlo(10)) === nothing
    @test_throws ArgumentError P.expectation(EA.MonteCarlo(10), identity, Normal(0.0, 1.0))
    rng, other = StableRNG(1), StableRNG(2)
    @test P.get_expectation_rng(P.with_expectation_rng(EA.MonteCarlo(10), rng)) === rng
    @test P.get_expectation_rng(P.with_expectation_rng(EA.MonteCarlo(10, other), rng)) === other
end

@testitem "MonteCarlo: stochastic, reproducible with the same RNG" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, StableRNGs
    q = Gamma(2.0, 2.0)
    estimate(seed) = P.expectation(EA.MonteCarlo(20_000, StableRNG(seed)), identity, q)
    @test estimate(1) == estimate(1)
    @test estimate(1) ≈ 4.0 atol = 4 * std(q) / sqrt(20_000)
    @test P.is_expectation_stochastic(EA.MonteCarlo(10), identity, q) === true
    @test P.is_expectation_exact(EA.MonteCarlo(10), identity, q) === false
end

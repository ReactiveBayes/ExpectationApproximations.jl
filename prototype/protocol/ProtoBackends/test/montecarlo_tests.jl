@testitem "MonteCarlo: needs an RNG, which withrng provides" tags = [:montecarlo] setup = [Helpers] begin
    using Distributions, StableRNGs

    @test_throws ArgumentError B.MonteCarlo(0)
    @test P.getrng(B.MonteCarlo(10)) === nothing
    @test_throws ArgumentError P.expectation(B.MonteCarlo(10), identity, Normal(0.0, 1.0))

    rng, other = StableRNG(1), StableRNG(2)
    @test P.getrng(P.withrng(B.MonteCarlo(10), rng)) === rng
    @test P.getrng(P.withrng(B.MonteCarlo(10, other), rng)) === other
end

@testitem "MonteCarlo: stochastic, reproducible with the same RNG" tags = [:montecarlo] setup = [Helpers] begin
    using Distributions, StableRNGs

    q = Gamma(2.0, 2.0)
    estimate(seed) = P.expectation(B.MonteCarlo(20_000, StableRNG(seed)), identity, q)
    @test estimate(1) == estimate(1)
    @test estimate(1) ≈ 4.0 atol = 4 * std(q) / sqrt(20_000)
    @test P.isstochastic(B.MonteCarlo(10), identity, q) === true
    @test P.isexact(B.MonteCarlo(10), identity, q) === false
end

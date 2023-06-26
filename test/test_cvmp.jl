module TestCVMP

using Test
using ExpectationApproximations
using ExponentialFamily
using Distributions
using Random
using StableRNGs
using Optimisers

@testset "CVMP" begin
    @testset "Constructor CVMP method" begin
        cvmp_method = CVMP(
            StableRNG(42),
            1000,
            Optimisers.Adam(),
            ForwardDiffGrad(),
            1000,
            Val(true),
            true,
            SimpleInitStrategy()
        )

        @test cvmp_method.n_iterations == 1000
        @test cvmp_method.n_gradpoints == 1000
        @test cvmp_method.enforce_proper_messages == Val(true)
        @test cvmp_method.warn == true
        @test typeof(cvmp_method.initstrategy) === SimpleInitStrategy
    end

    @testset "Constructor CVMP Factor" begin
        cvmp_factor = CVMPFactor(
            ExponentialFamily.KnownExponentialFamilyDistribution(Bernoulli, 0.7),
            ExponentialFamily.KnownExponentialFamilyDistribution(Bernoulli, 0.9),
            ExponentialFamily.KnownExponentialFamilyDistribution(Bernoulli, 1)
        )

        @test ExpectationApproximations.get_exponential_family(cvmp_factor) == ExponentialFamily.KnownExponentialFamilyDistribution(Bernoulli, 0.7)
        @test ExpectationApproximations.get_positive_measure(cvmp_factor) == ExponentialFamily.KnownExponentialFamilyDistribution(Bernoulli, 0.9)
        @test ExpectationApproximations.get_inmarginal(cvmp_factor) == ExponentialFamily.KnownExponentialFamilyDistribution(Bernoulli, 1)
    end
end

end
module ExpectationApproximationsCVITest

using Test
using Random
using StableRNGs
using Distributions
using Flux

using ExpectationApproximations
using ExponentialFamily
using Distributions
using Optimisers

@testset "CVI" begin

    @testset "Constructor CVI method" begin
        cvi_method = CVI(StableRNG(42), 1000, Optimisers.Adam(), ForwardDiffGrad(), 900, Val(true), true)
        @show cvi_method
        @test cvi_method.n_iterations == 1000
        @test cvi_method.n_gradpoints == 900
        @test cvi_method.enforce_proper_messages == Val(true)
        @test cvi_method.warn == true
    end

    @testset "Constructor CVI Factor" begin
        cvi_factor = CVIFactor(
            ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.7]),
            ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.9]),
        )

        @test ExpectationApproximations.get_exponential_family(cvi_factor) == ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.7])
        @test ExpectationApproximations.get_positive_measure(cvi_factor) == ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.9])
    end

    @testset "cvi Bernoulli x Bernoulli prod" begin
        cvi_method = CVI(StableRNG(42), 5000, Optimisers.Descent(0.001), ForwardDiffGrad(), 60, Val(true), false)

        for i in -1:1, j in -1:1
            positive_measure = ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [i+1e-12])
            ef = ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [j+1e-12])
            product = ExpectationApproximations.approximate_distribution(cvi_method, CVIFactor(ef, positive_measure))
            dist = convert(Distribution, product)
            @test mean(dist) ≈ mean(convert(Distribution, product)) atol=0.3
        end
    end

    @testset "cvi `prod` tests (n_gradpoints = 60)" begin
        rng = StableRNG(42)

        tests = (
            (method = CVI(StableRNG(42), 5000, Optimisers.Descent(0.001), ForwardDiffGrad(), 60, Val(true), false), tol = 5e-1),
        )

        # Check several prods against their analytical solutions
        for test in tests, i in 1:5
            # Univariate `Gamma`
            g1 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)
            g2 = GammaShapeRate(rand(rng) + 1, rand(rng) + 1)

            cvi_factor = CVIFactor(
                convert(ExpectationApproximationsCVITest.ExponentialFamilyDistribution, g1),
                convert(ExpectationApproximationsCVITest.ExponentialFamilyDistribution, g2)
            )

            product = ExpectationApproximations.approximate_distribution(test.method, cvi_factor)
            dist = convert(Distribution, product)
        end
    end

    @testset "Normal x Normal (Log-likelihood preconditioner prod)" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Optimisers.Descent(0.001)
        method = CVI(rng, 5000, optimizer, ForwardDiffGrad(), 1, Val(false), false)

        for i in 1:3
            m_out, m_in = NormalMeanVariance(i, 1), NormalMeanVariance(0, 1)
            cvi_factor = CVIFactor(
                convert(ExpectationApproximationsCVITest.ExponentialFamilyDistribution, m_out),
                convert(ExpectationApproximationsCVITest.ExponentialFamilyDistribution, m_in)
            )

            product = ExpectationApproximations.approximate_distribution(method, cvi_factor)

            @test isapprox(convert(Distribution, product), NormalWeightedMeanPrecision(i, 2), atol = 0.7)
        end
    end

    # @static if VERSION ≥ v"1.7" # Base.@invoke is available only in Julia >= 1.7
    #     @testset "Normal x Normal (Fisher preconditioner prod)" begin
    #         seed = 123
    #         rng = StableRNG(seed)
    #         optimizer = Descent(0.001)
    #         meta = CVI(rng, 50, 5000, optimizer, ForwardDiffGrad(), 1, Val(false), false)

    #         for i in 1:3
    #             m_out, m_in = NormalMeanVariance(i, 1), NormalMeanVariance(0, 1)
    #             λ = Base.@invoke prod(meta::CVI, (ContinuousUnivariateLogPdf((z) -> logpdf(m_out, z)))::AbstractContinuousGenericLogPdf, m_in::Any)
    #             @test isapprox(convert(Distribution, λ), NormalWeightedMeanPrecision(i, 2), atol = 0.5)
    #         end
    #     end
    # end

    # @static if VERSION ≥ v"1.7" # Base.@invoke is available only in Julia >= 1.7
    #     @testset "MvNormal x MvNormal 1D (Fisher preconditioner prod)" begin
    #         seed = 123
    #         rng = StableRNG(seed)
    #         optimizer = Descent(0.001)
    #         meta = CVI(rng, 1, 5000, optimizer, ForwardDiffGrad(), 1, Val(false), false)

    #         for i in 1:3
    #             m_out, m_in = MvNormalMeanCovariance([i], [1]), MvNormalMeanCovariance([0], [1])
    #             λ = Base.@invoke prod(
    #                 meta::CVI, (ContinuousMultivariateLogPdf(ReactiveMP.UnspecifiedDomain(), (x) -> logpdf(m_out, x)))::AbstractContinuousGenericLogPdf, m_in::Any
    #             )
    #             @test isapprox(convert(Distribution, λ), MvNormalWeightedMeanPrecision([i], [2]), atol = 0.5)
    #         end
    #     end

    #     @testset "MvNormal x MvNormal 2D (Fisher preconditioner prod)" begin
    #         seed = 123
    #         rng = StableRNG(seed)
    #         optimizer = Descent(0.001)
    #         meta = CVI(rng, 1, 5000, optimizer, ForwardDiffGrad(), 10, Val(false), false)
    #         for i in 1:3
    #             m_out, m_in = MvGaussianMeanCovariance(fill(i, 2)), MvGaussianMeanCovariance(zeros(2))
    #             g_cvi1 = Base.@invoke prod(meta::CVI, (ContinuousMultivariateLogPdf(2, (x) -> logpdf(m_out, x)))::AbstractContinuousGenericLogPdf, m_in::Any)
    #             g_analytical = MvNormalWeightedMeanPrecision(fill(i, 2), diageye(2) * 2)
    #             @test all(isapprox.(mean(g_analytical), mean(g_cvi1), atol = 0.2))
    #         end
    #     end
    # end
end

end

@testitem "CVMP" begin
    include("./hybrid_setuptests.jl")

    @testset "Constructor CVMP method" begin
        cvmp_method = CVMP(StableRNG(42), 1000, Optimisers.Adam(), ForwardDiffGrad(), 1000, Val(true), true, SimpleInitStrategy())

        @test cvmp_method.n_iterations == 1000
        @test cvmp_method.n_gradpoints == 1000
        @test cvmp_method.enforce_proper_messages == Val(true)
        @test cvmp_method.warn == true
        @test typeof(cvmp_method.initstrategy) === SimpleInitStrategy
    end

    @testset "Constructor CVMP Factor" begin
        cvmp_factor = CVMPFactor(
            ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.7]),
            ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.9]),
            ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [1])
        )

        @test ExpectationApproximations.get_exponential_family(cvmp_factor) == ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.7])
        @test ExpectationApproximations.get_positive_measure(cvmp_factor) == ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [0.9])
        @test ExpectationApproximations.get_marginal(cvmp_factor) == ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [1])
    end

    @testset "compute_gradient" begin
        @test ExpectationApproximations.compute_gradient(ForwardDiffGrad(), (x) -> x^2, 2) == 4
        @test ExpectationApproximations.compute_gradient(ForwardDiffGrad(), identity, 2) == 1
    end

    @testset "total_derivative" begin
        @test ExpectationApproximations.total_derivative(ForwardDiffGrad(), (x) -> x^2, 2) == 8
        @test ExpectationApproximations.total_derivative(ForwardDiffGrad(), identity, 2) == 2
    end

    @testset "cvmp" begin
        @testset "Bernoulli x Bernoulli prod" begin
            cvmp_method = CVMP(StableRNG(42), 500, Optimisers.Adam(0.01), ForwardDiffGrad(), 300, Val(true), true, SimpleInitStrategy())
            for i in -1:1, j in -1:1
                positive_measure = ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [i + 1e-12])
                ef = ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [j + 1e-12])
                marginal = ExponentialFamily.ExponentialFamilyDistribution(Bernoulli, [i + j])
                cvmp_factor = CVMPFactor(ef, positive_measure, marginal)
                product = ExpectationApproximations.approximate_distribution(cvmp_method, cvmp_factor, identity)
                dist = convert(Distribution, product)
                @test mean(dist) ≈ mean(convert(Distribution, marginal)) atol = 0.3
            end
        end

        @testset "Gamma x Gamma prod" begin
            cvmp_method = CVMP(StableRNG(42), 100, Optimisers.Adam(0.01), ForwardDiffGrad(), 10000, Val(true), true, SimpleInitStrategy())
            ef = ExponentialFamily.ExponentialFamilyDistribution(Gamma, [1.0, -1.0])
            positive_measure = ExponentialFamily.ExponentialFamilyDistribution(Gamma, [1.0, -1.0])
            marginal = ExponentialFamily.ExponentialFamilyDistribution(Gamma, [2.0, -2.0])
            cvmp_factor = CVMPFactor(ef, positive_measure, marginal)
            product = ExpectationApproximations.approximate_distribution(cvmp_method, cvmp_factor, identity)
            product_test = ExpectationApproximations.approximate(ExpectationApproximations.AsDistribution(), cvmp_method, cvmp_factor, identity)
            dist = convert(Distribution, product)
            dist_test = convert(Distribution, product_test)
            @test mean(dist) ≈ mean(convert(Distribution, marginal)) atol = 0.4
            @test mean(dist) ≈ mean(dist_test) atol = 0.01
        end
    end
end
